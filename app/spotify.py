"""Holly's Spotify tool: play, pause, skip, what's playing, the volume.

"Holly, play some Talking Heads", "skip this", "turn it down", "what's this
song?": Gemma calls the `spotify` tool, and this drives Spotify's Web API
(spotipy). It controls whichever Spotify is open (the desktop app on this PC,
a phone, a speaker): Spotify does the playing, not Holly. Playback control needs
Spotify Premium.

Signing in: create an app at developer.spotify.com/dashboard (redirect URI
http://127.0.0.1:8888/callback, Web API), put its client ID in the user
environment variable SPOTIFY_CLIENT_ID, then once:

    .venv\\Scripts\\python -m app.spotify login

A browser asks you to allow access; the login is kept in data/spotify_token.json
(git-ignored) and renewed by itself from then on. It uses PKCE, the sign-in for
apps on your own machine, so there's no client secret. Holly never opens the
browser himself: not signed in, he says so.

Trying it without Holly:  python -m app.spotify do play "Talking Heads" artist
                          python -m app.spotify do now_playing
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOauthError, SpotifyPKCE

from .config import SpotifyConfig, load_config
from .search import _user_env

# spotipy logs every refusal ("Restriction violated" when already paused); the tool says it instead.
logging.getLogger("spotipy").setLevel(logging.CRITICAL)

SCOPES = ("user-read-playback-state user-modify-playback-state user-read-currently-playing "
          "playlist-read-private playlist-read-collaborative")

TOOL = {
    "type": "function",
    "function": {
        "name": "spotify",
        "description": "The crew's music on Spotify: play something, pause, resume, skip to the next or "
                       "previous track, say what's playing, change the volume, shuffle, or queue a song.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["play", "pause", "resume", "next", "previous",
                                                      "now_playing", "volume", "shuffle", "queue"]},
                "query": {"type": "string", "description":
                          "For play or queue: what to play, as the crew said it: 'Talking Heads', "
                          "'Once in a Lifetime by Talking Heads', 'my running playlist'."},
                "kind": {"type": "string", "enum": ["track", "artist", "album", "playlist", "liked"],
                         "description": "For play: what the query is. 'liked' is the crew's liked songs "
                                        "(no query needed)."},
                "volume": {"type": "string", "description": "For volume: 0 to 100, or 'up' or 'down'."},
                "on": {"type": "boolean", "description": "For shuffle: on or off."},
            },
            "required": ["action"],
        },
    },
}


class NotReady(Exception):
    """Something the crew has to sort out (sign in, open Spotify); the message says what."""


class _Login:
    """The kept login, renewed when it expires; never the browser sign-in, which
    would wait on a page nobody's looking at. spotipy's client calls this."""

    def __init__(self, pkce: SpotifyPKCE):
        self.pkce = pkce

    def get_access_token(self, as_dict: bool = False):
        try:
            info = self.pkce.validate_token(self.pkce.cache_handler.get_cached_token())
        except SpotifyOauthError as e:  # revoked at Spotify, or the app's gone
            raise NotReady(f"Spotify's login no longer works ({e.error}). Tell the crew to run "
                           "'python -m app.spotify login' again.") from e
        if not info:
            raise NotReady("Spotify isn't signed in. Tell the crew to run 'python -m app.spotify login' "
                           "once, and allow access in the browser.")
        return info if as_dict else info["access_token"]


def _pkce(cfg: SpotifyConfig, open_browser: bool = False) -> SpotifyPKCE:
    client_id = _user_env("SPOTIFY_CLIENT_ID")
    if not client_id:
        raise NotReady("Spotify isn't set up: SPOTIFY_CLIENT_ID isn't set.")
    cfg.token_file.parent.mkdir(parents=True, exist_ok=True)
    return SpotifyPKCE(client_id=client_id, redirect_uri=cfg.redirect_uri, scope=SCOPES,
                       cache_handler=spotipy.CacheFileHandler(cache_path=str(cfg.token_file)),
                       open_browser=open_browser, requests_timeout=cfg.timeout_s)


def available() -> bool:
    """Whether Holly gets the tool: a client ID is set (signing in can come later)."""
    return bool(_user_env("SPOTIFY_CLIENT_ID"))


def _artists(item: dict) -> str:
    names = [a["name"] for a in item.get("artists") or []]
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else (names[0] if names else "")


# Release clutter in names, not worth saying: "- 2024 Remaster", "(Super Deluxe Edition)",
# "[2024 Remaster]". Not "(Naive Melody)", which is part of the title.
_CLUTTER = r"remaster(?:ed)?|edition|version|deluxe|anniversary|expanded|mono|stereo|bonus"
_DASH_CLUTTER = re.compile(r"\s+-\s+[^-]*\b(?:" + _CLUTTER + r")\b.*$", re.IGNORECASE)
_BRACKET_CLUTTER = re.compile(r"\s*[(\[][^)\]]*\b(?:" + _CLUTTER + r")\b[^)\]]*[)\]]", re.IGNORECASE)


def _plain(name: str) -> str:
    return _BRACKET_CLUTTER.sub("", _DASH_CLUTTER.sub("", name or "")).strip() or name


def _describe(item: dict) -> str:
    if item.get("type") == "episode":
        return f"'{item['name']}', from the podcast {item.get('show', {}).get('name', '?')}"
    album = _plain(item.get("album", {}).get("name", ""))
    return f"'{_plain(item['name'])}' by {_artists(item)}" + (f", from {album}" if album else "")


def _mmss(ms: int | None) -> str:
    s = (ms or 0) // 1000
    return f"{s // 60}:{s % 60:02d}"


class Spotify:
    def __init__(self, cfg: SpotifyConfig):
        self.cfg = cfg
        self._sp: spotipy.Spotify | None = None
        self._me: str | None = None

    @property
    def sp(self) -> spotipy.Spotify:
        if self._sp is None:
            self._sp = spotipy.Spotify(auth_manager=_Login(_pkce(self.cfg)),
                                       requests_timeout=self.cfg.timeout_s, retries=1, status_retries=1)
        return self._sp

    # Devices ------------------------------------------------------------------

    def _device(self) -> str | None:
        """The device to play on: None for the one already playing; otherwise the
        configured one, or this PC, or any. With none open, opens the desktop
        app on this PC (if allowed) and waits for it to appear."""
        deadline = time.monotonic() + (self.cfg.launch_wait_s if self.cfg.launch else 0)
        launched = False
        while True:
            devices = [d for d in self.sp.devices().get("devices", []) if not d.get("is_restricted")]
            if any(d.get("is_active") for d in devices):
                return None
            want = self.cfg.device.lower()
            pick = (next((d for d in devices if want and want in d["name"].lower()), None)
                    or next((d for d in devices if d.get("type") == "Computer"), None)
                    or (devices[0] if devices else None))
            if pick:
                return pick["id"]
            if not launched and self.cfg.launch and sys.platform == "win32":
                print("  (opening Spotify)", flush=True)
                os.startfile("spotify:")  # the desktop app's own link
                launched = True
            if time.monotonic() > deadline:
                raise NotReady("Spotify isn't open anywhere: not on this PC, a phone or a speaker. "
                               "Tell the crew to open it, then ask again.")
            time.sleep(1.0)

    # Finding music ------------------------------------------------------------

    def _search(self, query: str, kind: str) -> dict | None:
        found = self.sp.search(q=query, type=kind, limit=5, market=self.cfg.market or None)
        # Spotify always finds something: "zzqxv nonsense" found the Gummy Bear song.
        # Only results sharing a word with what was asked.
        asked = set(re.findall(r"\w+", query.lower()))
        items = [i for i in found.get(kind + "s", {}).get("items", []) if i
                 and asked & set(re.findall(r"\w+", " ".join(
                     [i.get("name", ""), _artists(i), i.get("album", {}).get("name", ""),
                      (i.get("owner") or {}).get("display_name") or ""]).lower()))]
        if not items:
            return None
        if kind == "track" and " by " in query.lower():
            # "Once in a Lifetime by Talking Heads": the one by that artist, if it's there.
            title, _, artist = query.lower().rpartition(" by ")
            return next((i for i in items if artist.strip() in _artists(i).lower()), items[0])
        return items[0]

    def _own_playlist(self, query: str) -> dict | None:
        words = [w for w in query.lower().replace("playlist", " ").split() if w not in ("my", "the")]
        if not words:
            return None
        page = self.sp.current_user_playlists(limit=50)
        for _ in range(4):  # up to 200 playlists
            for p in page.get("items", []):
                if p and all(w in p["name"].lower() for w in words):
                    return p
            if not page.get("next"):
                break
            page = self.sp.next(page)
        return None

    def _find(self, query: str, kind: str | None) -> tuple[str, dict] | None:
        """(kind, item) for what was asked for. With no kind: the artist if the
        name is an artist's, otherwise a track."""
        if kind == "playlist":
            if p := self._own_playlist(query) or self._search(query, "playlist"):
                return "playlist", p
            return None
        if kind in ("track", "artist", "album"):
            item = self._search(query, kind)
            return (kind, item) if item else None
        artist = self._search(query, "artist")
        if artist and artist["name"].lower() == query.lower().strip():
            return "artist", artist
        if track := self._search(query, "track"):
            return "track", track
        return ("artist", artist) if artist else None

    # Actions ------------------------------------------------------------------

    def handle(self, args: dict) -> str:
        """Run one tool call; returns what Holly is told."""
        action = str(args.get("action") or "").strip().lower()
        try:
            run = getattr(self, "_" + action, None) if action in TOOL["function"]["parameters"][
                "properties"]["action"]["enum"] else None
            if not run:
                return f"There's no Spotify action '{action}'."
            return run(args)
        except NotReady as e:
            return str(e)
        except spotipy.SpotifyException as e:
            return self._trouble(e)

    @staticmethod
    def _trouble(e: spotipy.SpotifyException) -> str:
        if e.http_status == 404 and "device" in str(e.msg).lower():
            return "Spotify isn't open anywhere. Tell the crew to open it, then ask again."
        if e.http_status == 403 and "premium" in str(e.msg).lower():
            return "Spotify says that needs Premium."
        if e.http_status == 403 and "restriction" in str(e.msg).lower():  # nothing to go back to...
            return "Spotify wouldn't do that just now."
        if e.http_status == 403:
            return f"Spotify refused ({e.msg}). Tell the crew it didn't work."
        if e.http_status == 401:
            return "Spotify's login has lapsed. Tell the crew to run 'python -m app.spotify login' again."
        return f"Spotify said no ({e.http_status}: {e.msg}). Tell the crew it didn't work."

    def _play(self, args: dict) -> str:
        kind = str(args.get("kind") or "").strip().lower() or None
        query = " ".join(str(args.get("query") or "").split())
        if kind == "liked":
            device = self._device()
            self._me = self._me or self.sp.current_user()["id"]
            self.sp.start_playback(device_id=device, context_uri=f"spotify:user:{self._me}:collection")
            return "Playing the crew's liked songs."
        if not query:
            return self._resume(args)  # "play" alone: carry on
        found = self._find(query, kind)
        if not found:
            return f"Spotify found nothing for '{query}'. Tell the crew."
        kind, item = found
        device = self._device()
        if kind == "track":
            # In its album, from that track, so the music carries on after it.
            self.sp.start_playback(device_id=device, context_uri=item["album"]["uri"],
                                   offset={"uri": item["uri"]})
            what = _describe(item)
        else:
            self.sp.start_playback(device_id=device, context_uri=item["uri"])
            what = {"artist": f"{item['name']} (their most played)",
                    "album": f"the album '{item['name']}' by {_artists(item)}",
                    "playlist": f"the playlist '{item['name']}'"}[kind]
        return f"Now playing {what}."

    def _playing(self) -> dict | None:
        """What's playing, or paused, on the active device; None if nothing is."""
        now = self.sp.current_playback(additional_types="episode")
        return now if now and now.get("device") and now.get("item") else None

    def _pause(self, args: dict) -> str:
        if not (now := self._playing()):
            return "Nothing is playing on Spotify."
        if not now.get("is_playing"):
            return "It's already paused."
        self.sp.pause_playback()
        return "Paused."

    def _resume(self, args: dict) -> str:
        if (now := self._playing()) and now.get("is_playing"):
            return f"It's already playing: {_describe(now['item'])}."
        device = self._device()
        if device:  # nothing playing anywhere: carry on from where it was, here
            self.sp.transfer_playback(device, force_play=True)
        else:
            self.sp.start_playback()
        return "Playing again."

    def _next(self, args: dict) -> str:
        if not self._playing():
            return "Nothing is playing on Spotify, so there's nothing to skip."
        self.sp.next_track()
        return self._after_skip()

    def _previous(self, args: dict) -> str:
        if not self._playing():
            return "Nothing is playing on Spotify, so there's nothing to go back to."
        self.sp.previous_track()
        return self._after_skip()

    def _after_skip(self) -> str:
        time.sleep(0.6)  # Spotify takes a moment to report the new track
        now = self.sp.current_playback(additional_types="episode")
        if now and now.get("item"):
            return f"Skipped. Now playing {_describe(now['item'])}."
        return "Skipped."

    def _now_playing(self, args: dict) -> str:
        if not (now := self._playing()):
            return "Nothing is playing on Spotify."
        item, device = now["item"], now.get("device") or {}
        state = "playing" if now.get("is_playing") else "paused"
        return (f"{_describe(item)}; {state}, {_mmss(now.get('progress_ms'))} of "
                f"{_mmss(item.get('duration_ms'))}, on {device.get('name', '?')}"
                + (f" at volume {device['volume_percent']}%" if device.get("volume_percent") is not None
                   else "") + ".")

    def _volume(self, args: dict) -> str:
        now = self.sp.current_playback() or {}
        device = now.get("device") or {}
        if not device:
            return "Nothing is playing on Spotify, so there's no volume to change."
        if device.get("supports_volume") is False:
            return f"Spotify can't change the volume on {device.get('name', 'that device')}."
        current = device.get("volume_percent") or 0
        said = str(args.get("volume") or "").strip().lower().rstrip("%")
        if said in ("up", "louder"):
            level = current + self.cfg.volume_step
        elif said in ("down", "quieter", "lower"):
            level = current - self.cfg.volume_step
        else:
            try:
                level = round(float(said))
            except ValueError:
                return "The volume needs a number from 0 to 100, or 'up' or 'down'."
        level = max(0, min(100, level))
        self.sp.volume(level)
        return f"Volume {current}% -> {level}%."

    def _shuffle(self, args: dict) -> str:
        on = args.get("on")
        on = True if on is None else (on if isinstance(on, bool) else str(on).lower() in ("true", "on", "yes"))
        if not self._playing():
            return "Nothing is playing on Spotify: shuffle can be set once something is."
        self.sp.shuffle(on)
        return f"Shuffle {'on' if on else 'off'}."

    def _queue(self, args: dict) -> str:
        query = " ".join(str(args.get("query") or "").split())
        if not query or not (track := self._search(query, "track")):
            return f"Spotify found no song for '{query}'."
        self.sp.add_to_queue(track["uri"])
        return f"Queued {_describe(track)}."


def login(cfg: SpotifyConfig) -> None:
    pkce = _pkce(cfg, open_browser=True)
    print("A browser will ask you to allow Holly to control Spotify...", flush=True)
    pkce.get_access_token()
    me = spotipy.Spotify(auth_manager=_Login(pkce)).current_user()
    print(f"Signed in as {me.get('display_name') or me['id']}; "
          f"the login is kept in {cfg.token_file}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Holly's Spotify tool: sign in, or try an action.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="sign in once (opens a browser)")
    sub.add_parser("logout", help="forget the login")
    sub.add_parser("devices", help="list the devices Spotify can play on")
    do = sub.add_parser("do", help="run one action, as Holly would")
    do.add_argument("action")
    do.add_argument("query", nargs="?", default="")
    do.add_argument("kind", nargs="?", default="")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config().spotify
    try:
        if args.command == "login":
            login(cfg)
        elif args.command == "logout":
            Path(cfg.token_file).unlink(missing_ok=True)
            print("Signed out.")
        elif args.command == "devices":
            for d in Spotify(cfg).sp.devices().get("devices", []):
                print(f"{d['name']} ({d['type']}){' - active' if d.get('is_active') else ''}")
        else:
            key = "volume" if args.action == "volume" else "query"
            call = {"action": args.action, key: args.query, "kind": args.kind}
            if args.action == "shuffle":
                call["on"] = args.query.lower() not in ("off", "false", "no")
            print(Spotify(cfg).handle(call))
    except NotReady as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
