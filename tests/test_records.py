"""The ship's records: what's kept of a wiki page, and pages found by name (from a few
made-up pages, no LM Studio)."""

from app.archive import Archive
from app.records import page_entries, source_of
from conftest import FakeLm

EPISODE = """
<aside class="portable-infobox"><h2>Better Than Life</h2><div>Written by: Somebody Else, a writer of note
and other things besides</div><div>Played by: an actor whose name has no business in Holly's memory</div></aside>
<p>Better Than Life is the second episode of Series II, first broadcast on BBC2 one autumn evening.</p>
<h2>Summary</h2>
<p>The crew plug themselves into a total-immersion game that gives each player whatever they most want,
<sup class="reference">[1]</sup> until it starts to turn their own minds against them.</p>
<h2>Behind the Scenes</h2>
<p>The game's scenes were filmed on location over two wet days, and the set was rebuilt for a later week.</p>
<h2>Cast</h2>
<ul><li>A guest star appeared as the game's cheerful host, in a role written especially for them.</li></ul>
<table><tr><td>A table of ratings for every week of the series, with a column for each channel.</td></tr></table>
"""


def page(*sections: tuple[str, str]) -> str:
    """A page's html: each (section, paragraph); a section of "" is the opening."""
    return "".join((f"<h2>{s}</h2>" if s else "") + f"<p>{text}</p>" for s, text in sections)


def test_a_page_keeps_the_story_and_leaves_out_the_real_world():
    """Only what happens in the show is kept: no infobox, nothing behind the scenes, no
    cast, and an episode's opening (its broadcast) isn't kept either, so he doesn't talk
    about the people who played him."""
    entries = page_entries("RD: Better Than Life", EPISODE)
    assert len(entries) == 1
    assert entries[0].startswith("(Ship's records: Better Than Life (episode), Summary) The crew plug themselves")
    assert "turn their own minds against them" in entries[0]
    for left_out in ("Written by", "Played by", "broadcast", "filmed", "guest star", "ratings", "[1]"):
        assert left_out not in entries[0]


def test_a_page_that_isnt_an_episode_keeps_its_opening():
    entries = page_entries("Dave Lister", page(("", "Dave Lister is the last human alive, three million years "
                                                     "from home, and the ship's only remaining crewman.")))
    assert entries == ["(Ship's records: Dave Lister) Dave Lister is the last human alive, three million "
                       "years from home, and the ship's only remaining crewman."]


def records(cfg, pages: dict[str, str]) -> Archive:
    """The ship's records made from `pages` (title: html), as app.records.build makes them."""
    r = Archive(cfg.recall, FakeLm(), "m", cfg.recall.records_file)
    r.add([(None, e) for title, html in pages.items() for e in page_entries(title, html)])
    return r


PAGES = {
    "RD: Better Than Life": EPISODE,
    "Better Than Life": page(("", "Better Than Life is a game played inside your own head, which makes any "
                                  "wish come true and is very hard to leave.")),
    # Better, than and life, over and over: by keyword alone, this came first.
    "Skutter": page(("", "A skutter is a small maintenance droid; life with them is better than life "
                         "without, they say, and better than life with no droids."),
                    ("History", "Skutters thought life was better than life aboard any other ship, a better "
                                "life, the better life.")),
    # The main page of a person called Lister, and another Lister's, which says it more often.
    "Dave Lister": page(("", "Dave Lister is the last human alive, three million years from home."),
                        ("Biography", "He was a technician, third class, the lowest rank aboard the ship."),
                        ("Personality", "He is easy-going, untidy and fond of curry and lager above all.")),
    "Jim Lister": page(("", "Jim Lister is Lister's son: Lister, Lister and Lister again, as Lister would "
                            "say of a Lister.")),
}


def test_a_page_is_found_by_its_name(cfg):
    """Asked about episodes, he knew almost nothing, though every one was in the records:
    the keyword search found every page saying the words asked, not the one they named."""
    hits = records(cfg, PAGES).search("Better Than Life", question="What happened in Better Than Life?")
    assert [source_of(t) for _, t in hits[:2]] == ["Better Than Life (episode), Summary", "Better Than Life"]


def test_a_surname_finds_the_persons_main_page(cfg):
    """"Lister" names no page: it's the biggest page of a person called so."""
    hits = records(cfg, PAGES).search("Lister", question="Who is Lister?")
    assert source_of(hits[0][1]) == "Dave Lister"
    assert {source_of(t) for _, t in hits[:3]} == {"Dave Lister", "Dave Lister, Biography",
                                                    "Dave Lister, Personality"}
