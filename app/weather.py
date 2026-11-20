"""Holly's weather forecasts, from Open-Meteo (free, no key, open-meteo.com).

Web search can't do the weather: the results are links to weather sites,
and the forecasts are on their pages, so Holly could only say he couldn't
find out. Open-Meteo looks the place up by name, then returns the forecast
as numbers (~0.4 s each), which go to him as a few lines of plain facts.
"""

from __future__ import annotations

import datetime
import time

import httpx

from .config import SearchConfig
from .search import Lookup

TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The weather now and the forecast for the next week, for one place.",
        "parameters": {
            "type": "object",
            "properties": {"place": {"type": "string", "description":
                "A town or village, with its county or country: 'Keswick, Cumbria', "
                "'Paris, France'. For a beach, landmark or county, use the nearest town."}},
            "required": ["place"],
        },
    },
}

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAYS = 7

# WMO weather codes, as Open-Meteo reports them.
CONDITIONS = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}

# Ways of naming the UK that don't appear in the geocoder's place details.
COUNTRY_NAMES = {"uk": "GB", "united kingdom": "GB", "britain": "GB", "great britain": "GB"}


def find_place(place: str, cfg: SearchConfig) -> dict | None:
    """The best geocoder match for 'name, county or country', or None.

    The geocoder takes a bare name and returns matches by population, so a
    name alone found an American namesake before the British place.
    With a county or country after the comma, matches are filtered by it;
    without one, places in the home country come first. It knows towns,
    villages and parks, but not counties ("Cumbria") or most beaches."""
    name, _, hint = place.partition(",")
    name, hint = name.strip(), hint.strip().lower()
    names = [name] + ([name[4:]] if name.lower().startswith("the ") else [])  # "The Hague": "Hague"
    for n in names:
        resp = httpx.get(GEOCODE, params={"name": n, "count": 10, "language": "en", "format": "json"},
                         timeout=cfg.timeout_s)
        resp.raise_for_status()
        hits = resp.json().get("results") or []
        if hint:
            code = COUNTRY_NAMES.get(hint)
            matching = [h for h in hits if (code and h.get("country_code") == code) or any(
                hint in (h.get(k) or "").lower() for k in ("admin1", "admin2", "admin3", "admin4", "country"))]
            hits = matching or [h for h in hits if h.get("country_code") == cfg.home_country]
        else:
            hits.sort(key=lambda h: h.get("country_code") != cfg.home_country)
        if hits:
            return hits[0]
    return None


def _day_name(day: datetime.date, today: datetime.date) -> str:
    label = day.strftime("%A %d %B").replace(" 0", " ")
    offset = (day - today).days
    return {0: f"Today, {label}", 1: f"Tomorrow, {label}"}.get(offset, label)


def forecast(place: str, cfg: SearchConfig) -> Lookup:
    """The forecast for `place`, as a few lines of facts for Holly."""
    t0 = time.perf_counter()

    def done(answer: str, found: bool) -> Lookup:
        return Lookup(place, answer, int(found), (time.perf_counter() - t0) * 1000, 0)

    try:
        where = find_place(place, cfg)
        if not where:
            # Offered "or tell the crew you couldn't find out", he always took it.
            return done(f"The weather service doesn't know a place called {place}: it knows towns and "
                        "villages, not counties, beaches or landmarks. Call the weather tool again now, "
                        "with the nearest town (for a county, its main town). Don't guess the weather.",
                        False)
        resp = httpx.get(FORECAST, timeout=cfg.timeout_s, params={
            "latitude": where["latitude"], "longitude": where["longitude"], "timezone": "auto",
            "forecast_days": DAYS, "wind_speed_unit": "mph",
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max",
        })
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return done(f"The weather service didn't answer ({type(e).__name__}). Tell the crew you "
                    "couldn't find out. Don't guess the weather.", False)

    parts: list[str] = []  # "Keswick, Cumbria, England, United Kingdom", without repeats
    for x in (where.get("name"), where.get("admin2"), where.get("admin1"), where.get("country")):
        if x and x not in parts:
            parts.append(x)
    label = ", ".join(parts)
    now, daily = data["current"], data["daily"]
    lines = [f"Weather for {label}. Temperatures in degrees Celsius, wind in miles an hour.",
             f"Right now: {CONDITIONS.get(now['weather_code'], 'unknown')}, "
             f"{now['temperature_2m']:.0f} degrees, wind {now['wind_speed_10m']:.0f}."]
    today = datetime.date.fromisoformat(daily["time"][0])
    for i, day in enumerate(daily["time"]):
        rain = daily["precipitation_probability_max"][i]
        lines.append(
            f"{_day_name(datetime.date.fromisoformat(day), today)}: "
            f"{CONDITIONS.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]:.0f} to {daily['temperature_2m_max'][i]:.0f} degrees, "
            + (f"{rain:.0f} per cent chance of rain, " if rain is not None else "")
            + f"wind up to {daily['wind_speed_10m_max'][i]:.0f}, gusts {daily['wind_gusts_10m_max'][i]:.0f}.")
    return done("\n".join(lines), True)
