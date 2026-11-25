"""The weather tool's place lookup: which of the geocoder's matches he's given, with the
geocoder mocked (no network)."""

import httpx
import pytest

from app import weather
from app.weather import find_place, forecast


def place(name: str, cc: str, country: str, admin1: str = "", admin2: str = "") -> dict:
    return {"name": name, "country_code": cc, "country": country, "admin1": admin1, "admin2": admin2,
            "latitude": 1.0, "longitude": 2.0}


NEW_YORK_LINCS = place("New York", "GB", "United Kingdom", "England", "Lincolnshire")
NEW_YORK_US = place("New York", "US", "United States", "New York")
KESWICK_ON = place("Keswick", "CA", "Canada", "Ontario")
KESWICK_VA = place("Keswick", "US", "United States", "Virginia")
KESWICK = place("Keswick", "GB", "United Kingdom", "England", "Cumbria")


def geocoder(monkeypatch, places: dict[str, list[dict]]) -> list[str]:
    """Answer weather's requests from `places`, by the name asked (any other name finds
    nothing, as the geocoder says it: no "results" at all). Returns what was asked for, in
    order: each name, and "forecast" for a forecast."""
    asked: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host != "geocoding-api.open-meteo.com":
            asked.append("forecast")
            return httpx.Response(500)
        name = request.url.params["name"]
        asked.append(name)
        return httpx.Response(200, json={"results": places[name]} if places.get(name) else {"generationtime_ms": 0.2})

    monkeypatch.setattr(weather.httpx, "get", httpx.Client(transport=httpx.MockTransport(handle)).get)
    return asked


@pytest.mark.parametrize("country", ["USA", "US", "U.S.A.", "America", "United States"])
def test_new_york_usa_is_the_american_one(cfg, monkeypatch, country):
    """"New York, USA" found New York, Lincolnshire: "usa" is in no place's details, so
    nothing matched, and the home country's namesake was taken."""
    geocoder(monkeypatch, {"New York": [NEW_YORK_LINCS, NEW_YORK_US]})
    assert find_place(f"New York, {country}", cfg.search)["country_code"] == "US"


def test_a_two_letter_country_code_counts(cfg, monkeypatch):
    """"de" is in none of Berlin, Germany's details: it's the country code that finds it."""
    berlin_nh = place("Berlin", "US", "United States", "New Hampshire")
    berlin = place("Berlin", "DE", "Germany", "Land Berlin")
    geocoder(monkeypatch, {"Berlin": [berlin_nh, berlin]})
    assert find_place("Berlin, DE", cfg.search) == berlin


def test_a_bare_name_prefers_the_home_country(cfg, monkeypatch):
    """The geocoder gives matches by population: a name alone found an American namesake
    before the British place."""
    geocoder(monkeypatch, {"Keswick": [KESWICK_ON, KESWICK_VA, KESWICK]})
    assert cfg.search.home_country == "GB"
    assert find_place("Keswick", cfg.search) == KESWICK


def test_the_x_is_tried_as_x_when_nothing_is_found(cfg, monkeypatch):
    """"The Lizard" knows nothing by that name; "Lizard" does, and the home country's
    comes before Lizard Island, Australia."""
    lizard_island = place("Lizard Island", "AU", "Australia", "Queensland")
    the_lizard = place("Lizard", "GB", "United Kingdom", "England", "Cornwall")
    the_hague = place("The Hague", "NL", "Netherlands", "South Holland")
    asked = geocoder(monkeypatch, {"Lizard": [lizard_island, the_lizard], "The Hague": [the_hague]})
    assert find_place("The Lizard", cfg.search) == the_lizard
    assert asked == ["The Lizard", "Lizard"]
    # Found as it was given: no second try.
    asked.clear()
    assert find_place("The Hague", cfg.search) == the_hague and asked == ["The Hague"]


def test_a_county_after_the_comma_picks_the_place_in_it(cfg, monkeypatch):
    geocoder(monkeypatch, {"Keswick": [KESWICK_ON, KESWICK_VA, KESWICK]})
    assert find_place("Keswick, Cumbria", cfg.search) == KESWICK
    assert find_place("Keswick, Virginia", cfg.search) == KESWICK_VA
    # A county or country that none of them is in: the home country's, if there is one.
    assert find_place("Keswick, Narnia", cfg.search) == KESWICK


def test_a_county_alone_sends_him_to_its_main_town(cfg, monkeypatch):
    """The geocoder knows towns and villages, not counties. Offered "or tell the crew you
    couldn't find out", he always took it: he's told to call again with the nearest town."""
    asked = geocoder(monkeypatch, {"Keswick": [KESWICK]})
    found = forecast("Cumbria", cfg.search)
    assert found.results == 0
    assert "doesn't know a place called Cumbria" in found.answer
    assert "Call the weather tool again now, with the nearest town (for a county, its main town)" in found.answer
    assert "Don't guess the weather" in found.answer
    assert asked == ["Cumbria"]  # no forecast asked for
