"""The clock tool: the time here, in UTC, and in a time zone he names."""

import pytest

from app.clock import now


def test_the_time_here_and_in_utc():
    lines = now().splitlines()
    assert len(lines) == 2 and lines[0].startswith("Here: ") and lines[1].startswith("UTC: ")


@pytest.mark.parametrize("zone, offset", [("UTC", "+0"), ("Asia/Tokyo", "+9")])
def test_a_time_zone_adds_a_line_for_it(zone, offset):
    lines = now(zone).splitlines()
    assert len(lines) == 3
    assert lines[2].startswith(f"{zone}: ") and lines[2].endswith(f"(UTC{offset}).")


@pytest.mark.parametrize("zone", ["Mars/Olympus", "New York", "../etc"])
def test_no_such_time_zone_points_to_iana_names_or_the_weather_tool(zone):
    """A zone that isn't one says so, and where to look instead: an IANA name, or the
    weather tool, which gives a place's local time."""
    last = now(zone).splitlines()[-1]
    assert last.startswith(f"There is no time zone called {zone!r}")
    assert "IANA name like 'Europe/Paris'" in last and "the weather tool" in last
