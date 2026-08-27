"""The market clock and the shared trading-day numbering.

Both are load-bearing in ways that fail quietly if they are wrong: an hour of
timezone drift collides on the `(ticker, timestamp)` primary key, and a day
number that means two different dates makes the chronological train/test split
stop being chronological without raising anything.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocki.ingest.reader import ticker_and_day_from_path
from stocki.live.sessions import (
    BARS_PER_SESSION,
    DayIndex,
    DayNumberingError,
    eastern_offset,
    eastern_to_utc,
    expected_timestamps,
    news_window,
    recent_weekdays,
    session_window,
)

from .conftest import write_session

REPO_DATA = Path(__file__).resolve().parents[2] / "data"


# --- the clock ------------------------------------------------------------


def test_summer_session_starts_at_1330z():
    """Which is what every committed file holds: 2026-08-05 13:30:00+00:00."""
    opened = eastern_to_utc(datetime(2026, 8, 5, 9, 30))

    assert opened == datetime(2026, 8, 5, 13, 30, tzinfo=UTC)


def test_winter_session_starts_at_1430z():
    """Standard time is an hour further from UTC. The data folder happens to
    hold only summer sessions, so nothing else would catch this."""
    opened = eastern_to_utc(datetime(2026, 1, 5, 9, 30))

    assert opened == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    "moment,hours",
    [
        (datetime(2026, 3, 7, 12, 0), -5),  # the day before the switch
        (datetime(2026, 3, 8, 12, 0), -4),  # second Sunday in March
        (datetime(2026, 10, 31, 12, 0), -4),
        (datetime(2026, 11, 1, 12, 0), -5),  # first Sunday in November
    ],
)
def test_daylight_saving_boundaries(moment, hours):
    assert eastern_offset(moment).total_seconds() == hours * 3600


def test_a_session_is_78_bars_from_1330z_to_1955z():
    stamps = expected_timestamps(date(2026, 8, 5))

    assert len(stamps) == BARS_PER_SESSION
    assert stamps[0] == datetime(2026, 8, 5, 13, 30, tzinfo=UTC)
    assert stamps[-1] == datetime(2026, 8, 5, 19, 55, tzinfo=UTC)


def test_the_session_window_ends_after_the_last_bar():
    """Half-open, so `start <= t < end` keeps the 19:55 bar and nothing later."""
    start, end = session_window(date(2026, 8, 5))

    assert start == datetime(2026, 8, 5, 13, 30, tzinfo=UTC)
    assert end == datetime(2026, 8, 5, 20, 0, tzinfo=UTC)


def test_the_news_window_opens_at_midnight_and_shuts_at_the_bell():
    """Wider than the session at the front, because pre-market headlines are the
    ones a session reacts to -- and not a minute wider at the back, because
    news_count is a model input and later news would be look-ahead."""
    start, end = news_window(date(2026, 8, 25))

    assert start == datetime(2026, 8, 25, 4, 0, tzinfo=UTC)  # 00:00 Eastern
    assert end == datetime(2026, 8, 25, 20, 0, tzinfo=UTC)  # 16:00 Eastern
    assert end == session_window(date(2026, 8, 25))[1]


def test_recent_weekdays_skips_the_weekend():
    """Monday 2026-08-24 looking back three days reaches the Thursday."""
    days = recent_weekdays(3, ending=date(2026, 8, 24))

    assert days == [date(2026, 8, 20), date(2026, 8, 21), date(2026, 8, 24)]


# --- the day index --------------------------------------------------------


def test_index_recovers_the_dates_from_filenames(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    write_session(tmp_path, ticker="NVDA", day=2)

    index = DayIndex.from_data_dir(tmp_path)

    assert index.last_number == 2
    assert set(index.dates.values()) == {1, 2}


def test_a_new_date_gets_the_next_number(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    index = DayIndex.from_data_dir(tmp_path)

    number = index.assign(index.last_date + timedelta(days=1))

    assert number == 2


def test_the_same_date_keeps_its_number(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    index = DayIndex.from_data_dir(tmp_path)
    known = index.last_date

    assert index.assign(known) == 1
    assert index.assign(known) == 1  # idempotent, so a re-fetch reuses the file


def test_a_date_inside_the_collected_range_is_refused(tmp_path):
    """Handing it `last + 1` would put a higher number on an earlier date, and
    every chronological split downstream would silently be wrong."""
    write_session(tmp_path, ticker="NVDA", day=1)
    write_session(tmp_path, ticker="NVDA", day=5)
    index = DayIndex.from_data_dir(tmp_path)

    with pytest.raises(DayNumberingError, match="inside the range already collected"):
        index.assign(date(2020, 1, 1))


def test_one_day_number_cannot_mean_two_dates(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=3)
    drifted = write_session(tmp_path, ticker="AAPL", day=3)

    # Move AAPL's day 3 to a different calendar date than NVDA's.
    text = drifted.read_text(encoding="utf-8").replace("2026-08-07", "2026-09-07")
    drifted.write_text(text, encoding="utf-8")

    with pytest.raises(DayNumberingError, match="must mean one date"):
        DayIndex.from_data_dir(tmp_path)


def test_the_real_data_folder_is_consistently_numbered():
    """day1 is 2026-07-20 and day20 is 2026-08-14 for every ticker that has
    them -- the property `stocki fetch` has to preserve."""
    if not (REPO_DATA / "AMZN" / "day1.csv").is_file():
        pytest.skip(f"no session files under {REPO_DATA}")

    index = DayIndex.from_data_dir(REPO_DATA)

    assert index.days[1] == date(2026, 7, 20)
    assert index.days[20] == date(2026, 8, 14)
    assert index.last_number == 20
    assert index.assign(date(2026, 8, 17)) == 21


def test_index_ignores_files_that_are_not_sessions(tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    (tmp_path / "NVDA" / "notes.csv").write_text("hello", encoding="utf-8")

    index = DayIndex.from_data_dir(tmp_path)

    assert index.last_number == 1
    with pytest.raises(ValueError):
        ticker_and_day_from_path(tmp_path / "NVDA" / "notes.csv")
