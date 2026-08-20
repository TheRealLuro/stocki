"""Reading and validating a session CSV before anything touches the database."""

from datetime import UTC, datetime, timedelta

import pytest

from stocki.ingest.columns import CSV_COLUMNS
from stocki.ingest.reader import read_session, ticker_and_day_from_path
from stocki.ingest.validate import EXPECTED_BARS, validate_session

from .conftest import make_rows

# --- reader ---------------------------------------------------------------


def test_ticker_and_day_come_from_the_path(session_file):
    path = session_file(ticker="NVDA", day=13)

    assert ticker_and_day_from_path(path) == ("NVDA", 13)


def test_reads_every_bar(session_file):
    session = read_session(session_file())

    assert len(session.rows) == EXPECTED_BARS


def test_rows_are_keyed_by_sql_name_not_csv_name(session_file):
    row = read_session(session_file()).rows[0]

    assert "stock_splits" in row
    assert "Stock Splits" not in row


def test_numbers_are_parsed_as_numbers(session_file):
    row = read_session(session_file()).rows[0]

    assert row["close"] == pytest.approx(100.05)
    assert row["volume"] == 1000
    assert isinstance(row["volume"], int)


def test_prices_round_trip_at_full_float64_precision(session_file):
    rows = make_rows()
    rows[0]["close"] = "309.3599853516"

    row = read_session(session_file(rows=rows)).rows[0]

    assert row["close"] == float("309.3599853516")


def test_timestamps_become_aware_datetimes(session_file):
    row = read_session(session_file()).rows[0]

    assert row["timestamp"] == datetime(2026, 8, 5, 13, 30, tzinfo=UTC)


def test_empty_cells_become_none(session_file):
    """NVDA ships an empty news headline; that is NULL, not the string ''."""
    row = read_session(session_file()).rows[0]

    assert row["news_latest_headline"] is None


def test_session_carries_its_provenance(session_file):
    path = session_file(ticker="JPM", day=7)
    session = read_session(path)

    assert session.ticker == "JPM"
    assert session.day == 7
    assert session.source_file.endswith("JPM/day7.csv")


# --- validation -----------------------------------------------------------


def test_a_good_session_has_no_problems(session_file):
    assert validate_session(read_session(session_file())) == []


def _problems(session_file, rows, ticker="AAPL", day=1):
    return validate_session(read_session(session_file(ticker=ticker, day=day, rows=rows)))


def test_rejects_a_short_session(session_file):
    problems = _problems(session_file, make_rows(bars=60))

    assert any("60" in p and "78" in p for p in problems)


def test_rejects_a_gap_in_the_timestamps(session_file):
    rows = make_rows()
    del rows[40]
    rows.append(rows[-1].copy())
    last = datetime.fromisoformat(rows[-2]["timestamp"]) + timedelta(minutes=5)
    rows[-1]["timestamp"] = last.isoformat(sep=" ")

    assert any("spacing" in p.lower() or "gap" in p.lower() for p in _problems(session_file, rows))


def test_rejects_out_of_order_timestamps(session_file):
    rows = make_rows()
    rows[10]["timestamp"], rows[11]["timestamp"] = rows[11]["timestamp"], rows[10]["timestamp"]

    assert _problems(session_file, rows) != []


def test_rejects_high_below_low(session_file):
    rows = make_rows()
    rows[5]["high"] = "1.0"

    assert any("high" in p.lower() for p in _problems(session_file, rows))


def test_rejects_a_close_outside_the_high_low_range(session_file):
    rows = make_rows()
    rows[5]["close"] = "999.0"

    assert _problems(session_file, rows) != []


def test_rejects_negative_volume(session_file):
    rows = make_rows()
    rows[5]["volume"] = "-1"

    assert any("volume" in p.lower() for p in _problems(session_file, rows))


def test_rejects_a_ticker_column_that_disagrees_with_the_filename(session_file):
    rows = make_rows(ticker="TSLA")

    problems = _problems(session_file, rows, ticker="AAPL")

    assert any("ticker" in p.lower() for p in problems)


def test_rejects_bars_from_more_than_one_calendar_day(session_file):
    rows = make_rows()
    rows[-1]["timestamp"] = "2026-08-06 13:30:00+00:00"

    assert _problems(session_file, rows) != []


def test_rejects_a_missing_price(session_file):
    rows = make_rows()
    rows[3]["close"] = ""

    assert any("close" in p.lower() for p in _problems(session_file, rows))


def test_rejects_a_header_that_is_missing_a_column(session_file, tmp_path):
    path = tmp_path / "AAPL" / "day2.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.parent.joinpath("day2.csv")
    good = session_file(ticker="AAPL", day=1).read_text(encoding="utf-8").splitlines()
    good[0] = ",".join(CSV_COLUMNS[:-1])
    text.write_text("\n".join(good), encoding="utf-8")

    problems = validate_session(read_session(text))

    assert any("header" in p.lower() or "column" in p.lower() for p in problems)


def test_problems_name_the_file_so_the_report_is_actionable(session_file):
    rows = make_rows(bars=10)

    problems = _problems(session_file, rows, ticker="META", day=9)

    assert problems and all(isinstance(p, str) and p for p in problems)
