"""Loading sessions into Postgres, and proving the table matches the CSVs."""


import pytest

from stocki.errors import StockiConnectionError
from stocki.ingest.load import ingest_directory, ingest_session, verify_directory
from stocki.ingest.reader import read_session

from .conftest import make_rows, write_session

pytestmark = pytest.mark.db


def _count(db, sql="SELECT count(*) FROM bars_raw"):
    return db.execute(sql).fetchone()[0]


def test_connection_failure_says_how_to_fix_it():
    from dataclasses import replace

    from stocki.config import get_settings
    from stocki.db.session import connect

    dead = replace(get_settings(), host="127.0.0.1", port=1, dsn_override=None)

    with pytest.raises(StockiConnectionError) as caught:
        connect(dead)

    assert "docker compose up" in str(caught.value)


def test_loads_every_bar(db, tmp_path):
    path = write_session(tmp_path, ticker="AAPL", day=13)

    written = ingest_session(db, read_session(path))

    assert written == 78
    assert _count(db) == 78


def test_provenance_columns_are_filled_in(db, tmp_path):
    ingest_session(db, read_session(write_session(tmp_path, ticker="JPM", day=7)))

    row = db.execute(
        "SELECT day, source_file, ingested_at FROM bars_raw LIMIT 1"
    ).fetchone()

    assert row[0] == 7
    assert row[1] == "JPM/day7.csv"
    assert row[2] is not None


def test_ingesting_twice_does_not_duplicate_rows(db, tmp_path):
    path = write_session(tmp_path, ticker="AAPL", day=13)

    ingest_session(db, read_session(path))
    ingest_session(db, read_session(path))

    assert _count(db) == 78


def test_reingesting_a_corrected_file_updates_the_values(db, tmp_path):
    path = write_session(tmp_path, ticker="AAPL", day=13)
    ingest_session(db, read_session(path))

    rows = make_rows(day=13)
    rows[0]["close"] = "999.5"
    write_session(tmp_path, ticker="AAPL", day=13, rows=rows)
    ingest_session(db, read_session(path))

    first = db.execute("SELECT close FROM bars_raw ORDER BY timestamp LIMIT 1").fetchone()[0]
    assert first == 999.5
    assert _count(db) == 78


def test_a_bad_file_writes_nothing_at_all(db, tmp_path):
    """Quarantine must be all-or-nothing: a rejected file leaves no partial rows."""
    bad = make_rows()
    bad[70]["high"] = "0.01"
    write_session(tmp_path, ticker="AAPL", day=1, rows=bad)

    report = ingest_directory(db, tmp_path)

    assert _count(db) == 0
    assert report.quarantined
    assert not report.ok


def test_good_files_still_load_when_a_sibling_is_quarantined(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=1)
    write_session(tmp_path, ticker="TSLA", day=1, rows=make_rows(bars=12))

    report = ingest_directory(db, tmp_path)

    assert _count(db) == 78
    assert report.loaded == ["AAPL/day1.csv"]
    assert "TSLA/day1.csv" in report.quarantined


def test_the_table_round_trips_the_csv_exactly(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=13)
    ingest_directory(db, tmp_path)

    assert verify_directory(db, tmp_path) == []


def test_verify_reports_a_row_that_drifted_from_the_file(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=13)
    ingest_directory(db, tmp_path)

    db.execute(
        "UPDATE bars_raw SET close = close + 1 "
        "WHERE timestamp = (SELECT min(timestamp) FROM bars_raw)"
    )
    db.commit()

    assert verify_directory(db, tmp_path) != []


def test_verify_reports_a_file_that_was_never_loaded(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=13)

    problems = verify_directory(db, tmp_path)

    assert any("AAPL/day13.csv" in p for p in problems)


# --- views ----------------------------------------------------------------


def test_fundamentals_view_collapses_the_repeated_rows(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=1)
    write_session(tmp_path, ticker="AAPL", day=2)
    write_session(tmp_path, ticker="NVDA", day=1)
    ingest_directory(db, tmp_path)

    assert _count(db, "SELECT count(*) FROM bars_raw") == 234
    assert _count(db, "SELECT count(*) FROM v_fundamentals") == 3


def test_coverage_view_reports_bars_per_session(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=1)
    ingest_directory(db, tmp_path)

    row = db.execute("SELECT ticker, day, bar_count FROM v_coverage").fetchone()

    assert row == ("AAPL", 1, 78)


def test_bars_view_exposes_only_the_time_series_columns(db, tmp_path):
    write_session(tmp_path, ticker="AAPL", day=1)
    ingest_directory(db, tmp_path)

    columns = [d.name for d in db.execute("SELECT * FROM v_bars LIMIT 1").description]

    assert columns == ["ticker", "day", "timestamp", "open", "high", "low", "close", "volume"]


# --- the read-only role ---------------------------------------------------


def test_the_api_role_can_read(db, db_settings, tmp_path):
    from dataclasses import replace

    from stocki.db.session import connect

    write_session(tmp_path, ticker="AAPL", day=1)
    ingest_directory(db, tmp_path)
    db.commit()

    readonly = replace(db_settings, user=db_settings.ro_user, password=db_settings.ro_password)
    with connect(readonly) as conn:
        assert conn.execute("SELECT count(*) FROM bars_raw").fetchone()[0] == 78


def test_the_api_role_cannot_write(db, db_settings, tmp_path):
    """Even a successful injection through the API only reads."""
    from dataclasses import replace

    import psycopg

    from stocki.db.session import connect

    write_session(tmp_path, ticker="AAPL", day=1)
    ingest_directory(db, tmp_path)
    db.commit()

    readonly = replace(db_settings, user=db_settings.ro_user, password=db_settings.ro_password)
    with connect(readonly) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM bars_raw")
