"""Shared fixtures: synthetic session files and a live-Postgres guard."""

import csv
from datetime import UTC, datetime, timedelta

import pytest

from stocki.ingest.columns import CSV_COLUMNS, TEXT_COLUMNS, sql_name

BAR_INTERVAL = timedelta(minutes=5)
SESSION_START = datetime(2026, 8, 5, 13, 30, tzinfo=UTC)


def make_rows(ticker="AAPL", day=1, bars=78, start=None, interval=BAR_INTERVAL):
    """A well-formed session: 5-minute bars with sane OHLC relationships.

    Each day gets its own calendar date, the way the real files do -- otherwise
    two days of the same ticker would collide on the (ticker, timestamp) key.
    """
    if start is None:
        start = SESSION_START + timedelta(days=day - 1)
    rows = []
    for i in range(bars):
        price = 100.0 + i * 0.1
        row = {}
        for column in CSV_COLUMNS:
            if sql_name(column) in TEXT_COLUMNS:
                row[column] = ""
            else:
                row[column] = "1.0"
        row["timestamp"] = (start + interval * i).isoformat(sep=" ")
        row["open"] = f"{price:.10f}"
        row["close"] = f"{price + 0.05:.10f}"
        row["high"] = f"{price + 0.20:.10f}"
        row["low"] = f"{price - 0.20:.10f}"
        row["volume"] = str(1000 + i)
        row["ticker"] = ticker
        row["thscode"] = ticker
        row["thsname_en"] = f"{ticker} Inc."
        row["currency"] = "USD"
        row["hour"] = str((start + interval * i).hour)
        row["minute"] = str((start + interval * i).minute)
        row["news_count"] = "3"
        rows.append(row)
    return rows


def write_session(directory, ticker="AAPL", day=1, rows=None):
    """Write rows to <directory>/<TICKER>/day<N>.csv and return the path."""
    rows = make_rows(ticker, day) if rows is None else rows
    folder = directory / ticker
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"day{day}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def session_file(tmp_path):
    """Factory: session_file(ticker=..., day=..., rows=...) -> Path."""

    def _make(ticker="AAPL", day=1, rows=None):
        return write_session(tmp_path, ticker=ticker, day=day, rows=rows)

    return _make


@pytest.fixture(scope="session")
def db_settings():
    """Settings pointed at a throwaway `stocki_test` database, or skip."""
    from dataclasses import replace

    import psycopg

    from stocki.config import get_settings

    base = replace(get_settings(), dsn_override=None)
    try:
        with psycopg.connect(base.dsn, autocommit=True, connect_timeout=3) as conn:
            found = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = 'stocki_test'"
            ).fetchone()
            if not found:
                conn.execute("CREATE DATABASE stocki_test")
    except psycopg.OperationalError as exc:
        pytest.skip(f"no Postgres at {base.address}: {exc}")
    return replace(base, database="stocki_test")


@pytest.fixture
def db(db_settings):
    """A connection to an empty, schema-applied test database."""
    from stocki.db.session import apply_schema, connect

    conn = connect(db_settings)
    apply_schema(conn)
    conn.execute("TRUNCATE bars_raw")
    conn.commit()
    yield conn
    conn.close()
