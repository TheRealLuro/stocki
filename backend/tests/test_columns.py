"""The CSV -> Postgres column contract.

The whole point of `bars_raw` is that it mirrors the CSVs. These tests are the
drift detector: if the data files ever change shape, they fail here rather than
silently loading a different dataset.
"""

import csv
import re

import pytest

from stocki.config import get_settings
from stocki.ingest.columns import (
    CSV_COLUMNS,
    PROVENANCE_COLUMNS,
    RENAMED,
    SCHEMA_SQL_PATH,
    pg_type,
    sql_name,
    table_columns,
)


def _data_files():
    files = sorted(get_settings().data_dir.glob("*/*.csv"))
    if not files:
        pytest.skip("no data/<TICKER>/*.csv files present")
    return files


def _header_of(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return tuple(next(csv.reader(fh)))


def test_csv_columns_match_the_files_on_disk():
    assert CSV_COLUMNS == _header_of(_data_files()[0])


def test_there_are_118_csv_columns():
    assert len(CSV_COLUMNS) == 118


def test_every_data_file_shares_one_header():
    mismatched = [p for p in _data_files() if _header_of(p) != CSV_COLUMNS]
    assert mismatched == []


def test_sql_name_fixes_the_two_illegal_identifiers():
    assert sql_name("Dividends") == "dividends"
    assert sql_name("Stock Splits") == "stock_splits"


def test_sql_name_leaves_every_other_column_untouched():
    untouched = [c for c in CSV_COLUMNS if c not in RENAMED]
    assert [sql_name(c) for c in untouched] == untouched


def test_every_sql_name_is_a_bare_lowercase_identifier():
    """No column should ever need double-quoting in a query."""
    bad = [c for c in CSV_COLUMNS if not re.fullmatch(r"[a-z_][a-z0-9_]*", sql_name(c))]
    assert bad == []


def test_table_appends_provenance_after_the_csv_columns():
    cols = table_columns()

    assert cols[:118] == tuple(sql_name(c) for c in CSV_COLUMNS)
    assert cols[118:] == PROVENANCE_COLUMNS
    assert len(cols) == 121


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("timestamp", "timestamptz"),
        ("volume", "bigint"),
        ("hour", "smallint"),
        ("minute", "smallint"),
        ("news_count", "integer"),
        ("ticker", "text"),
        ("thsname_cn", "text"),
        ("news_latest_headline", "text"),
        ("close", "double precision"),
        ("si_market_cap", "double precision"),
        ("stock_splits", "double precision"),
        ("day", "smallint"),
        ("source_file", "text"),
        ("ingested_at", "timestamptz"),
    ],
)
def test_pg_type_mapping(column, expected):
    assert pg_type(column) == expected


def test_prices_are_double_precision_so_csv_floats_round_trip_exactly():
    """309.3599853516 must come back bit-identical; numeric/real would not do that."""
    for column in ("open", "high", "low", "close"):
        assert pg_type(column) == "double precision"


def _schema_column_order():
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    body = re.search(r"CREATE TABLE IF NOT EXISTS bars_raw\s*\((.*?)\n\);", sql, re.S).group(1)
    names = []
    for line in body.splitlines():
        match = re.match(r"\s{4}([a-z_][a-z0-9_]*)\s+\S", line)
        if match and match.group(1) not in {"primary", "constraint"}:
            names.append(match.group(1))
    return tuple(names)


def test_schema_sql_declares_every_table_column_in_order():
    assert _schema_column_order() == table_columns()
