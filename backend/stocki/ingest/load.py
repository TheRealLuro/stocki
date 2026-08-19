"""Load session files into Postgres.

Every file is validated in full before a single row is written, so a rejected
file is quarantined rather than half-loaded, and one bad file never stops its
siblings from loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from .columns import CSV_COLUMNS, sql_name, table_columns
from .reader import Session, find_sessions, read_session
from .validate import EXPECTED_BARS, validate_session

CSV_SQL_COLUMNS = tuple(sql_name(c) for c in CSV_COLUMNS)
KEY_COLUMNS = ("ticker", "timestamp")


def _insert_sql() -> str:
    columns = table_columns()
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in columns if c not in KEY_COLUMNS
    )
    return (
        f"INSERT INTO bars_raw ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(KEY_COLUMNS)}) DO UPDATE SET {updates}"
    )


INSERT_SQL = _insert_sql()


@dataclass
class IngestReport:
    """What happened during an ingest run."""

    loaded: list[str] = field(default_factory=list)
    quarantined: dict[str, list[str]] = field(default_factory=dict)
    rows_written: int = 0

    @property
    def ok(self) -> bool:
        return not self.quarantined

    def summary(self) -> str:
        lines = [
            f"loaded {len(self.loaded)} session files ({self.rows_written} bars)",
        ]
        if self.quarantined:
            lines.append(f"quarantined {len(self.quarantined)}:")
            for _, problems in sorted(self.quarantined.items()):
                for problem in problems:
                    lines.append(f"  {problem}")
        return "\n".join(lines)


def ingest_session(conn: psycopg.Connection, session: Session) -> int:
    """Write an already-validated session. Use `ingest_directory` to validate first."""
    stamped = datetime.now(UTC)
    rows = [
        [row.get(column) for column in CSV_SQL_COLUMNS]
        + [session.day, session.source_file, stamped]
        for row in session.rows
    ]
    with conn.transaction():
        conn.cursor().executemany(INSERT_SQL, rows)
    return len(rows)


def ingest_directory(
    conn: psycopg.Connection,
    data_dir: Path | str,
    *,
    expected_bars: int = EXPECTED_BARS,
) -> IngestReport:
    """Validate then load every data/<TICKER>/day<N>.csv under `data_dir`."""
    report = IngestReport()
    for path in find_sessions(data_dir):
        session = read_session(path)
        problems = validate_session(session, expected_bars=expected_bars)
        if problems:
            report.quarantined[session.source_file] = problems
            continue
        report.rows_written += ingest_session(conn, session)
        report.loaded.append(session.source_file)
    return report


def verify_directory(conn: psycopg.Connection, data_dir: Path | str) -> list[str]:
    """Compare every loaded row against its source CSV, cell by cell.

    If this returns an empty list, the table is the CSVs.
    """
    select = (
        f"SELECT {', '.join(CSV_SQL_COLUMNS)} FROM bars_raw "
        "WHERE ticker = %s AND day = %s ORDER BY timestamp"
    )
    problems: list[str] = []

    for path in find_sessions(data_dir):
        session = read_session(path)
        where = session.source_file
        stored = conn.execute(select, (session.ticker, session.day)).fetchall()

        if not stored:
            problems.append(f"{where}: present on disk but not loaded into bars_raw")
            continue
        if len(stored) != len(session.rows):
            problems.append(
                f"{where}: {len(stored)} rows in bars_raw, {len(session.rows)} in the file"
            )
            continue

        for index, (db_row, file_row) in enumerate(zip(stored, session.rows, strict=True)):
            expected = tuple(file_row.get(column) for column in CSV_SQL_COLUMNS)
            if tuple(db_row) == expected:
                continue
            # Only walk the 118 columns for a row that actually differs.
            for column, stored_value, wanted in zip(
                CSV_SQL_COLUMNS, db_row, expected, strict=True
            ):
                if stored_value != wanted:
                    problems.append(
                        f"{where}: row {index} column {column} is {stored_value!r} "
                        f"in bars_raw but {wanted!r} in the file"
                    )
    return problems
