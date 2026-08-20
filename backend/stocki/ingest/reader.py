"""Read one session CSV into typed rows.

The day number lives only in the filename (day13.csv), and the ticker only in
the folder name, so both are recovered from the path and carried on the Session.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .columns import (
    BIGINT_COLUMNS,
    CSV_COLUMNS,
    INTEGER_COLUMNS,
    SMALLINT_COLUMNS,
    TEXT_COLUMNS,
    TIMESTAMPTZ_COLUMNS,
    sql_name,
)

DAY_PATTERN = re.compile(r"day(\d+)\.csv$", re.IGNORECASE)
INT_COLUMNS = BIGINT_COLUMNS | INTEGER_COLUMNS | SMALLINT_COLUMNS


class UnreadableSession(ValueError):
    """The path is not a recognisable data/<TICKER>/day<N>.csv."""


@dataclass(frozen=True)
class Session:
    """One data/<TICKER>/day<N>.csv, parsed but not yet validated."""

    ticker: str
    day: int
    path: Path
    header: tuple[str, ...]
    rows: list[dict[str, object]]
    problems: tuple[str, ...] = field(default=())

    @property
    def source_file(self) -> str:
        """Provenance string stored on every row, e.g. AAPL/day13.csv."""
        return f"{self.ticker}/{self.path.name}"


def ticker_and_day_from_path(path: Path | str) -> tuple[str, int]:
    path = Path(path)
    match = DAY_PATTERN.search(path.name)
    if not match:
        raise UnreadableSession(f"{path.name} is not named day<N>.csv")
    return path.parent.name, int(match.group(1))


def _coerce(column: str, raw: str) -> object:
    """CSV text into a Python value. An empty cell is NULL, not an empty string."""
    if raw == "":
        return None
    if column in TIMESTAMPTZ_COLUMNS:
        return datetime.fromisoformat(raw)
    if column in TEXT_COLUMNS:
        return raw
    if column in INT_COLUMNS:
        try:
            return int(raw)
        except ValueError:
            return int(float(raw))
    return float(raw)


def read_session(path: Path | str) -> Session:
    """Parse a session file. Parse failures are recorded on the Session, never raised."""
    path = Path(path)
    ticker, day = ticker_and_day_from_path(path)

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = tuple(next(reader, ()))
        raw_rows = [row for row in reader if row]

    if header != CSV_COLUMNS:
        missing = sorted(set(CSV_COLUMNS) - set(header))
        extra = sorted(set(header) - set(CSV_COLUMNS))
        detail = f"{len(header)} columns, expected {len(CSV_COLUMNS)}"
        if missing:
            detail += f"; missing {missing[:5]}"
        if extra:
            detail += f"; unexpected {extra[:5]}"
        return Session(ticker, day, path, header, [], (f"header mismatch: {detail}",))

    names = [sql_name(c) for c in CSV_COLUMNS]
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for index, raw in enumerate(raw_rows):
        if len(raw) != len(names):
            problems.append(f"row {index} has {len(raw)} fields, expected {len(names)}")
            continue
        row: dict[str, object] = {}
        for name, value in zip(names, raw, strict=True):
            try:
                row[name] = _coerce(name, value)
            except ValueError:
                problems.append(f"row {index}: column {name} holds {value!r}, which will not parse")
                row[name] = None
        rows.append(row)

    return Session(ticker, day, path, header, rows, tuple(problems))


def find_sessions(data_dir: Path | str) -> list[Path]:
    """Every data/<TICKER>/day<N>.csv, sorted by ticker then day number."""
    paths = [p for p in Path(data_dir).glob("*/*.csv") if DAY_PATTERN.search(p.name)]
    return sorted(paths, key=ticker_and_day_from_path)
