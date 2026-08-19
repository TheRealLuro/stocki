"""Session validation.

A file is checked in full before anything is written, so a bad file is
quarantined rather than half-loaded. Each rule reports once with a count
instead of eighty near-identical lines.
"""

from __future__ import annotations

from datetime import timedelta

from .columns import CSV_COLUMNS
from .reader import Session

EXPECTED_BARS = 78
BAR_INTERVAL = timedelta(minutes=5)
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "ticker")


def _first_and_count(hits: list[str]) -> str:
    if len(hits) == 1:
        return hits[0]
    return f"{hits[0]} (and {len(hits) - 1} more)"


def validate_session(session: Session, expected_bars: int = EXPECTED_BARS) -> list[str]:
    """Return every problem found. An empty list means the file is loadable."""
    where = session.source_file
    problems = [f"{where}: {p}" for p in session.problems]

    if session.header != CSV_COLUMNS:
        return problems

    rows = session.rows
    if len(rows) != expected_bars:
        problems.append(f"{where}: has {len(rows)} bars, expected {expected_bars}")
    if not rows:
        return problems

    missing: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        for column in REQUIRED_COLUMNS:
            if row.get(column) is None:
                missing.setdefault(column, []).append(f"row {index}")
    for column, hits in missing.items():
        problems.append(f"{where}: column {column} is empty at {_first_and_count(hits)}")

    problems += _check_timestamps(where, rows)
    problems += _check_prices(where, rows)
    problems += _check_ticker(where, session.ticker, rows)
    return problems


def _check_timestamps(where: str, rows: list[dict]) -> list[str]:
    stamps = [r["timestamp"] for r in rows if r.get("timestamp") is not None]
    if len(stamps) < 2:
        return []

    problems = []
    disordered = [f"row {i}" for i in range(1, len(stamps)) if stamps[i] <= stamps[i - 1]]
    if disordered:
        problems.append(
            f"{where}: timestamps are not ascending at {_first_and_count(disordered)}"
        )

    gaps = [
        f"row {i} is {stamps[i] - stamps[i - 1]} after the previous bar"
        for i in range(1, len(stamps))
        if stamps[i] - stamps[i - 1] != BAR_INTERVAL
    ]
    if gaps:
        problems.append(f"{where}: bar spacing is not 5 minutes at {_first_and_count(gaps)}")

    dates = {s.date() for s in stamps}
    if len(dates) > 1:
        listed = ", ".join(str(d) for d in sorted(dates))
        problems.append(f"{where}: one file must hold one session, but this spans {listed}")

    return problems


def _check_prices(where: str, rows: list[dict]) -> list[str]:
    problems = []
    bad_high: list[str] = []
    bad_low: list[str] = []
    bad_volume: list[str] = []

    for index, row in enumerate(rows):
        o, h, low, c = row.get("open"), row.get("high"), row.get("low"), row.get("close")
        volume = row.get("volume")
        if None not in (o, h, low, c):
            if h < low or h < max(o, c):
                bad_high.append(f"row {index} (high={h}, open={o}, close={c}, low={low})")
            if low > min(o, c):
                bad_low.append(f"row {index} (low={low}, open={o}, close={c})")
        if volume is not None and volume < 0:
            bad_volume.append(f"row {index} (volume={volume})")

    if bad_high:
        problems.append(
            f"{where}: high is below the other prices at {_first_and_count(bad_high)}"
        )
    if bad_low:
        problems.append(f"{where}: low is above the other prices at {_first_and_count(bad_low)}")
    if bad_volume:
        problems.append(f"{where}: volume is negative at {_first_and_count(bad_volume)}")
    return problems


def _check_ticker(where: str, expected: str, rows: list[dict]) -> list[str]:
    found = {row.get("ticker") for row in rows} - {None}
    if found and found != {expected}:
        listed = ", ".join(sorted(str(t) for t in found))
        return [f"{where}: ticker column says {listed} but the path says {expected}"]
    return []
