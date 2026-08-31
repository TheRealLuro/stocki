"""The trading-day index, and the market clock that fills it.

``day13.csv`` is not "the 13th file", it is *the 13th trading day collected*,
and it means the same calendar date for every ticker: day 1 is 2026-07-20 for
AMZN, GOOGL and NVDA alike. That shared numbering is load-bearing -- the
train/test split in ``stocki.datasets.windows`` slices on ``day``, so if a fetch
gave the same date two different numbers across tickers, the chronological
split would quietly stop being chronological.

The day number lives only in the filename, so :class:`DayIndex` recovers the
mapping by reading the first timestamp out of every file already on disk, and
hands new dates the next numbers in order.

Alpha Vantage timestamps arrive naive in US/Eastern. Converting them is the
other thing that must not be approximate: the files hold UTC (13:30 to 19:55
during daylight saving), so a bar mislabelled by an hour would collide with the
neighbouring bar on the ``(ticker, timestamp)`` primary key.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ..errors import StockiLiveError
from ..ingest.reader import DAY_PATTERN, ticker_and_day_from_path

#: A regular session: 09:30 to 15:55 Eastern inclusive, 5-minute bars.
MARKET_OPEN = time(9, 30)
MARKET_LAST_BAR = time(15, 55)
BAR_INTERVAL = timedelta(minutes=5)
BARS_PER_SESSION = 78


# --- the market clock -----------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth `weekday` (Monday=0) of a month, e.g. the 2nd Sunday in March."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def eastern_offset(moment: datetime) -> timedelta:
    """The UTC offset of US Eastern at a naive local `moment`.

    Daylight saving runs from 02:00 on the second Sunday in March to 02:00 on
    the first Sunday in November. A trading session is 09:30 to 16:00, never
    within seven hours of either switch, so the ambiguous and non-existent hours
    cannot arise for the timestamps this module converts.

    This is spelled out rather than delegated to ``zoneinfo`` because Windows
    ships no IANA database, and a fetch that works in Docker but not on a
    teammate's laptop is worse than fifteen lines of arithmetic.
    """
    starts = datetime.combine(_nth_weekday(moment.year, 3, 6, 2), time(2, 0))
    ends = datetime.combine(_nth_weekday(moment.year, 11, 6, 1), time(2, 0))
    daylight = starts <= moment.replace(tzinfo=None) < ends
    return timedelta(hours=-4 if daylight else -5)


def eastern_to_utc(moment: datetime) -> datetime:
    """A naive US/Eastern datetime as an aware UTC one."""
    naive = moment.replace(tzinfo=None)
    return (naive - eastern_offset(naive)).replace(tzinfo=UTC)


def session_window(day: date) -> tuple[datetime, datetime]:
    """The UTC half-open window `[first bar, after the last bar)` of a session.

    The end is one interval past the 15:55 bar, so it is the right bound for a
    "did this article land during the session" test.
    """
    first = eastern_to_utc(datetime.combine(day, MARKET_OPEN))
    last = eastern_to_utc(datetime.combine(day, MARKET_LAST_BAR))
    return first, last + BAR_INTERVAL


def news_window(day: date) -> tuple[datetime, datetime]:
    """The UTC window whose news belongs to a session: midnight to the close.

    Not the same as :func:`session_window`, deliberately. Pre-market headlines
    are exactly the ones a session reacts to, so the window opens at 00:00
    Eastern rather than at 09:30 -- but it closes at the closing bell, because
    an article published at 16:46 could not have moved a price that stopped
    printing at 16:00. Attaching it to this session would be look-ahead, and
    ``news_count`` is a model input.
    """
    midnight = eastern_to_utc(datetime.combine(day, time(0, 0)))
    close = eastern_to_utc(datetime.combine(day, MARKET_LAST_BAR)) + BAR_INTERVAL
    return midnight, close


def expected_timestamps(day: date) -> list[datetime]:
    """The 78 UTC bar timestamps a complete session must hold, in order."""
    first, _ = session_window(day)
    return [first + BAR_INTERVAL * i for i in range(BARS_PER_SESSION)]


def is_weekday(day: date) -> bool:
    """Monday to Friday. Exchange holidays are not modelled -- see `DayIndex`."""
    return day.weekday() < 5


def recent_weekdays(count: int, *, ending: date) -> list[date]:
    """The last `count` weekdays up to and including `ending`, oldest first.

    A holiday simply returns no bars from the provider, and the session is
    skipped with a note -- which is how the gaps already in `data/` (AAPL days
    1-12, MSFT day 1) are represented too.
    """
    days: list[date] = []
    cursor = ending
    while len(days) < count:
        if is_weekday(cursor):
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


# --- the index ------------------------------------------------------------


class DayNumberingError(StockiLiveError):
    """The day numbers on disk disagree, or a new date cannot be numbered."""


@dataclass
class DayIndex:
    """Which calendar date each ``day<N>`` refers to, read from ``data/``."""

    dates: dict[date, int] = field(default_factory=dict)

    @property
    def days(self) -> dict[int, date]:
        return {number: day for day, number in self.dates.items()}

    @property
    def last_number(self) -> int:
        return max(self.dates.values(), default=0)

    @property
    def last_date(self) -> date | None:
        return max(self.dates, default=None)

    def number_for(self, day: date) -> int | None:
        return self.dates.get(day)

    def assign(self, day: date) -> int:
        """The day number for `day`, allocating the next one if it is new.

        A date *inside* the range already collected but with no number of its
        own is refused: giving it ``last + 1`` would put a later number on an
        earlier date and break every chronological split downstream.
        """
        existing = self.dates.get(day)
        if existing is not None:
            return existing

        latest = self.last_date
        if latest is not None and day < latest:
            raise DayNumberingError(
                f"{day} falls inside the range already collected (days 1-{self.last_number}, "
                f"through {latest}) but has no day number. Renumbering the existing files is "
                "not something fetch will do on its own -- fetch dates after "
                f"{latest}, or renumber data/ deliberately first."
            )

        number = self.last_number + 1
        self.dates[day] = number
        return number

    @classmethod
    def from_data_dir(cls, data_dir: Path | str) -> DayIndex:
        """Read the date of every ``data/<TICKER>/day<N>.csv`` already on disk."""
        index = cls()
        seen: dict[int, tuple[date, str]] = {}

        for path in sorted(Path(data_dir).glob("*/*.csv")):
            if not DAY_PATTERN.search(path.name):
                continue
            try:
                _, number = ticker_and_day_from_path(path)
            except ValueError:
                continue
            day = first_session_date(path)
            if day is None:
                continue

            claimed = seen.get(number)
            if claimed is not None and claimed[0] != day:
                raise DayNumberingError(
                    f"day {number} is {claimed[0]} in {claimed[1]} but {day} in "
                    f"{path.parent.name}/{path.name}; a day number must mean one date "
                    "across every ticker"
                )
            seen[number] = (day, f"{path.parent.name}/{path.name}")
            index.dates[day] = number

        return index


def first_session_date(path: Path) -> date | None:
    """The calendar date of a session file, from its first data row."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            row = next(reader, None)
    except OSError:
        return None
    if not row:
        return None
    try:
        return datetime.fromisoformat(row[0]).date()
    except ValueError:
        return None
