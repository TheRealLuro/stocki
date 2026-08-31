"""Pull live sessions from Alpha Vantage and write them as ``data/`` CSVs.

The design decision worth stating: this writes **files**, not database rows.

``data/<TICKER>/day<N>.csv`` is the source of truth in this repo -- ``stocki
ingest`` copies it into Postgres, the API serves that, and the training package
reads the API. Landing live data at the front of that chain means every stage
downstream keeps working untouched, gets the same validation the committed files
get, and stays diffable and re-ingestable. Writing straight into ``bars_raw``
would have skipped the validation, left ``stocki verify`` permanently failing,
and put the model on data no one could look at.

So the whole feature is one new producer of the existing format::

    Alpha Vantage -> data/<TICKER>/day<N>.csv -> stocki ingest -> bars_raw
                                                                    |
                                    /api/v1/bars -> model_training -> model

Nothing in ``model/``, ``model_training/`` or ``frontend/`` changes.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from ..config import Settings, get_settings
from ..ingest.columns import (
    BIGINT_COLUMNS,
    CSV_COLUMNS,
    INTEGER_COLUMNS,
    SMALLINT_COLUMNS,
    TEXT_COLUMNS,
    TIMESTAMPTZ_COLUMNS,
    sql_name,
)
from ..ingest.reader import Session, read_session
from ..ingest.validate import validate_session
from . import fields
from .client import AlphaVantage, months_between, parse_number
from .sessions import (
    BARS_PER_SESSION,
    DayIndex,
    DayNumberingError,
    eastern_to_utc,
    first_session_date,
    news_window,
    recent_weekdays,
    session_window,
)

logger = logging.getLogger("stocki.live")

INT_COLUMNS = BIGINT_COLUMNS | INTEGER_COLUMNS | SMALLINT_COLUMNS

#: Intraday bars carry no corporate actions, and every committed row holds 0.0.
NO_CORPORATE_ACTION = 0.0


@dataclass
class Bar:
    """One 5-minute bar, timestamped in UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class FetchReport:
    """What a fetch run did, in the shape `IngestReport` reports an ingest."""

    written: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    requests_made: int = 0
    cache_hits: int = 0

    @property
    def ok(self) -> bool:
        return not self.rejected

    def summary(self) -> str:
        lines = [
            f"wrote {len(self.written)} session files "
            f"({self.requests_made} Alpha Vantage requests, {self.cache_hits} cached)"
        ]
        for name in sorted(self.written):
            lines.append(f"  + {name}")
        if self.skipped:
            lines.append(f"skipped {len(self.skipped)}:")
            for name, why in sorted(self.skipped.items()):
                lines.append(f"  - {name}: {why}")
        if self.rejected:
            lines.append(f"rejected {len(self.rejected)} that failed validation:")
            for name, problems in sorted(self.rejected.items()):
                for problem in problems:
                    lines.append(f"  ! {problem}")
        return "\n".join(lines)


# --- reading the provider payloads ----------------------------------------


def parse_intraday(payload: dict, interval: str = "5min") -> dict:
    """The intraday response as `{session date: [Bar, ...]}`, oldest bar first.

    Alpha Vantage keys each bar with the *start* of its interval in US/Eastern,
    which is exactly how the committed files are stamped -- so 09:30 Eastern
    becomes the 13:30Z row, not a 13:35Z one. Sessions are grouped by the
    Eastern date, because that is what a trading day is.
    """
    series = payload.get(f"Time Series ({interval})")
    if not isinstance(series, dict):
        available = [k for k in payload if "Time Series" in k]
        raise ValueError(
            f"no '{interval}' series in the response"
            + (f"; it holds {available}" if available else "")
        )

    sessions: dict = {}
    for stamp, values in series.items():
        try:
            local = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        bar = Bar(
            timestamp=eastern_to_utc(local),
            open=float(values["1. open"]),
            high=float(values["2. high"]),
            low=float(values["3. low"]),
            close=float(values["4. close"]),
            volume=int(float(values["5. volume"])),
        )
        sessions.setdefault(local.date(), []).append(bar)

    for bars in sessions.values():
        bars.sort(key=lambda b: b.timestamp)
    return sessions


def quote_price(payload: dict) -> float | None:
    """The last traded price out of a GLOBAL_QUOTE response."""
    quote = payload.get("Global Quote") or {}
    return parse_number(quote.get("05. price"))


# --- building rows --------------------------------------------------------


def static_columns(
    ticker: str,
    *,
    overview: dict,
    income: dict,
    balance: dict,
    cash: dict,
    articles: list,
    name_cn: str | None,
    current_price: float | None,
) -> dict:
    """Every column that is constant across a session -- the other 105.

    These repeat on all 78 rows, exactly as they do in the committed files;
    ``v_fundamentals`` is the view that collapses them again.
    """
    annual_income = fields.report(income, "annual")
    annual_balance = fields.report(balance, "annual")
    annual_cash = fields.report(cash, "annual")
    quarterly_balance = fields.report(balance, "quarterly")

    return {
        **fields.identity_columns(ticker, overview, name_cn=name_cn),
        **fields.stock_info_columns(
            overview, quarterly_balance, current_price=current_price
        ),
        **fields.income_columns(annual_income, annual_balance),
        **fields.balance_columns(annual_balance),
        **fields.cashflow_columns(annual_cash, annual_balance),
        **fields.recommendation_columns(overview),
        **fields.news_columns(articles),
    }


def session_rows(bars: list, static: dict) -> list:
    """The session as rows keyed by CSV column name, ready to write."""
    rows = []
    for bar in bars:
        row = {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "Dividends": NO_CORPORATE_ACTION,
            "Stock Splits": NO_CORPORATE_ACTION,
            "hour": bar.timestamp.hour,
            "minute": bar.timestamp.minute,
            **static,
        }
        rows.append({column: row.get(column) for column in CSV_COLUMNS})
    return rows


def _cell(column: str, value) -> str:
    """One value as the files write it. An empty cell is NULL, not ''."""
    if value is None:
        return ""
    name = sql_name(column)
    if name in TIMESTAMPTZ_COLUMNS:
        return value.isoformat(sep=" ")
    if name in TEXT_COLUMNS:
        return str(value)
    if name in INT_COLUMNS:
        return str(int(value))
    return str(float(value))


def write_session(path: Path, rows: list) -> None:
    """Write a session file byte-compatibly with the committed ones.

    ``csv.writer`` defaults to CRLF and quotes only what needs it, which is what
    ``data/`` already uses -- ``"Amazon.Com, Inc."`` is quoted, nothing else on
    the row is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_cell(column, row.get(column)) for column in CSV_COLUMNS])


def check_session(ticker: str, day: int, path: Path, rows: list, expected_bars: int) -> list:
    """Run the ingest validator over built rows, before anything is written."""
    sql_rows = [{sql_name(k): v for k, v in row.items()} for row in rows]
    session = Session(ticker, day, path, CSV_COLUMNS, sql_rows)
    return validate_session(session, expected_bars=expected_bars)


# --- disk ----------------------------------------------------------------


def known_tickers(data_dir: Path | str) -> list:
    """The ticker folders already in `data/`, sorted."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def existing_name_cn(data_dir: Path | str, ticker: str) -> str | None:
    """The Chinese name already recorded for a ticker, if there is one.

    Alpha Vantage has no such field, and inventing one would be worse than an
    empty cell -- so an existing ticker keeps the name the earlier files use and
    a new ticker simply has none.
    """
    folder = Path(data_dir) / ticker
    if not folder.is_dir():
        return None
    index = CSV_COLUMNS.index("thsname_cn")
    for path in sorted(folder.glob("*.csv"), reverse=True):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                row = next(reader, None)
        except OSError:
            continue
        if row and len(row) > index and row[index].strip():
            return row[index]
    return None


# --- the run --------------------------------------------------------------


def fetch_sessions(
    tickers: list,
    *,
    dates: list | None = None,
    days: int = 1,
    ending: date | None = None,
    data_dir: Path | str | None = None,
    client: AlphaVantage | None = None,
    settings: Settings | None = None,
    with_fundamentals: bool = True,
    with_news: bool = True,
    with_quote: bool = False,
    allow_partial: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> FetchReport:
    """Fetch `tickers` for `dates` and write them into `data_dir`.

    Request budget, which is what the free plan makes you care about: one
    intraday call per ticker per calendar month covered, one news call for the
    whole universe, and four fundamentals calls per ticker that are cached for
    ``STOCKI_LIVE_CACHE_HOURS``. So the first run over ten tickers costs about
    51 requests and every run after it costs 11.
    """
    settings = settings or get_settings()
    data_dir = Path(data_dir or settings.data_dir)
    client = client or AlphaVantage(settings)

    if dates is None:
        dates = recent_weekdays(days, ending=ending or datetime.now(UTC).date())
    dates = sorted(dates)
    if not dates:
        raise ValueError("no dates to fetch")

    report = FetchReport()
    index = DayIndex.from_data_dir(data_dir)
    expected_bars = BARS_PER_SESSION

    news_split = _fetch_news(client, tickers, dates) if with_news else {}

    for ticker in tickers:
        try:
            sessions = _fetch_bars(client, ticker, dates)
        except Exception as exc:  # one bad ticker must not lose the others
            for day in dates:
                report.skipped[f"{ticker}/{day}"] = str(exc)
            logger.warning("%s: %s", ticker, exc)
            continue

        payloads = _fetch_fundamentals(client, ticker) if with_fundamentals else {}
        price = None
        if with_quote:
            price = quote_price(client.global_quote(ticker))
        # Read before the loop writes anything, so a multi-day run does not
        # pick this up out of a file it has just written.
        name_cn = existing_name_cn(data_dir, ticker)

        for day in dates:
            label = f"{ticker}/{day}"
            bars = _regular_session(sessions.get(day, []), day)

            if not bars:
                report.skipped[label] = "no bars returned (market holiday, or not traded)"
                continue
            if len(bars) < expected_bars and not allow_partial:
                report.skipped[label] = (
                    f"{len(bars)} of {expected_bars} bars -- the session is still open. "
                    "Pass --allow-partial to write it anyway."
                )
                continue

            try:
                number = index.assign(day)
            except DayNumberingError as exc:
                report.skipped[label] = str(exc)
                continue

            path = data_dir / ticker / f"day{number}.csv"
            if path.exists() and not overwrite:
                report.skipped[label] = f"{path.name} already exists (pass --overwrite)"
                continue

            static = static_columns(
                ticker,
                overview=payloads.get("overview", {}),
                income=payloads.get("income", {}),
                balance=payloads.get("balance", {}),
                cash=payloads.get("cash", {}),
                articles=news_split.get(day, {}).get(ticker.upper(), []),
                name_cn=name_cn,
                current_price=price if price is not None else bars[-1].close,
            )
            rows = session_rows(bars, static)

            problems = check_session(
                ticker, number, path, rows, len(rows) if allow_partial else expected_bars
            )
            if problems:
                report.rejected[label] = problems
                continue

            if not dry_run:
                write_session(path, rows)
            report.written.append(f"{ticker}/day{number}.csv ({day}, {len(rows)} bars)")

    report.requests_made = client.requests_made
    report.cache_hits = client.cache_hits
    return report


def _regular_session(bars: list, day: date) -> list:
    """Only the 09:30-15:55 bars, in order, deduplicated on timestamp."""
    first, end = session_window(day)
    kept: dict = {}
    for bar in bars:
        if first <= bar.timestamp < end:
            kept[bar.timestamp] = bar
    return [kept[stamp] for stamp in sorted(kept)]


def _fetch_bars(client: AlphaVantage, ticker: str, dates: list) -> dict:
    """Intraday bars covering `dates`, one request per calendar month spanned."""
    sessions: dict = {}
    for month in months_between(dates[0], dates[-1]):
        payload = client.intraday(ticker, month=month)
        for day, bars in parse_intraday(payload).items():
            sessions.setdefault(day, []).extend(bars)
    return sessions


def _fetch_fundamentals(client: AlphaVantage, ticker: str) -> dict:
    """The four company reports. Cached, so most runs spend nothing here."""
    return {
        "overview": client.overview(ticker),
        "income": client.income_statement(ticker),
        "balance": client.balance_sheet(ticker),
        "cash": client.cash_flow(ticker),
    }


def _fetch_news(client: AlphaVantage, tickers: list, dates: list) -> dict:
    """One news call for every ticker and every date, split up afterwards.

    Returns ``{session date: {ticker: [articles]}}``. Splitting by date as well
    as by ticker is the point: ``news_count`` is a per-session column, so a
    three-day fetch must not give all three days the same three days of news.
    """
    start, _ = news_window(dates[0])
    _, end = news_window(dates[-1])
    try:
        payload = client.news(tickers, time_from=start, time_to=end)
    except Exception as exc:  # news is the least important column here
        logger.warning("news unavailable (%s); news_count will be 0", exc)
        return {}

    return {
        day: fields.news_by_ticker(payload, tickers, *news_window(day))
        for day in dates
    }


def verify_written(path: Path, expected_bars: int = BARS_PER_SESSION) -> list:
    """Re-read a written file through the ingest reader and validate it.

    Proof that what landed on disk is a file `stocki ingest` will accept, rather
    than a file this module believes it wrote.
    """
    session = read_session(path)
    return validate_session(session, expected_bars=expected_bars)


__all__ = [
    "Bar",
    "FetchReport",
    "existing_name_cn",
    "fetch_sessions",
    "first_session_date",
    "known_tickers",
    "parse_intraday",
    "session_rows",
    "static_columns",
    "verify_written",
    "write_session",
]
