"""The public data API.

Three lines from a clone to tensors::

    docker compose up -d && pip install -e backend && stocki ingest

    from stocki.datasets import load_stocki
    ds = load_stocki()

No DSN, no config, no custom classes to learn: everything below hands back
plain numpy arrays and pandas frames.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg

from ..config import Settings
from ..db.session import connect
from ..errors import StockiEmptyError
from ..ingest.columns import CSV_COLUMNS, sql_name
from .windows import (
    DEFAULT_HORIZON,
    DEFAULT_TEST_DAYS,
    DEFAULT_WINDOW,
    build_windows,
    split_mask,
)

BAR_COLUMNS = ("ticker", "day", "timestamp", "open", "high", "low", "close", "volume")
PANEL_FIELDS = ("open", "high", "low", "close", "volume")
RAW_COLUMNS = tuple(sql_name(c) for c in CSV_COLUMNS)


class Bunch(dict):
    """A dict whose keys are also attributes, exactly like sklearn's."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return list(self)


@dataclass(frozen=True)
class Panel:
    """A dense ticker x day x bar x field cube. Missing sessions are NaN."""

    values: np.ndarray
    tickers: list[str]
    days: np.ndarray
    fields: tuple[str, ...] = PANEL_FIELDS

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


@contextmanager
def _connection(
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
) -> Iterator[psycopg.Connection]:
    """Use the caller's connection, or open and close one of our own."""
    if conn is not None:
        yield conn
        return
    opened = connect(settings)
    try:
        yield opened
    finally:
        opened.close()


def _frame(conn: psycopg.Connection, sql: str, params: Sequence = ()) -> pd.DataFrame:
    cursor = conn.execute(sql, params)
    columns = [d.name for d in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def _ensure_loaded(conn: psycopg.Connection) -> None:
    if not conn.execute("SELECT EXISTS (SELECT 1 FROM bars_raw)").fetchone()[0]:
        raise StockiEmptyError(
            "no bars are loaded yet -- run `stocki ingest` to read data/ into Postgres"
        )


def known_tickers(conn: psycopg.Connection | None = None, settings: Settings | None = None):
    """Every ticker currently in the database, sorted."""
    with _connection(conn, settings) as active:
        rows = active.execute("SELECT DISTINCT ticker FROM bars_raw ORDER BY ticker").fetchall()
    return [row[0] for row in rows]


def _resolve_tickers(requested, available: list[str]) -> list[str] | None:
    if requested is None:
        return None
    wanted = [requested] if isinstance(requested, str) else list(requested)
    unknown = [t for t in wanted if t not in available]
    if unknown:
        raise ValueError(f"unknown ticker(s) {unknown}; available: {available}")
    return wanted


def _resolve_days(days) -> list[int] | None:
    if days is None:
        return None
    return [int(days)] if isinstance(days, (int, np.integer)) else [int(d) for d in days]


def _filters(tickers, days) -> tuple[str, list]:
    clauses, params = [], []
    if tickers is not None:
        clauses.append("ticker = ANY(%s)")
        params.append(tickers)
    if days is not None:
        clauses.append("day = ANY(%s)")
        params.append(days)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


# --- exploration ----------------------------------------------------------


def load_bars(
    tickers=None,
    days=None,
    as_frame: bool = True,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
):
    """The tidy long table: one row per 5-minute bar, no windowing, no labels.

    This is the one for plotting and sanity checks. `as_frame=False` returns a
    numpy record array, so `bars["close"].mean()` works without pandas.
    """
    with _connection(conn, settings) as active:
        _ensure_loaded(active)
        wanted = _resolve_tickers(tickers, known_tickers(active))
        where, params = _filters(wanted, _resolve_days(days))
        frame = _frame(
            active,
            f"SELECT {', '.join(BAR_COLUMNS)} FROM v_bars{where} ORDER BY ticker, day, timestamp",
            params,
        )
    return frame if as_frame else frame.to_records(index=False)


def load_raw(
    ticker: str,
    day: int,
    as_frame: bool = True,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
):
    """Every column of one session, in CSV order.

    `load_raw("AAPL", 13)` is `data/AAPL/day13.csv`, straight out of Postgres.
    """
    with _connection(conn, settings) as active:
        frame = _frame(
            active,
            f"SELECT {', '.join(RAW_COLUMNS)} FROM bars_raw "
            "WHERE ticker = %s AND day = %s ORDER BY timestamp",
            [ticker, int(day)],
        )
    return frame if as_frame else frame.to_records(index=False)


def load_fundamentals(
    tickers=None,
    as_frame: bool = True,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
):
    """One row per (ticker, day) instead of the same values repeated 78 times."""
    with _connection(conn, settings) as active:
        _ensure_loaded(active)
        wanted = _resolve_tickers(tickers, known_tickers(active))
        where, params = _filters(wanted, None)
        frame = _frame(active, f"SELECT * FROM v_fundamentals{where}", params)
    return frame if as_frame else frame.to_records(index=False)


def load_panel(
    tickers=None,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
) -> Panel:
    """The numpy-native cube: ticker x day x bar x OHLCV.

    Sessions that were never collected are NaN rather than dropped or zeroed,
    so `mean` tells you something is missing and `nanmean` gives the truth.
    """
    bars = load_bars(tickers, as_frame=True, conn=conn, settings=settings)

    ticker_list = sorted(bars["ticker"].unique().tolist())
    day_list = np.sort(bars["day"].unique())
    bars_per_session = int(bars.groupby(["ticker", "day"]).size().max())

    values = np.full(
        (len(ticker_list), len(day_list), bars_per_session, len(PANEL_FIELDS)),
        np.nan,
        dtype=np.float64,
    )
    ticker_at = {name: i for i, name in enumerate(ticker_list)}
    day_at = {int(day): i for i, day in enumerate(day_list)}

    for (ticker, day), session in bars.groupby(["ticker", "day"], sort=False):
        block = session[list(PANEL_FIELDS)].to_numpy(dtype=np.float64)
        values[ticker_at[ticker], day_at[int(day)], : len(block)] = block

    return Panel(values=values, tickers=ticker_list, days=day_list)


def coverage(
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Which (ticker, day) sessions exist, and how many bars each holds."""
    with _connection(conn, settings) as active:
        _ensure_loaded(active)
        return _frame(active, "SELECT * FROM v_coverage")


def describe(
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
) -> str:
    """The data card, without loading any windows."""
    frame = coverage(conn=conn, settings=settings)
    days = frame["day"]
    all_days = set(range(int(days.min()), int(days.max()) + 1))

    lines = [
        "Stocki intraday dataset",
        "=======================",
        f"{len(frame)} sessions across {frame['ticker'].nunique()} tickers, "
        f"days {days.min()}-{days.max()}",
        f"{int(frame['bar_count'].sum())} bars total",
        "",
        "coverage by ticker:",
    ]
    gaps = []
    for ticker, group in frame.groupby("ticker"):
        present = sorted(int(d) for d in group["day"])
        lines.append(
            f"  {ticker:<6} {len(present):>3} sessions (days {present[0]}-{present[-1]})"
        )
        missing = sorted(all_days - set(present))
        if missing:
            gaps.append(f"  {ticker} is missing day(s) {', '.join(str(d) for d in missing)}")

    if gaps:
        lines += ["", "gaps -- these tickers contribute fewer samples:"] + gaps
    return "\n".join(lines)


# --- the main entry point -------------------------------------------------


def load_stocki(
    tickers=None,
    days=None,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = 0.0,
    channels: Sequence[str] | None = None,
    normalize: str | None = "window-z",
    subset: str = "all",
    test_days: int = DEFAULT_TEST_DAYS,
    channels_first: bool = False,
    as_frame: bool = False,
    *,
    conn: psycopg.Connection | None = None,
    settings: Settings | None = None,
) -> Bunch:
    """Load CNN-ready windows as an sklearn-style Bunch.

    ``ds.data`` is (n_windows, window, n_channels) float32, ``ds.target`` is
    int8 where 1 means the close rose. ``ds.ticker``, ``ds.day`` and
    ``ds.timestamps`` line up with them row for row, so slicing works::

        nvda = ds.data[ds.ticker == "NVDA"]

    ``subset="train"`` / ``"test"`` splits chronologically by day, so no window
    can straddle the boundary.
    """
    bars = load_bars(tickers, days, as_frame=True, conn=conn, settings=settings)
    windows = build_windows(
        bars,
        window=window,
        horizon=horizon,
        threshold=threshold,
        channels=channels,
        normalize=normalize,
        channels_first=False,
    )

    keep = split_mask(windows.day, subset, test_days)
    x = windows.X[keep]
    y = windows.y[keep]
    ticker = windows.ticker[keep]
    day = windows.day[keep]
    timestamps = windows.timestamps[keep]

    bunch = Bunch(
        data=np.ascontiguousarray(x.transpose(0, 2, 1)) if channels_first else x,
        target=y,
        feature_names=list(windows.feature_names),
        ticker=ticker,
        day=day,
        timestamps=timestamps,
        DESCR=_data_card(
            windows.feature_names, x, y, ticker, day, window, horizon, threshold, subset
        ),
    )
    if as_frame:
        bunch.frame = _tidy_frame(x, y, ticker, day, timestamps, windows.feature_names)
    return bunch


def _tidy_frame(x, y, ticker, day, timestamps, names) -> pd.DataFrame:
    """One row per window: metadata, target, and the flattened channels."""
    n_windows, steps, _ = x.shape
    columns = [f"{name}_t{step}" for step in range(steps) for name in names]
    frame = pd.DataFrame(x.reshape(n_windows, -1), columns=columns)
    frame.insert(0, "timestamp", timestamps)
    frame.insert(0, "day", day)
    frame.insert(0, "ticker", ticker)
    frame["target"] = y
    return frame


def _data_card(names, x, y, ticker, day, window, horizon, threshold, subset) -> str:
    per_ticker = pd.Series(ticker).value_counts().sort_index()
    up_rate = float(y.mean()) if len(y) else float("nan")
    lines = [
        "Stocki intraday direction dataset",
        "=================================",
        f"{len(x)} windows of shape {x.shape[1:]} ({len(names)} channels)",
        f"channels: {', '.join(names)}",
        f"label: 1 when close rises more than {threshold:.4%} over the next "
        f"{horizon} bar(s) after a {window}-bar window",
        f"class balance: {up_rate:.1%} up",
        f"subset: {subset}",
        "",
        "windows per ticker:",
    ]
    lines += [f"  {name:<6} {count:>6}" for name, count in per_ticker.items()]
    lines += [
        "",
        f"days present: {int(day.min())}-{int(day.max())}" if len(day) else "days present: none",
        "no window crosses a session boundary; normalisation is per window.",
    ]
    return "\n".join(lines)
