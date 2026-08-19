"""Bars, fundamentals, and news -- the charting and detail routes."""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query

from ...ingest.columns import static_columns
from ..deps import (
    DEFAULT_PAGE,
    MAX_DAY,
    MAX_PAGE,
    get_connection,
    optional_ticker,
    valid_day,
    valid_ticker,
)
from ..schemas import Bar, BarPage, Fundamentals, News

router = APIRouter(tags=["bars"])

BAR_FIELDS = "ticker, day, timestamp, open, high, low, close, volume"
STATIC = static_columns()


def _as_bar(row) -> Bar:
    return Bar(
        ticker=row[0],
        day=row[1],
        timestamp=row[2],
        open=row[3],
        high=row[4],
        low=row[5],
        close=row[6],
        volume=row[7],
    )


@router.get("/bars", response_model=BarPage, summary="Paginated bars for charting")
def list_bars(
    ticker: str | None = Depends(optional_ticker),
    day: int | None = Query(default=None, ge=1, le=MAX_DAY),
    limit: int = Query(default=DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    offset: int = Query(default=0, ge=0),
    conn: psycopg.Connection = Depends(get_connection),
):
    clauses, params = [], []
    if ticker is not None:
        clauses.append("ticker = %s")
        params.append(ticker)
    if day is not None:
        clauses.append("day = %s")
        params.append(day)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    total = conn.execute(f"SELECT count(*) FROM v_bars{where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT {BAR_FIELDS} FROM v_bars{where} ORDER BY ticker, day, timestamp "
        "LIMIT %s OFFSET %s",
        [*params, limit, offset],
    ).fetchall()

    return BarPage(
        items=[_as_bar(row) for row in rows], total=total, limit=limit, offset=offset
    )


@router.get("/bars/{ticker}/{day}", response_model=list[Bar], summary="One session")
def session_bars(
    ticker: str = Depends(valid_ticker),
    day: int = Depends(valid_day),
    conn: psycopg.Connection = Depends(get_connection),
):
    """The same rows as data/<TICKER>/day<N>.csv, in the same order."""
    rows = conn.execute(
        f"SELECT {BAR_FIELDS} FROM v_bars WHERE ticker = %s AND day = %s ORDER BY timestamp",
        [ticker, day],
    ).fetchall()
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"no session for {ticker} on day {day}"
        )
    return [_as_bar(row) for row in rows]


@router.get(
    "/fundamentals/{ticker}/{day}",
    response_model=Fundamentals,
    summary="The per-session fundamentals snapshot",
)
def fundamentals(
    ticker: str = Depends(valid_ticker),
    day: int = Depends(valid_day),
    conn: psycopg.Connection = Depends(get_connection),
):
    """One object instead of the same ~105 values repeated across 78 bars."""
    row = conn.execute(
        f"SELECT {', '.join(STATIC)} FROM v_fundamentals WHERE ticker = %s AND day = %s",
        [ticker, day],
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no session for {ticker} on day {day}"
        )
    return Fundamentals(
        ticker=ticker, day=day, fundamentals=dict(zip(STATIC, row, strict=True))
    )


@router.get("/news/{ticker}/{day}", response_model=News, summary="News for a session")
def news(
    ticker: str = Depends(valid_ticker),
    day: int = Depends(valid_day),
    conn: psycopg.Connection = Depends(get_connection),
):
    row = conn.execute(
        "SELECT news_count, news_latest_headline FROM v_news WHERE ticker = %s AND day = %s",
        [ticker, day],
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no session for {ticker} on day {day}"
        )
    return News(ticker=ticker, day=day, news_count=row[0], news_latest_headline=row[1])
