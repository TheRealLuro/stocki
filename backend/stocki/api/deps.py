"""Request-scoped dependencies: settings, connections, and validated parameters.

Every parameter that reaches SQL is checked here first, and `ticker` is checked
against the live universe so a typo comes back as a 422 that lists the real ones.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
from fastapi import Depends, HTTPException, Path, Query, Request

from ..config import Settings
from ..datasets.loaders import known_tickers
from ..db.session import connect

MAX_DAY = 20
MIN_WINDOW = 8
MAX_WINDOW = 78
MAX_HORIZON = 16
DEFAULT_PAGE = 200
MAX_PAGE = 1000
MAX_WINDOWS_PER_REQUEST = 500


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_connection(request: Request) -> Iterator[psycopg.Connection]:
    """One read-only connection per request, always closed."""
    conn = connect(request.app.state.settings)
    try:
        yield conn
    finally:
        conn.close()


def valid_ticker(
    ticker: str = Path(min_length=1, max_length=12),
    conn: psycopg.Connection = Depends(get_connection),
) -> str:
    available = known_tickers(conn)
    if ticker not in available:
        raise HTTPException(
            status_code=422,
            detail=f"unknown ticker {ticker!r}; available tickers are {available}",
        )
    return ticker


def optional_ticker(
    ticker: str | None = Query(default=None, max_length=12),
    conn: psycopg.Connection = Depends(get_connection),
) -> str | None:
    if ticker is None:
        return None
    available = known_tickers(conn)
    if ticker not in available:
        raise HTTPException(
            status_code=422,
            detail=f"unknown ticker {ticker!r}; available tickers are {available}",
        )
    return ticker


def valid_day(day: int = Path(ge=1, le=MAX_DAY)) -> int:
    return day
