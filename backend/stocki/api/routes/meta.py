"""Health, the ticker universe, and coverage."""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...db.session import connect
from ..deps import get_connection
from ..schemas import CoverageRow, Health, TickerSummary

health_router = APIRouter(tags=["meta"])
router = APIRouter(tags=["meta"])

TICKER_SQL = """
SELECT ticker,
       max(thsname_en)      AS name_en,
       max(thsname_cn)      AS name_cn,
       max(currency)        AS currency,
       count(DISTINCT day)  AS session_count,
       min(day)             AS first_day,
       max(day)             AS last_day,
       count(*)             AS bar_count
FROM bars_raw
GROUP BY ticker
ORDER BY ticker
"""


@health_router.get("/health", response_model=Health, summary="Liveness and row count")
def health(request: Request):
    """What the compose healthcheck hits. 503 when Postgres is unreachable."""
    try:
        with connect(request.app.state.settings, timeout=3) as conn:
            count = conn.execute("SELECT count(*) FROM bars_raw").fetchone()[0]
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "unreachable", "bar_count": None},
        )
    return Health(status="ok", database="ok", bar_count=count)


@router.get("/tickers", response_model=list[TickerSummary], summary="The ticker universe")
def tickers(conn: psycopg.Connection = Depends(get_connection)):
    rows = conn.execute(TICKER_SQL).fetchall()
    return [
        TickerSummary(
            ticker=row[0],
            name_en=row[1],
            name_cn=row[2],
            currency=row[3],
            session_count=row[4],
            first_day=row[5],
            last_day=row[6],
            bar_count=row[7],
        )
        for row in rows
    ]


@router.get("/coverage", response_model=list[CoverageRow], summary="Which sessions exist")
def coverage(conn: psycopg.Connection = Depends(get_connection)):
    """The ticker-by-day matrix. Gaps show up here as data, not as a surprise."""
    rows = conn.execute(
        "SELECT ticker, day, bar_count, first_bar, last_bar FROM v_coverage"
    ).fetchall()
    return [
        CoverageRow(
            ticker=row[0], day=row[1], bar_count=row[2], first_bar=row[3], last_bar=row[4]
        )
        for row in rows
    ]
