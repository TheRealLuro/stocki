"""Dataset routes.

These call the very same `stocki.datasets` code Cason imports, so the dashboard
and the model can never disagree about what the data is. They are also the
expensive endpoints, so they carry their own tighter rate-limit budget.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg
from fastapi import APIRouter, Depends, Query, Response

from ...datasets.loaders import load_stocki
from ...datasets.windows import DEFAULT_HORIZON, DEFAULT_TEST_DAYS, DEFAULT_WINDOW
from ..deps import (
    MAX_HORIZON,
    MAX_WINDOW,
    MAX_WINDOWS_PER_REQUEST,
    MIN_WINDOW,
    get_connection,
    optional_ticker,
)
from ..schemas import DatasetStats, WindowsPayload
from ..security import DATASET_BUCKET, rate_limit

router = APIRouter(
    prefix="/dataset", tags=["dataset"], dependencies=[Depends(rate_limit(DATASET_BUCKET))]
)


@dataclass(frozen=True)
class Params:
    ticker: str | None
    window: int
    horizon: int
    threshold: float
    subset: str
    test_days: int


def dataset_params(
    ticker: str | None = Depends(optional_ticker),
    window: int = Query(default=DEFAULT_WINDOW, ge=MIN_WINDOW, le=MAX_WINDOW),
    horizon: int = Query(default=DEFAULT_HORIZON, ge=1, le=MAX_HORIZON),
    threshold: float = Query(default=0.0, ge=-1.0, le=1.0),
    subset: str = Query(default="all", pattern="^(all|train|test)$"),
    test_days: int = Query(default=DEFAULT_TEST_DAYS, ge=1, le=19),
) -> Params:
    return Params(ticker, window, horizon, threshold, subset, test_days)


def _load(params: Params, conn: psycopg.Connection):
    return load_stocki(
        tickers=params.ticker,
        window=params.window,
        horizon=params.horizon,
        threshold=params.threshold,
        subset=params.subset,
        test_days=params.test_days,
        conn=conn,
    )


@router.get("/stats", response_model=DatasetStats, summary="The data card as JSON")
def stats(
    params: Params = Depends(dataset_params),
    conn: psycopg.Connection = Depends(get_connection),
):
    ds = _load(params, conn)
    return DatasetStats(
        n_windows=len(ds.data),
        shape=list(ds.data.shape[1:]),
        feature_names=ds.feature_names,
        up_rate=float(ds.target.mean()) if len(ds.target) else 0.0,
        tickers=sorted(set(ds.ticker.tolist())),
        days=sorted({int(d) for d in ds.day}),
        window=params.window,
        horizon=params.horizon,
        subset=params.subset,
    )


@router.get(
    "/windows",
    response_model=None,
    summary="The tensors themselves (json or npz)",
)
def windows(
    params: Params = Depends(dataset_params),
    limit: int = Query(default=100, ge=1, le=MAX_WINDOWS_PER_REQUEST),
    format: str = Query(default="json", pattern="^(json|npz)$"),
    conn: psycopg.Connection = Depends(get_connection),
):
    """Capped at 500 windows per request. For the full set, import the package."""
    ds = _load(params, conn)
    data = ds.data[:limit]
    target = ds.target[:limit]
    ticker = ds.ticker[:limit]
    day = ds.day[:limit]
    timestamps = ds.timestamps[:limit]

    if format == "npz":
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            data=data,
            target=target,
            ticker=ticker,
            day=day,
            timestamps=timestamps,
            feature_names=np.array(ds.feature_names),
        )
        return Response(
            content=buffer.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="stocki-windows.npz"'},
        )

    return WindowsPayload(
        shape=list(data.shape),
        feature_names=ds.feature_names,
        data=data.tolist(),
        target=target.tolist(),
        ticker=ticker.tolist(),
        day=[int(d) for d in day],
        timestamps=pd.DatetimeIndex(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
    )
