"""Response models.

Every route declares one, so the OpenAPI schema at /docs is accurate enough
for the frontend to generate a typed client from it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = Field(examples=["ok"])
    database: str = Field(examples=["ok"])
    bar_count: int | None


class TickerSummary(BaseModel):
    ticker: str
    name_en: str | None = None
    name_cn: str | None = None
    currency: str | None = None
    session_count: int
    first_day: int
    last_day: int
    bar_count: int


class CoverageRow(BaseModel):
    ticker: str
    day: int
    bar_count: int
    first_bar: datetime
    last_bar: datetime


class Bar(BaseModel):
    ticker: str
    day: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class BarPage(BaseModel):
    items: list[Bar]
    total: int
    limit: int
    offset: int


class Fundamentals(BaseModel):
    ticker: str
    day: int
    fundamentals: dict[str, Any]


class News(BaseModel):
    ticker: str
    day: int
    news_count: int | None = None
    news_latest_headline: str | None = None


class DatasetStats(BaseModel):
    n_windows: int
    shape: list[int]
    feature_names: list[str]
    up_rate: float
    tickers: list[str]
    days: list[int]
    window: int
    horizon: int
    subset: str


class WindowsPayload(BaseModel):
    shape: list[int]
    feature_names: list[str]
    data: list[Any]
    target: list[int]
    ticker: list[str]
    day: list[int]
    timestamps: list[str]
