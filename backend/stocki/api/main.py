"""The Stocki API.

    uvicorn stocki.api.main:app --host 0.0.0.0 --port 8000

Everything under /api/v1 is read-only. The dataset routes call the same
`stocki.datasets` code the training package imports, so the dashboard and the
model always see the same numbers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, get_settings
from . import errors
from .routes import bars, dataset, meta
from .security import GuardMiddleware, RateLimiter, rate_limit

API_PREFIX = "/api/v1"

DESCRIPTION = """
Read-only access to the Stocki intraday dataset.

* `/api/v1/bars/{ticker}/{day}` mirrors `data/<TICKER>/day<N>.csv`
* `/api/v1/dataset/*` serves CNN-ready windows built by the same code the
  training package imports
* every error returns `{error, detail, request_id}`
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Stocki API",
        version="0.1.0",
        description=DESCRIPTION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.limiter = RateLimiter()

    app.add_middleware(GuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    errors.install(app)

    app.include_router(meta.health_router)

    v1 = APIRouter(prefix=API_PREFIX, dependencies=[Depends(rate_limit())])
    v1.include_router(meta.router)
    v1.include_router(bars.router)
    v1.include_router(dataset.router)
    app.include_router(v1)

    return app


app = create_app()
