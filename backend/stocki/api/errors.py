"""One error shape for every failure.

Clients always get ``{error, detail, request_id}``. A 500 logs the traceback
server-side against that id and returns none of it, so a browser never shows a
stack trace or a connection string.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("stocki.api")

REQUEST_ID_HEADER = "X-Request-ID"
GENERIC_500 = "an unexpected error occurred; quote the request_id when reporting it"


def request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if existing is None:
        existing = uuid.uuid4().hex
        request.state.request_id = existing
    return existing


def envelope(request: Request, status_code: int, error: str, detail) -> JSONResponse:
    rid = request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "detail": detail, "request_id": rid},
        headers={REQUEST_ID_HEADER: rid},
    )


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    names = {404: "not_found", 422: "validation_error", 429: "rate_limited"}
    return envelope(request, exc.status_code, names.get(exc.status_code, "error"), exc.detail)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = [
        {"field": ".".join(str(p) for p in err.get("loc", ())), "problem": err.get("msg", "")}
        for err in exc.errors()
    ]
    return envelope(request, 422, "validation_error", detail)


def install(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
