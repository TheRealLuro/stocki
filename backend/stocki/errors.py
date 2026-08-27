"""Errors that tell you what to do next.

Cason and Nathaniel should never have to read a psycopg traceback to find out
that the database is not running.
"""

from __future__ import annotations


class StockiError(Exception):
    """Base class for every error this package raises on purpose."""


class StockiConnectionError(StockiError):
    """Postgres is unreachable."""


class StockiEmptyError(StockiError):
    """The database is up but holds no data yet."""


class StockiValidationError(StockiError):
    """A session file failed validation and was not loaded."""


class StockiLiveError(StockiError):
    """Live data could not be fetched from the upstream provider."""


class StockiCredentialsError(StockiLiveError):
    """No Alpha Vantage API key is configured."""


class StockiQuotaError(StockiLiveError):
    """The provider refused the request because the plan's quota is spent."""


class StockiPlanError(StockiLiveError):
    """The endpoint exists but this API plan does not include it."""
