"""The Alpha Vantage HTTP client.

One module talks to Alpha Vantage, the same way ``model_training/api_client.py``
is the only module that talks to the backend. Stdlib only -- ``urllib.request``
-- so a feature most runs never touch adds no dependency to the backend.

Three things make this more than a wrapper around ``urlopen``:

**Alpha Vantage answers HTTP 200 for its failures.** A spent quota, an unknown
symbol and an endpoint your plan does not include all arrive as a 200 with a
one-key JSON body (``Error Message`` / ``Note`` / ``Information``). Left alone
that surfaces as a ``KeyError`` five frames later, so :func:`_check_envelope`
turns each into a named error that says what to do about it.

**The free plan allows 25 requests a day, about one a second.** Calls are spaced
by ``settings.live_min_interval`` and counted against a per-run budget, so a
fetch that would burn the day's allowance stops with a message instead of
half-writing a dataset.

**Company reports change quarterly; prices change constantly.** The four
fundamentals endpoints are cached on disk with a TTL, which is what makes a
daily refresh affordable -- see :data:`CACHEABLE`. Bars, quotes and news are
never cached.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..config import ALPHAVANTAGE_KEY_NAMES, Settings, get_settings
from ..errors import (
    StockiCredentialsError,
    StockiLiveError,
    StockiPlanError,
    StockiQuotaError,
)

logger = logging.getLogger("stocki.live")

BASE_URL = "https://www.alphavantage.co/query"

#: Company reports, safe to reuse between runs. Everything else is fetched live.
CACHEABLE = frozenset({"OVERVIEW", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"})

#: Alpha Vantage writes missing numbers as the string "None", and sometimes "-".
MISSING = frozenset({"", "-", "none", "null", "nan"})

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


class AlphaVantageError(StockiLiveError):
    """Alpha Vantage could not be reached, or answered with something unusable."""


def _looks_like(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _check_envelope(function: str, payload: object) -> dict:
    """Turn the 200-with-an-excuse bodies into errors that name the fix."""
    if not isinstance(payload, dict):
        raise AlphaVantageError(
            f"{function} returned {type(payload).__name__}, expected an object"
        )

    note = str(
        payload.get("Error Message")
        or payload.get("Note")
        or payload.get("Information")
        or ""
    ).strip()
    if not note:
        return payload

    # Order matters: the rate-limit note also advertises the premium plans, so
    # the endpoint gate has to be recognised by its own distinctive phrase.
    if _looks_like(note, "premium endpoint"):
        raise StockiPlanError(
            f"Alpha Vantage does not serve {function} on this API key's plan. "
            f"It said: {note}"
        )
    if _looks_like(
        note, "rate limit", "call frequency", "sparingly", "per day", "per minute"
    ):
        raise StockiQuotaError(f"Alpha Vantage rate-limited {function}. It said: {note}")
    raise AlphaVantageError(f"Alpha Vantage rejected {function}: {note}")


def _is_burst_limit(exc: Exception) -> bool:
    """A per-second burst is worth waiting out; a spent daily quota is not."""
    return _looks_like(str(exc), "sparingly", "per second", "call frequency")


@dataclass(frozen=True)
class CacheEntry:
    """A cached payload and when it was written."""

    payload: dict
    fetched_at: datetime

    def age_hours(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(UTC)) - self.fetched_at).total_seconds() / 3600.0


class AlphaVantage:
    """A budgeted, throttled, partly-cached Alpha Vantage session.

    One instance per fetch run: it carries the request count, so the budget is
    enforced across every ticker rather than per call.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        cache_hours: float | None = None,
        budget: int | None = None,
        min_interval: float | None = None,
        refresh: bool = False,
        timeout: float = 60.0,
    ):
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.alphavantage_key
        self.cache_dir = Path(cache_dir or self.settings.live_cache_dir)
        self.cache_hours = self.settings.live_cache_hours if cache_hours is None else cache_hours
        self.budget = self.settings.live_request_budget if budget is None else budget
        self.min_interval = (
            self.settings.live_min_interval if min_interval is None else min_interval
        )
        self.refresh = refresh
        self.timeout = timeout

        self.requests_made = 0
        self.cache_hits = 0
        self._last_call: float | None = None

    # -- credentials ---------------------------------------------------

    def require_key(self) -> str:
        """The API key, or an error naming every variable that was checked."""
        if not self.api_key:
            names = ", ".join(ALPHAVANTAGE_KEY_NAMES)
            raise StockiCredentialsError(
                "no Alpha Vantage API key found -- put one in backend/.env as "
                f"ALPHA_ADVANTAGE_KEY=<your key> (these names are read, in order: {names}). "
                "Free keys are issued at https://www.alphavantage.co/support/#api-key"
            )
        return self.api_key

    # -- transport -----------------------------------------------------

    def _throttle(self) -> None:
        if self._last_call is None or self.min_interval <= 0:
            return
        waited = time.monotonic() - self._last_call
        if waited < self.min_interval:
            time.sleep(self.min_interval - waited)

    def _spend(self, function: str) -> None:
        if self.budget is not None and self.requests_made >= self.budget:
            raise StockiQuotaError(
                f"stopping before request {self.requests_made + 1} ({function}): this run has "
                f"spent its budget of {self.budget} Alpha Vantage calls. The free plan allows "
                "25 a day. Fetch fewer tickers, keep the fundamentals cache warm, or raise "
                "STOCKI_LIVE_REQUEST_BUDGET."
            )
        self.requests_made += 1

    def _get(self, params: dict) -> bytes:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE_URL}?{query}"
        delay = 2.0
        last_error = ""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                request = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code not in _RETRYABLE_STATUS:
                    raise AlphaVantageError(f"{last_error} from Alpha Vantage") from exc
            except urllib.error.URLError as exc:
                last_error = str(exc.reason)
            except TimeoutError:
                last_error = f"timed out after {self.timeout}s"
            finally:
                self._last_call = time.monotonic()

            if attempt < _MAX_ATTEMPTS:
                time.sleep(delay)
                delay *= 2

        raise AlphaVantageError(
            f"cannot reach Alpha Vantage after {_MAX_ATTEMPTS} attempts ({last_error}) "
            "-- check the network connection"
        )

    def request(self, function: str, **params) -> dict:
        """One call, spending budget and honouring the cache for company reports."""
        symbol = str(params.get("symbol") or params.get("tickers") or "_")

        if function in CACHEABLE and not self.refresh:
            entry = self._read_cache(function, symbol)
            if entry is not None and entry.age_hours() < self.cache_hours:
                self.cache_hits += 1
                logger.debug("cache hit %s %s (%.1fh old)", function, symbol, entry.age_hours())
                return entry.payload

        checked: dict = {}
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._spend(function)
            body = self._get({"function": function, "apikey": self.require_key(), **params})
            try:
                payload = json.loads(body)
            except ValueError as exc:
                raise AlphaVantageError(
                    f"{function} did not return JSON: {body[:200]!r}"
                ) from exc

            try:
                checked = _check_envelope(function, payload)
            except StockiQuotaError as exc:
                # A per-second burst clears on its own; the daily cap does not.
                if attempt < _MAX_ATTEMPTS and _is_burst_limit(exc):
                    logger.warning("Alpha Vantage burst limit hit; backing off")
                    time.sleep(max(self.min_interval, 1.0) * 4 * attempt)
                    continue
                raise
            break

        if function in CACHEABLE:
            self._write_cache(function, symbol, checked)
        return checked

    # -- cache ---------------------------------------------------------

    def _cache_path(self, function: str, symbol: str) -> Path:
        safe = "".join(c for c in symbol if c.isalnum() or c in "-_.") or "_"
        return self.cache_dir / safe / f"{function}.json"

    def _read_cache(self, function: str, symbol: str) -> CacheEntry | None:
        path = self._cache_path(function, symbol)
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry(
                payload=stored["payload"],
                fetched_at=datetime.fromisoformat(stored["fetched_at"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None  # absent, corrupt, or written by an older version

    def _write_cache(self, function: str, symbol: str, payload: dict) -> None:
        path = self._cache_path(function, symbol)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "function": function,
                        "symbol": symbol,
                        "fetched_at": datetime.now(UTC).isoformat(),
                        "payload": payload,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # a read-only checkout should not fail the fetch
            logger.warning("could not cache %s %s: %s", function, symbol, exc)

    # -- endpoints -----------------------------------------------------

    def intraday(
        self,
        symbol: str,
        *,
        interval: str = "5min",
        month: str | None = None,
        outputsize: str = "full",
        extended_hours: bool = False,
    ) -> dict:
        """5-minute bars. ``month`` is ``YYYY-MM`` and selects a historical month.

        Regular hours only by default, because a session in ``data/`` is the 78
        bars from 09:30 to 15:55 Eastern and nothing else.
        """
        return self.request(
            "TIME_SERIES_INTRADAY",
            symbol=symbol,
            interval=interval,
            month=month,
            outputsize=outputsize,
            extended_hours="true" if extended_hours else "false",
        )

    def overview(self, symbol: str) -> dict:
        return self.request("OVERVIEW", symbol=symbol)

    def income_statement(self, symbol: str) -> dict:
        return self.request("INCOME_STATEMENT", symbol=symbol)

    def balance_sheet(self, symbol: str) -> dict:
        return self.request("BALANCE_SHEET", symbol=symbol)

    def cash_flow(self, symbol: str) -> dict:
        return self.request("CASH_FLOW", symbol=symbol)

    def global_quote(self, symbol: str) -> dict:
        return self.request("GLOBAL_QUOTE", symbol=symbol)

    def news(
        self,
        tickers,
        *,
        time_from: datetime | None = None,
        time_to: datetime | None = None,
        limit: int = 1000,
    ) -> dict:
        """News for one or many tickers in a single call.

        Every article carries a ``ticker_sentiment`` list, so one request covers
        the whole universe and ``fields.news_by_ticker`` splits the feed
        afterwards. That is the difference between 1 request and 10.
        """
        joined = tickers if isinstance(tickers, str) else ",".join(tickers)
        return self.request(
            "NEWS_SENTIMENT",
            tickers=joined,
            time_from=_news_time(time_from),
            time_to=_news_time(time_to),
            limit=limit,
            sort="LATEST",
        )


def _news_time(moment: datetime | None) -> str | None:
    """Alpha Vantage wants YYYYMMDDTHHMM, in UTC."""
    if moment is None:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y%m%dT%H%M")


def parse_number(raw) -> float | None:
    """An Alpha Vantage number as a float, with its several spellings of missing."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text.lower() in MISSING:
        return None
    try:
        return float(text.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def months_between(first, last) -> list[str]:
    """The ``YYYY-MM`` slices covering a date range, oldest first.

    Intraday history is served a month at a time, so a request that spans a
    month boundary costs two calls.
    """
    if first is None or last is None:
        return []
    months, cursor = [], first.replace(day=1)
    end = last.replace(day=1)
    while cursor <= end:
        months.append(cursor.strftime("%Y-%m"))
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    return months
