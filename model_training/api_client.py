"""HTTP client for the stocki backend API.

This is the only file that talks to the network. It reads bars from
``/api/v1/bars`` -- the paginated route, which is the only one that can serve
the whole dataset (``/api/v1/dataset/windows`` takes no ``offset`` and is capped
at 500 windows, so it is useful for cross-checking but not for training).

Stdlib only: ``urllib.request``, so there is no extra dependency to install and
no database driver involved.

    from api_client import fetch_bars, health
    bars = fetch_bars()          # structured array, 14,586 rows
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import config

#: dtype of the tidy bars table -- one row per 5-minute bar, matching the
#: columns of ``GET /api/v1/bars`` and the ``v_bars`` view behind it.
BAR_DTYPE = np.dtype(
    [
        ("ticker", "U8"),
        ("day", "i2"),
        ("timestamp", "datetime64[ns]"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("volume", "i8"),
    ]
)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class StockiAPIError(RuntimeError):
    """The API could not be reached, or answered with something unusable."""


# =====================================================================
# Transport
# =====================================================================


def _url(path: str, params: dict | None = None, base_url: str | None = None) -> str:
    base = (base_url or config.API_BASE_URL).rstrip("/")
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
    return f"{base}{path}" + (f"?{query}" if query else "")


def _request(
    path: str,
    params: dict | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> bytes:
    """GET ``path``, retrying transient failures with exponential backoff.

    Raises ``StockiAPIError`` -- with the API's own ``detail`` message when it
    sent one -- rather than letting a urllib traceback escape.
    """
    url = _url(path, params, base_url)
    timeout = config.API_TIMEOUT if timeout is None else timeout
    delay = config.API_RETRY_BACKOFF
    last_error = ""

    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"Accept": "*/*"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            detail = _detail(body)
            if exc.code not in _RETRYABLE_STATUS:
                raise StockiAPIError(f"{exc.code} from {url}: {detail}") from exc
            last_error = f"{exc.code}: {detail}"
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except TimeoutError:
            last_error = f"timed out after {timeout}s"

        if attempt < config.API_MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    raise StockiAPIError(
        f"cannot reach the stocki API at {url} after {config.API_MAX_RETRIES} attempts "
        f"({last_error}) -- is `docker compose up -d` running? Override the base URL "
        f"with STOCKI_API_URL."
    )


def _detail(body: bytes) -> str:
    """Pull ``detail`` out of the API's ``{error, detail, request_id}`` envelope."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body[:200].decode("utf-8", "replace")
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)
    return str(payload)


def get_json(path: str, params: dict | None = None, base_url: str | None = None) -> dict:
    """GET ``path`` and decode the JSON body."""
    body = _request(path, params, base_url)
    try:
        return json.loads(body)
    except ValueError as exc:
        raise StockiAPIError(f"{path} did not return JSON: {body[:200]!r}") from exc


# =====================================================================
# Endpoints
# =====================================================================


def health(base_url: str | None = None) -> dict:
    """``{status, database, bar_count}``. The cheapest way to check liveness."""
    return get_json("/health", base_url=base_url)


def dataset_stats(base_url: str | None = None, **params) -> dict:
    """``/api/v1/dataset/stats`` -- the data card. Takes the load_stocki params."""
    return get_json("/api/v1/dataset/stats", params, base_url=base_url)


def dataset_windows_npz(
    limit: int = 500,
    base_url: str | None = None,
    **params,
) -> dict[str, np.ndarray]:
    """The backend's own pre-built windows, as arrays.

    Capped at 500 and with no ``offset``, so this can only ever return the first
    ``limit`` windows -- it exists to *verify* the local transform against the
    canonical one (see ``dataloader.verify_against_api``), not to feed training.
    """
    body = _request(
        "/api/v1/dataset/windows",
        {**params, "limit": limit, "format": "npz"},
        base_url=base_url,
    )
    with np.load(io.BytesIO(body), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def fetch_bars_uncached(
    base_url: str | None = None,
    page_size: int = config.API_PAGE_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """Page through ``/api/v1/bars`` and return every bar as a structured array.

    The route orders by ``ticker, day, timestamp``, which is the same order the
    backend windows in -- so windows built from this array line up with the
    backend's index for index.
    """
    rows: list[tuple] = []
    offset, total = 0, None

    while total is None or offset < total:
        page = get_json(
            "/api/v1/bars", {"limit": page_size, "offset": offset}, base_url=base_url
        )
        total = page["total"]
        items = page["items"]
        if not items:
            break

        rows.extend(
            (
                item["ticker"],
                item["day"],
                np.datetime64(item["timestamp"].rstrip("Z"), "ns"),
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
            )
            for item in items
        )
        offset += len(items)
        if verbose:
            print(f"\r  fetched {offset}/{total} bars", end="", flush=True)

    if verbose:
        print()
    if total is not None and len(rows) != total:
        raise StockiAPIError(f"API reported {total} bars but only {len(rows)} came back")

    return np.array(rows, dtype=BAR_DTYPE)


# =====================================================================
# Cache
# =====================================================================


def _cache_path(cache_dir: Path | str | None = None) -> Path:
    return Path(cache_dir or config.CACHE_DIR) / "bars.npz"


def fetch_bars(
    base_url: str | None = None,
    cache_dir: Path | str | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Every bar the API has, cached on disk between runs.

    The cache is validated against ``/health``'s ``bar_count`` -- one cheap
    request -- so a re-ingest on the backend invalidates it automatically. If the
    API is unreachable but a cache exists, the cache is used with a warning
    instead of failing the run.
    """
    path = _cache_path(cache_dir)
    cached: np.ndarray | None = None

    if path.is_file() and not refresh:
        try:
            with np.load(path, allow_pickle=False) as archive:
                cached = archive["bars"]
        except (OSError, ValueError, KeyError):
            cached = None  # corrupt or written by an older version -- refetch

    expected: int | None = None
    try:
        expected = int(health(base_url)["bar_count"])
    except (StockiAPIError, KeyError, TypeError, ValueError) as exc:
        if cached is None:
            raise
        print(f"warning: could not reach the API ({exc}); using cached bars at {path}")
        return cached

    if cached is not None and len(cached) == expected:
        if verbose:
            print(f"using {len(cached)} cached bars from {path}")
        return cached

    if verbose:
        reason = "no cache" if cached is None else f"cache holds {len(cached)}, API has {expected}"
        print(f"fetching bars from {base_url or config.API_BASE_URL} ({reason})")

    bars = fetch_bars_uncached(base_url, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bars=bars)
    if verbose:
        print(f"cached {len(bars)} bars to {path}")
    return bars


if __name__ == "__main__":
    print(f"API: {config.API_BASE_URL}")
    print(f"health: {health()}")
    table = fetch_bars()
    tickers, counts = np.unique(table["ticker"], return_counts=True)
    print(f"\n{len(table)} bars, {len(tickers)} tickers, days "
          f"{table['day'].min()}-{table['day'].max()}")
    for ticker, count in zip(tickers, counts):
        days = np.unique(table["day"][table["ticker"] == ticker])
        print(f"  {ticker:<6} {count:>6} bars  {len(days):>2} sessions (days {days.min()}-{days.max()})")
