"""Live market data from Alpha Vantage, in the format ``data/`` already uses.

    stocki fetch --tickers NVDA,AAPL --days 1 && stocki ingest

The point of this package is that it produces nothing new: it writes
``data/<TICKER>/day<N>.csv`` files with the same 118 columns, the same 78 bars,
the same UTC stamps and the same day numbering as the committed ones. Everything
downstream -- ingest, the views, the API, ``stocki.datasets``, the training
package, the ONNX export -- carries on unchanged, because from where they stand
nothing did change.

    from stocki.live import AlphaVantage, fetch_sessions

    report = fetch_sessions(["NVDA"], days=1)
    print(report.summary())

Nothing here is imported by ``stocki.datasets`` or ``stocki.api``, so a clone
with no API key behaves exactly as it did before.
"""

from .client import AlphaVantage, AlphaVantageError, months_between, parse_number
from .fetch import (
    Bar,
    FetchReport,
    existing_name_cn,
    fetch_sessions,
    known_tickers,
    parse_intraday,
    session_rows,
    static_columns,
    verify_written,
    write_session,
)
from .sessions import (
    BARS_PER_SESSION,
    DayIndex,
    DayNumberingError,
    eastern_to_utc,
    expected_timestamps,
    news_window,
    recent_weekdays,
    session_window,
)

__all__ = [
    "BARS_PER_SESSION",
    "AlphaVantage",
    "AlphaVantageError",
    "Bar",
    "DayIndex",
    "DayNumberingError",
    "FetchReport",
    "eastern_to_utc",
    "existing_name_cn",
    "expected_timestamps",
    "fetch_sessions",
    "known_tickers",
    "months_between",
    "news_window",
    "parse_intraday",
    "parse_number",
    "recent_weekdays",
    "session_rows",
    "session_window",
    "static_columns",
    "verify_written",
    "write_session",
]
