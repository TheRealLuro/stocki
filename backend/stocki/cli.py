"""The `stocki` command.

    stocki fetch     pull live sessions from Alpha Vantage into data/
    stocki ingest    read data/ into Postgres (idempotent)
    stocki verify    prove the table still matches the CSVs
    stocki stats     print the data card
    stocki serve     run the API

`fetch` writes the same files `ingest` already reads, so live data reaches the
model down the path the committed data always took:

    stocki fetch --days 1 && stocki ingest

Exit codes: 0 success, 1 the data disagrees with the files, 2 the tool could
not run at all (database down, nothing ingested yet, no API key).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import Settings, get_settings
from .datasets.loaders import describe
from .db.session import apply_schema, connect
from .errors import StockiError
from .ingest.load import ingest_directory, verify_directory
from .ingest.validate import EXPECTED_BARS

EXIT_OK = 0
EXIT_DATA_PROBLEM = 1
EXIT_CANNOT_RUN = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stocki", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser(
        "fetch", help="pull live sessions from Alpha Vantage into data/<TICKER>/day<N>.csv"
    )
    fetch.add_argument(
        "--tickers",
        default=None,
        help="comma-separated; default is every ticker folder already in data/",
    )
    fetch.add_argument(
        "--days", type=int, default=1, help="how many recent weekdays to fetch (default 1)"
    )
    fetch.add_argument("--date", default=None, help="a single YYYY-MM-DD session instead")
    fetch.add_argument("--data-dir", type=Path, default=None)
    fetch.add_argument(
        "--allow-partial",
        action="store_true",
        help="write a session with fewer than 78 bars (the market is still open)",
    )
    fetch.add_argument("--overwrite", action="store_true", help="replace existing day files")
    fetch.add_argument(
        "--refresh", action="store_true", help="ignore the fundamentals cache and refetch"
    )
    fetch.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="skip the 4 company-report calls per ticker; those columns are left empty",
    )
    fetch.add_argument("--no-news", action="store_true", help="skip the news call")
    fetch.add_argument(
        "--quote",
        action="store_true",
        help="spend a call per ticker on GLOBAL_QUOTE for si_current_price "
        "(default: use the session's last close)",
    )
    fetch.add_argument(
        "--budget",
        type=int,
        default=None,
        help="stop after this many Alpha Vantage requests (default 25, the free plan's day)",
    )
    fetch.add_argument("--dry-run", action="store_true", help="fetch and validate, write nothing")
    fetch.add_argument(
        "--ingest", action="store_true", help="run ingest afterwards if every file validated"
    )

    ingest = commands.add_parser("ingest", help="load data/<TICKER>/day<N>.csv into Postgres")
    ingest.add_argument("--data-dir", type=Path, default=None)
    ingest.add_argument(
        "--expected-bars",
        type=int,
        default=EXPECTED_BARS,
        help=f"exact bar count every session must hold (default {EXPECTED_BARS}). It "
        "applies to every file in --data-dir, so point that at a directory of "
        "equally-short sessions when ingesting `fetch --allow-partial` output",
    )

    verify = commands.add_parser("verify", help="compare every loaded row against its CSV")
    verify.add_argument("--data-dir", type=Path, default=None)

    commands.add_parser("stats", help="print the data card")

    serve = commands.add_parser("serve", help="run the API with uvicorn")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def _data_dir(args, settings: Settings) -> Path:
    return args.data_dir if args.data_dir else settings.data_dir


def _fetch(args, settings: Settings) -> int:
    from .live import AlphaVantage, fetch_sessions, known_tickers

    data_dir = _data_dir(args, settings)
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else known_tickers(data_dir)
    )
    if not tickers:
        print(
            f"stocki: no tickers to fetch -- {data_dir} holds no ticker folders, "
            "so pass --tickers NVDA,AAPL",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    dates = [datetime.strptime(args.date, "%Y-%m-%d").date()] if args.date else None
    client = AlphaVantage(settings, refresh=args.refresh, budget=args.budget)

    report = fetch_sessions(
        tickers,
        dates=dates,
        days=args.days,
        data_dir=data_dir,
        client=client,
        settings=settings,
        with_fundamentals=not args.no_fundamentals,
        with_news=not args.no_news,
        with_quote=args.quote,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print(report.summary())
    if not report.ok:
        print("\nrejected sessions were not written; nothing downstream saw them")
        return EXIT_DATA_PROBLEM

    if args.ingest and report.written and not args.dry_run:
        if args.allow_partial:
            # `ingest` validates every file against one exact bar count, so a
            # directory holding both 78-bar history and a short live session
            # cannot satisfy it either way. Say so instead of quarantining the
            # whole dataset.
            print(
                "\nnot ingesting automatically: --allow-partial wrote a short session, and "
                "`stocki ingest` checks every file against a single exact bar count. Ingest "
                "the partial session from a directory of its own:\n"
                "  stocki ingest --data-dir <dir> --expected-bars <bars>"
            )
            return EXIT_OK
        print()
        return _ingest(args, settings)
    return EXIT_OK


def _ingest(args, settings: Settings) -> int:
    data_dir = _data_dir(args, settings)
    expected_bars = getattr(args, "expected_bars", None) or EXPECTED_BARS
    with connect(settings) as conn:
        apply_schema(conn)
        report = ingest_directory(conn, data_dir, expected_bars=expected_bars)
        conn.commit()

    print(report.summary())
    if not report.ok:
        print("\nquarantined files were not written; fix them and run ingest again")
        return EXIT_DATA_PROBLEM
    return EXIT_OK


def _verify(args, settings: Settings) -> int:
    data_dir = _data_dir(args, settings)
    with connect(settings) as conn:
        problems = verify_directory(conn, data_dir)

    if problems:
        print(f"{len(problems)} mismatch(es) between bars_raw and {data_dir}:")
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return EXIT_DATA_PROBLEM

    print(f"bars_raw matches every CSV under {data_dir}")
    return EXIT_OK


def _stats(_args, settings: Settings) -> int:
    with connect(settings) as conn:
        print(describe(conn=conn))
    return EXIT_OK


def _serve(args, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "stocki.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return EXIT_OK


HANDLERS = {
    "fetch": _fetch,
    "ingest": _ingest,
    "verify": _verify,
    "stats": _stats,
    "serve": _serve,
}


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    try:
        return HANDLERS[args.command](args, settings)
    except StockiError as exc:
        print(f"stocki: {exc}", file=sys.stderr)
        return EXIT_CANNOT_RUN


if __name__ == "__main__":
    raise SystemExit(main())
