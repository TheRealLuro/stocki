"""The `stocki` command.

    stocki ingest    read data/ into Postgres (idempotent)
    stocki verify    prove the table still matches the CSVs
    stocki stats     print the data card
    stocki serve     run the API

Exit codes: 0 success, 1 the data disagrees with the files, 2 the tool could
not run at all (database down, nothing ingested yet).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, get_settings
from .datasets.loaders import describe
from .db.session import apply_schema, connect
from .errors import StockiError
from .ingest.load import ingest_directory, verify_directory

EXIT_OK = 0
EXIT_DATA_PROBLEM = 1
EXIT_CANNOT_RUN = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stocki", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="load data/<TICKER>/day<N>.csv into Postgres")
    ingest.add_argument("--data-dir", type=Path, default=None)

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


def _ingest(args, settings: Settings) -> int:
    data_dir = _data_dir(args, settings)
    with connect(settings) as conn:
        apply_schema(conn)
        report = ingest_directory(conn, data_dir)
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


HANDLERS = {"ingest": _ingest, "verify": _verify, "stats": _stats, "serve": _serve}


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
