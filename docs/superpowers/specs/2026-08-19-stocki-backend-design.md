# Stocki backend design

**Date:** 2026-08-19
**Scope:** the data pipeline and backend only — Cason owns the model, Nathaniel owns the dashboard.
**Status:** implemented and verified against the real dataset.

---

## What this is

A pipeline that takes `data/<TICKER>/day<N>.csv`, loads it into Postgres, and exposes it
through two contracts:

- **`stocki.datasets`** — a Python import that returns CNN-ready numpy, scikit-learn style.
- **`/api/v1/*`** — a read-only FastAPI service for the dashboard.

Both sit on the same database and the same transform code, so they cannot disagree.

```
data/<TICKER>/day<N>.csv
        |  validate, then load (idempotent, per-file transaction)
        v
    bars_raw  --> v_bars, v_fundamentals, v_news, v_coverage
        |                       |
        |                       +--> stocki.api  (FastAPI, read-only role)
        +--> stocki.datasets  ------> numpy / pandas
```

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where training data comes from | Python package reading Postgres directly | Fast, notebook-friendly, no HTTP hop. The API is for the dashboard. |
| Where windowing/labelling lives | In Python, at load time — not materialised in the DB | 14,586 rows is milliseconds of numpy. Changing a feature is a keyword argument, not a migration. |
| Label | Next-bar direction, configurable horizon and threshold | Matches the README task, gives a baseline immediately, stays flexible. |
| Features | OHLCV + `log_return`, `hl_range`, `co_range`, window-z normalised | Scale-free shape for the CNN. |
| Schema | `bars_raw` mirrors the CSVs column for column | `SELECT *` diffs against the file; `stocki verify` proves it. |
| Protections | Validation, safe errors, CORS, rate limits | Agreed scope. No auth, no user accounts. |

### Deviations from the first sketch, and why

- **118 CSV columns, not 120.** Counted from the files.
- **`co_range` replaced `vol_z`.** A session-wide volume z-score would read bars after the
  window (look-ahead), and per-window volume z-scoring is already what `window-z`
  normalisation does to the volume channel — so it was both leaky and redundant.
  `co_range = (close - open) / open` is bar-local and adds real information.
- **Two identifiers renamed:** `Dividends` → `dividends`, `Stock Splits` → `stock_splits`.
  Postgres folds unquoted identifiers to lower case and a space would force double-quoting
  in every query forever. Everything else is byte-identical.
- **Three provenance columns added:** `day` (it exists only in the *filename*), `source_file`,
  `ingested_at`.

## Contracts

### `stocki.datasets` (the model)

```python
ds = load_stocki()      # no arguments, no DSN
ds.data                 # float32 (8602, 32, 8)
ds.target               # int8 (8602,) — 1 = UP
ds.ticker / ds.day / ds.timestamps / ds.feature_names / ds.DESCR
```

Parameters: `tickers, days, window, horizon, threshold, channels, normalize, subset,
test_days, channels_first, as_frame`.

Exploration: `load_bars`, `load_raw`, `load_fundamentals`, `load_panel`, `coverage`,
`describe`. All return plain numpy or pandas.

### `/api/v1` (the dashboard)

`/health`, `/tickers`, `/coverage`, `/bars/{ticker}/{day}`, `/bars`,
`/fundamentals/{ticker}/{day}`, `/news/{ticker}/{day}`, `/dataset/stats`,
`/dataset/windows`. OpenAPI at `/openapi.json`.

Every error is `{error, detail, request_id}`. A 500 logs the traceback server-side and
returns none of it.

## Leakage controls

Three, layered:

1. **No window crosses a session boundary.** Days are not contiguous; windowing runs per
   `(ticker, day)` group.
2. **Splits are by day, not by window.** `train.timestamps.max() < test.timestamps.min()` is
   asserted in the test suite.
3. **Normalisation is per window.** Dataset-wide statistics would fold the test set into the
   training set.

Derived channels only read the current bar or the one immediately before it.

## Verified numbers

Confirmed by `tests/test_golden.py` against the real dataset:

| | Value |
|---|---|
| Session files | 187 (not 200 — AAPL has days 13–20 only, MSFT is missing day 1) |
| Bars per session | 78 (13:30–19:55 UTC) |
| Total bars | 14,586 |
| Default windows | 8,602 of shape (32, 8) |
| Train / test | 6,762 / 1,840 (days 1–16 / 17–20) |
| Class balance | 49.8% up |
| Round-trip to CSV | exact, all 187 files |

**Caveat for evaluation:** AAPL contributes 4 training days against 4 test days, while every
other ticker contributes 16 against 4. This is recorded in `ds.DESCR` and the README.

## Containers

`docker compose up -d` starts `postgres`, runs a one-shot `init` (`stocki ingest`), then
starts `api` on the `stocki-net` bridge network. Postgres is not published in the base
compose file; `docker-compose.override.yml` publishes 5434 for host-side training and psql.
The API connects as `stocki_ro`, a role holding `SELECT` and nothing else.

## Out of scope

Model architecture, training loop, and evaluation (Cason). Dashboard UI (Nathaniel).
Authentication and user accounts — deliberately excluded as a separate subsystem.
