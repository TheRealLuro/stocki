# Stocki backend

The data pipeline and API. Postgres holds the bars, `stocki.datasets` turns them
into CNN-ready tensors, and FastAPI serves the same data to the dashboard.

```
Alpha Vantage  ->  data/<TICKER>/day<N>.csv  ->  bars_raw  ->  stocki.datasets  ->  numpy / pandas
 (stocki fetch)                                       \                      \
                                                       v_bars, v_fundamentals  ->  /api/v1/...
```

`stocki fetch` writes the same files the repo already commits, so live data
reaches the model down the path the committed data always took — see
[Live data](#live-data).

---

## Quick start

```bash
docker compose up -d
pip install -r requirements.txt
```

`docker compose up -d` starts Postgres, applies the schema, loads every CSV, and
starts the API on <http://localhost:8000>. Interactive docs are at
<http://localhost:8000/docs>.

Verify it worked — `stocki stats` should print exactly this:

```
Stocki intraday dataset
=======================
187 sessions across 10 tickers, days 1-20
14586 bars total

coverage by ticker:
  AAPL     8 sessions (days 13-20)
  AMZN    20 sessions (days 1-20)
  GOOGL   20 sessions (days 1-20)
  JNJ     20 sessions (days 1-20)
  JPM     20 sessions (days 1-20)
  META    20 sessions (days 1-20)
  MSFT    19 sessions (days 2-20)
  NVDA    20 sessions (days 1-20)
  TSLA    20 sessions (days 1-20)
  V       20 sessions (days 1-20)

gaps -- these tickers contribute fewer samples:
  AAPL is missing day(s) 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  MSFT is missing day(s) 1
```

Different numbers mean the ingest is incomplete — run `stocki verify`.

### If Postgres is on a different port

The defaults assume `localhost:5434` (5432 and 5433 are often taken by other
local stacks). Override any of these in a `.env` file at the repo root, or as
environment variables — see `.env.example`:

```bash
STOCKI_DB_HOST=localhost
STOCKI_DB_PORT=5434
STOCKI_DB_NAME=stocki
STOCKI_DB_USER=stocki
STOCKI_DB_PASSWORD=stocki
```

If the database is down you get a message, not a traceback:

```
StockiConnectionError: cannot reach Postgres at localhost:5434 --
is `docker compose up -d` running?
```

---

## For the model: `stocki.datasets`

```python
from stocki.datasets import load_stocki

ds = load_stocki()          # no arguments, no DSN, no config
```

`ds` is an sklearn-style `Bunch` — attribute *and* dict access both work.

| Attribute | Type | Meaning |
|---|---|---|
| `ds.data` | `float32 (8602, 32, 8)` | windows x timesteps x channels |
| `ds.target` | `int8 (8602,)` | 1 = UP, 0 = DOWN |
| `ds.feature_names` | 8 strings | `open, high, low, close, volume, log_return, hl_range, co_range` |
| `ds.ticker` | `(8602,)` str | which stock each window came from |
| `ds.day` | `(8602,)` int | day number, for custom grouping |
| `ds.timestamps` | `(8602,)` datetime64 | last bar of each window, UTC |
| `ds.DESCR` | str | printable data card |

Everything is a keyword argument, so you never edit pipeline code:

```python
load_stocki(
    tickers=None,          # None = all 10, or "NVDA", or ["NVDA", "AAPL"]
    days=None,
    window=32,             # bars per window
    horizon=1,             # how many bars ahead the label looks
    threshold=0.0,         # fractional move required, e.g. 0.001 = 0.1%
    channels="default",    # or an explicit list of names
    normalize="window-z",  # or None for raw prices
    subset="all",          # "train" | "test"
    test_days=4,
    channels_first=False,  # True -> (N, 8, 32) for torch Conv1d
    as_frame=False,        # True -> ds.frame, a tidy DataFrame
)
```

### What you actually get

```python
>>> ds = load_stocki()
>>> ds.data.shape, ds.data.dtype
((8602, 32, 8), dtype('float32'))
>>> ds.target.shape, ds.target.dtype
((8602,), dtype('int8'))
>>> ds.data.nbytes / 1024**2
8.40                                  # the whole dataset fits in 8.4 MB
>>> ds.data.flags["C_CONTIGUOUS"]
True                                  # feeds a model without a copy
```

One real window — `ds.data[0]`, the first six of its 32 timesteps:

```
             open      high       low     close    volume  log_return  hl_range  co_range
t=0        0.5070    0.1746   -0.4219   -0.0837    4.8535     -0.0266    1.4869   -1.0630
t=1       -0.0838   -0.4475   -1.9963   -0.7941    2.0178     -1.3266    3.9856   -1.2786
t=2       -0.7846   -0.7355   -0.8354   -0.4675    0.1494      0.5716    0.3449    0.5739
t=3       -0.4627   -0.8149   -0.7240   -1.1820   -0.1354     -1.3361   -0.1230   -1.2952
t=4       -1.1879   -0.7773   -1.0262   -0.7859    0.2550      0.6999    0.7307    0.7297
t=5       -0.7724   -0.8252   -0.4298   -0.5981   -0.3508      0.3173   -0.8974    0.3161
...
t=31      -0.0301    0.2247    0.3654    0.3818   -0.5267      0.7188   -0.3853    0.7397
```

The values look small because normalisation happens **inside each window**:

```python
>>> ds.data[0].mean(axis=0)
array([ 0., -0., -0.,  0., -0., -0.,  0.,  0.], dtype=float32)
>>> ds.data[0].std(axis=0)
array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
```

Each channel is centred at 0 with unit spread using **only the 32 bars in that
window**. Dataset-wide statistics would fold the test set into training, so they
are never used. Pass `normalize=None` if you want raw prices.

### Where a label comes from

Worth tracing once, because this is the part that is easy to get wrong:

```python
>>> raw = load_stocki(tickers="AAPL", days=[13], normalize=None)
>>> closes = load_bars(tickers="AAPL", days=[13])["close"].to_numpy()
>>> closes[29:36]
array([307.97, 308.70, 309.20, 309.095, 308.925, 309.03, 308.6399])
```

Window 0 covers bars 0–31, so it **ends** at bar 31 (`close = 309.200`). With
`horizon=1` the label compares bar 32 (`close = 309.095`). Lower, so:

```python
>>> raw.target[0]
0                                     # DOWN
>>> raw.data[0, -1, 3]                # last close *inside* the window
309.20001220703125                    # bar 31 -- bar 32 is not in the input
```

The bar the label is derived from is never part of the window. The model cannot
see its own answer.

### Parameters and the shapes they produce

Measured on the real dataset:

| Call | `ds.data.shape` | UP rate |
|---|---|---|
| `load_stocki()` | `(8602, 32, 8)` | 0.498 |
| `load_stocki(window=16)` | `(11594, 16, 8)` | 0.496 |
| `load_stocki(window=60)` | `(3366, 60, 8)` | 0.505 |
| `load_stocki(horizon=6)` | `(7667, 32, 8)` | 0.501 |
| `load_stocki(channels=["close", "volume"])` | `(8602, 32, 2)` | 0.498 |
| `load_stocki(tickers="NVDA")` | `(920, 32, 8)` | 0.510 |

Windows per session is `78 - window - horizon + 1`, times 187 sessions.

### Train/test split

```python
>>> train = load_stocki(subset="train")
>>> test  = load_stocki(subset="test")
>>> train.data.shape, test.data.shape
((6762, 32, 8), (1840, 32, 8))
>>> train.timestamps.max()
numpy.datetime64('2026-08-10T19:50:00.000000000')
>>> test.timestamps.min()
numpy.datetime64('2026-08-11T16:05:00.000000000')
>>> train.timestamps.max() < test.timestamps.min()
True
>>> train.target.mean(), test.target.mean()
(0.495, 0.512)
```

Train is days 1–16, test is days 17–20. The split is by **day**, so no window
straddles the boundary — leakage is structurally impossible rather than
something you have to remember.

Need a validation set too? Split the training days yourself; the metadata is
right there:

```python
val   = train.data[train.day >= 14]      # days 14-16 held out for validation
inner = train.data[train.day <  14]      # days 1-13 to actually fit on
```

Do **not** use a random split — windows overlap by 31 of their 32 bars, so
random assignment puts near-identical windows on both sides.

> **Know this before reading per-ticker results:** AAPL was only collected from
> day 13, so it contributes 4 training days against 4 test days while every other
> ticker contributes 16 against 4. MSFT is missing day 1. `ds.DESCR` says so too.

### Feeding a model

```python
train = load_stocki(subset="train")
X, y = train.data, train.target            # (6762, 32, 8), (6762,)

# Keras / TensorFlow: channels-last is already correct.
# PyTorch Conv1d wants (N, channels, timesteps):
X_torch = load_stocki(subset="train", channels_first=True).data   # (6762, 8, 32)
```

```python
>>> load_stocki().data.shape                        # Keras
(8602, 32, 8)
>>> load_stocki(channels_first=True).data.shape     # torch Conv1d
(8602, 8, 32)
```

Same numbers, transposed — `keras[0].T == torch[0]`.

Straight into torch:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

train = load_stocki(subset="train", channels_first=True)
loader = DataLoader(
    TensorDataset(
        torch.from_numpy(train.data),                     # float32 (N, 8, 32)
        torch.from_numpy(train.target).float(),           # int8 -> float for BCE
    ),
    batch_size=64,
    shuffle=True,
)
```

`in_channels` is `len(ds.feature_names)` — 8 with the defaults, and it changes
if you pass `channels=[...]`. Read it from the data rather than hardcoding it.

### Exploring

**`load_bars()` — the tidy long table.** One row per 5-minute bar, no windowing,
no labels. This is the one for plotting and sanity checks.

```python
>>> from stocki.datasets import load_bars
>>> bars = load_bars()
>>> bars.shape
(14586, 8)
>>> bars.head(4)
  ticker  day                 timestamp        open        high         low       close   volume
0   AAPL   13 2026-08-05 13:30:00+00:00  309.359985  309.380005  307.649994  308.630005  3202077
1   AAPL   13 2026-08-05 13:35:00+00:00  308.635010  308.635010  305.670013  307.760010  1714140
2   AAPL   13 2026-08-05 13:40:00+00:00  307.774994  308.290009  307.130005  308.160004   733779
3   AAPL   13 2026-08-05 13:45:00+00:00  308.170013  308.195007  307.270111  307.285004   584362
```

dtypes: `ticker` str, `day` int64, `timestamp` datetime64[us, UTC], OHLC
float64, `volume` int64.

**No pandas needed.** `as_frame=False` gives a numpy record array with named
fields:

```python
>>> rec = load_bars(as_frame=False)
>>> rec.shape
(14586,)
>>> rec["close"][:5]
array([308.63  , 307.76  , 308.16  , 307.285 , 307.7701])
>>> rec["volume"][:5]
array([3202077, 1714140,  733779,  584362,  789224], dtype=int64)
>>> rec["close"].mean()
349.6510
```

Filtering: `load_bars(tickers="NVDA", days=[1, 2])`, `tickers=["NVDA", "AAPL"]`.
An unknown ticker raises `ValueError` listing the valid ones.

**`load_panel()` — the numpy cube.** For slicing rather than filtering:

```python
>>> panel = load_panel()
>>> panel.values.shape          # ticker x day x bar x field
(10, 20, 78, 5)
>>> panel.values.dtype
dtype('float64')
>>> panel.tickers
['AAPL', 'AMZN', 'GOOGL', 'JNJ', 'JPM', 'META', 'MSFT', 'NVDA', 'TSLA', 'V']
>>> panel.fields
('open', 'high', 'low', 'close', 'volume')

>>> panel.values[0, 12, :3]     # AAPL, day 13, first 3 bars
array([[    309.36,     309.38,     307.65,     308.63, 3202077.],
       [    308.635,    308.635,    305.67 ,    307.76 , 1714140.],
       [    307.775,    308.29 ,    307.13 ,    308.16 ,  733779.]])
```

**Missing sessions are `NaN`, never zero-filled.** A zero-fill would quietly
average in prices that never existed:

```python
>>> closes = panel.values[..., 3]        # the close field
>>> np.mean(closes[0, :3], axis=1)       # AAPL days 1-3 -- never collected
array([nan, nan, nan])                   # <- NaN tells you, instead of lying
>>> np.nanmean(closes[7, :3], axis=1)    # NVDA days 1-3
array([204.7396, 206.1074, 211.9099])
```

**`load_fundamentals()`** — 187 rows (one per ticker-day), 107 columns, instead
of the same ~105 values repeated across all 78 bars of a session.

**`describe()`** prints the data card without loading any windows.

**Slicing works the same way for training and exploration** — every metadata
array lines up with `ds.data` row for row:

```python
>>> ds = load_stocki()
>>> ds.data[ds.ticker == "NVDA"].shape
(920, 32, 8)
>>> ds.target[ds.ticker == "NVDA"].mean()
0.5098
>>> mask = (ds.ticker == "TSLA") & (ds.day == 20)
>>> ds.data[mask].shape, ds.target[mask].mean()
((46, 32, 8), 0.5000)
```

### One CSV, one call

```python
from stocki.datasets import load_raw

load_raw("AAPL", 13)    # all 118 columns of data/AAPL/day13.csv, from Postgres
```

---

## For the dashboard: the API

Base URL `http://localhost:8000`, everything read-only, OpenAPI at
`/openapi.json` so you can generate a typed client.

| Route | Returns |
|---|---|
| `GET /health` | `{status, database, bar_count}` |
| `GET /api/v1/tickers` | the 10 tickers with names, currency, coverage |
| `GET /api/v1/coverage` | the ticker x day matrix |
| `GET /api/v1/bars/{ticker}/{day}` | one session, mirroring the CSV |
| `GET /api/v1/bars?ticker=&day=&limit=&offset=` | paginated bars for charting |
| `GET /api/v1/fundamentals/{ticker}/{day}` | the ~105-field snapshot |
| `GET /api/v1/news/{ticker}/{day}` | count and latest headline |
| `GET /api/v1/dataset/stats` | shape, class balance, split sizes |
| `GET /api/v1/dataset/windows?format=json\|npz` | the tensors, up to 500 per request |

`/api/v1/dataset/*` runs the same `stocki.datasets` code the model imports, so
the dashboard and the model can never disagree about what the data is.

### Real responses

```
GET /health
{"status": "ok", "database": "ok", "bar_count": 14586}
```

```
GET /api/v1/bars/NVDA/1          -> array of 78 objects, one per bar
[{"ticker": "NVDA", "day": 1, "timestamp": "2026-07-20T13:30:00Z",
  "open": 205.9750061035, "high": 207.3399963379, "low": 205.0800018311,
  "close": 206.75, "volume": 6712839}, ...]
```

```
GET /api/v1/bars?ticker=NVDA&limit=10     -> a page, not a bare array
{"items": [...10 bars...], "total": 1560, "limit": 10, "offset": 0}
```

```
GET /api/v1/dataset/stats
{"n_windows": 8602, "shape": [32, 8],
 "feature_names": ["open","high","low","close","volume",
                   "log_return","hl_range","co_range"],
 "up_rate": 0.4984887235526622,
 "tickers": ["AAPL","AMZN","GOOGL","JNJ","JPM","META","MSFT","NVDA","TSLA","V"],
 "days": [1, 2, ..., 20], "window": 32, "horizon": 1, "subset": "all"}
```

```
GET /api/v1/dataset/windows?limit=2&ticker=NVDA
{"shape": [2, 32, 8], "feature_names": [...],
 "data": [[[0.1611, 1.7274, -0.7198, 1.4419, 5.2291, 0.0413, 4.0261, 1.6734], ...]],
 "target": [0, 0], "ticker": ["NVDA","NVDA"], "day": [1, 1],
 "timestamps": ["2026-07-20T16:05:00Z", "2026-07-20T16:10:00Z"]}
```

The dataset routes take the same parameters as `load_stocki`:
`?ticker=&window=&horizon=&threshold=&subset=&test_days=&limit=&format=`.
So `GET /api/v1/dataset/stats?subset=train` returns `n_windows: 6762`, and
`?subset=test` returns `1840`.

**`format=npz` for real data volumes.** JSON is fine for a preview; the binary
form is far smaller and loads directly into numpy:

```python
import io, numpy as np, requests

raw = requests.get("http://localhost:8000/api/v1/dataset/windows",
                   params={"limit": 500, "format": "npz"}).content
with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
    X = archive["data"]        # (500, 32, 8) float32
    y = archive["target"]      # (500,) int8
```

50 windows come to ~48 KB compressed. Capped at 500 per request — for the whole
dataset, import the package instead of paging the API.

### Errors

Every failure has the same three keys:

```json
{ "error": "validation_error", "detail": "...", "request_id": "9f2c..." }
```

| Status | When | `detail` |
|---|---|---|
| 404 | the session was never collected | `"no session for AAPL on day 1"` |
| 422 | unknown ticker | `"unknown ticker 'NOPE'; available tickers are ['AAPL', ...]"` |
| 422 | parameter out of bounds | `[{"field": "path.day", "problem": "Input should be less than or equal to 20"}]` |
| 429 | rate limit hit | `"rate limit of 120 requests/minute exceeded; try again shortly"` |
| 500 | something broke | a generic message — the traceback is logged server-side |

Note the difference between **404 and 422** on a ticker/day pair: 422 means the
ticker does not exist at all, 404 means it exists but that session was never
collected (`/bars/AAPL/1` — AAPL starts at day 13). Both are normal; render them
differently.

Every response carries an `X-Request-ID` header matching `request_id`. On a 500,
send us that id rather than a screenshot.

**Bounds:** `day` 1–20, `window` 8–78, `horizon` 1–16, `limit` ≤ 1000 (bars) or
≤ 500 (windows), `subset` one of `all|train|test`.

**Rate limits:** 120 requests/minute for most routes, 20/minute for
`/dataset/*` (they are the expensive ones). Both configurable via
`STOCKI_RATE_LIMIT` and `STOCKI_DATASET_RATE_LIMIT`.

**CORS** is locked to the origins in `STOCKI_CORS_ORIGINS` (default
`http://localhost:3000,http://localhost:5173`), never `*`. If your dev server
runs on another port, add it there or your fetches will be blocked.

---

## Live data

`stocki fetch` pulls sessions from Alpha Vantage and writes them as
`data/<TICKER>/day<N>.csv` — the same 118 columns, the same 78 bars, the same
UTC stamps, the same shared day numbering.

```bash
stocki fetch --days 1 && stocki ingest
```

That is the whole integration. Nothing in `model/`, `model_training/` or
`frontend/` changes, because from where they stand nothing did: the model still
trains, tunes and runs on `/api/v1/bars`, which still reads `bars_raw`, which is
still a copy of the files on disk.

### Setup

Put a key in `backend/.env` (free ones: <https://www.alphavantage.co/support/#api-key>):

```bash
ALPHA_ADVANTAGE_KEY=your-key-here
```

`ALPHAVANTAGE_API_KEY`, `ALPHA_VANTAGE_API_KEY` and `STOCKI_ALPHAVANTAGE_KEY`
are read too, in that order, so a teammate's spelling still works. Without a
key nothing else changes — `fetch` is the only command that reads it.

### Usage

```bash
stocki fetch                              # every ticker in data/, most recent weekday
stocki fetch --tickers NVDA,AAPL --days 5
stocki fetch --date 2026-08-17
stocki fetch --dry-run                    # fetch and validate, write nothing
stocki fetch --days 1 --ingest            # and load it into Postgres afterwards
```

| Flag | Effect |
|---|---|
| `--tickers` | comma-separated; default is every ticker folder in `data/` |
| `--days N` | the last N weekdays (default 1) |
| `--date` | one `YYYY-MM-DD` session instead |
| `--allow-partial` | write a session with fewer than 78 bars (market still open) |
| `--overwrite` | replace an existing `day<N>.csv` |
| `--refresh` | ignore the fundamentals cache |
| `--no-fundamentals` / `--no-news` | skip those calls; the columns are left `NULL` |
| `--quote` | spend a call per ticker on the live price for `si_current_price` |
| `--budget N` | stop after N requests (default 25) |
| `--dry-run`, `--ingest` | validate only / run `ingest` on success |

### The request budget

The free plan allows **25 requests a day** at about one a second, which is the
constraint the design is shaped around:

| Call | Cost | Cached |
|---|---|---|
| `TIME_SERIES_INTRADAY` | 1 per ticker per calendar month covered | no |
| `NEWS_SENTIMENT` | **1 for the whole universe** — articles carry the tickers they mention | no |
| `OVERVIEW`, `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW` | 4 per ticker | yes, `STOCKI_LIVE_CACHE_HOURS` (24h) |

So ten tickers cost ~51 requests the first time and **~11 every day after**,
because company reports change quarterly and are served from
`.cache/alphavantage/`. A run that would exceed the budget stops with a message
instead of half-writing a dataset.

### Where the columns come from

Most are copied. Some are derived, and those derivations were worked out from
the committed files and are checked against them in `tests/test_live_fields.py`
— 43 columns reproduce `data/AAPL/day13.csv` to the digit:

```
bs_total_debt          = currentLongTermDebt + longTermDebt   (98657000000)
bs_invested_capital    = equity + total debt                  (172390000000)
cf_free                = operatingCashflow - capex            (98767000000)
cf_begin_cash          = closing cash - net change            (29943000000)
si_total_cash          = MRQ cash + short-term investments    (62399000000)
```

Two scale conversions matter: `si_dividend_yield` is a percent in the files and
a fraction upstream; `si_inst_pct` / `si_insider_pct` are the other way round.
And `si_*` reads the **most recent quarter** while `inc_*` / `bs_*` / `cf_*`
read the **latest annual report** — that split is what makes `si_total_cash` and
`bs_cash` legitimately different numbers.

Seven columns have no upstream field and are written `NULL` rather than zero,
because a zero would be a claim about the company: `si_employees`,
`si_short_ratio`, `si_target_high`, `si_target_low`, `bs_gross_ppe`,
`bs_accum_depreciation`, `cf_deferred_tax`. `thsname_cn` has no upstream field
either, so an existing ticker keeps the name already recorded on disk and a new
one gets an empty cell.

### Known limitation: intraday bars need a paid plan

**`TIME_SERIES_INTRADAY` is a premium endpoint.** On a free key every form of
the request comes back with:

```
This is a premium endpoint. You may subscribe to any of the premium plans
at https://www.alphavantage.co/premium/ to instantly unlock all premium endpoints
```

Everything else `fetch` needs — the four fundamentals endpoints, news and quotes
— works on the free tier and is verified against real responses. So on a free
key `fetch` fills 105 of the 118 columns and reports the bars as unavailable;
with a paid key it writes complete sessions. The client raises `StockiPlanError`
naming the endpoint rather than failing obscurely.

Daily bars (`TIME_SERIES_DAILY`) *are* free, but a daily bar is not a 5-minute
bar. Synthesising 78 intraday bars from one daily OHLC would put prices that
never traded into the training set, so `fetch` does not do it.

---

## Command line

```bash
stocki fetch      # pull live sessions from Alpha Vantage into data/
stocki ingest     # read data/ into Postgres (idempotent, safe to re-run)
stocki verify     # prove bars_raw still matches every CSV
stocki stats      # print the data card
stocki serve      # run the API without Docker
```

Exit codes: `0` fine, `1` the data disagrees with the files, `2` could not run
at all (database down, nothing ingested).

---

## The database

### Connecting

```bash
psql postgresql://stocki:stocki@localhost:5434/stocki
```

Inside the Docker network the host is `postgres:5432` instead. From Python,
`stocki.datasets` handles this for you — reach for raw SQL only when you want
something the loaders do not cover.

### One table, four views

`bars_raw` mirrors the CSVs column for column, so this **is**
`data/AAPL/day13.csv`:

```sql
SELECT * FROM bars_raw WHERE ticker = 'AAPL' AND day = 13 ORDER BY timestamp;
```

121 columns: the 118 from the file, plus `day` (which exists only in the
*filename* and would otherwise be lost), `source_file`, and `ingested_at`. Two
renames for SQL legality — `Dividends` -> `dividends`, `Stock Splits` ->
`stock_splits` — so **no column ever needs double-quoting**. Nothing else
changes, and `stocki verify` proves it file by file.

Prefer the views for everyday work. They are named queries, not copies:

| View | Rows | Columns | Use it for |
|---|---|---|---|
| `v_bars` | 14,586 | ticker, day, timestamp, OHLCV | charts, time series, anything price-shaped |
| `v_fundamentals` | 187 | ticker, day + ~105 static fields | per-session fundamentals, deduplicated |
| `v_news` | 187 | ticker, day, news_count, headline | news per session |
| `v_coverage` | 187 | ticker, day, bar_count, first_bar, last_bar | which sessions exist |

`v_fundamentals` matters: those ~105 columns are **identical across all 78 bars
of a session**, so querying them from `bars_raw` gives you 78 duplicate rows per
ticker-day. The view collapses them to one.

### Queries you will actually want

**Which sessions exist, and where the gaps are:**

```sql
SELECT ticker, count(*) AS sessions, min(day) AS first_day, max(day) AS last_day
FROM v_coverage GROUP BY ticker ORDER BY ticker;
```
```
 ticker | sessions | first_day | last_day
--------+----------+-----------+----------
 AAPL   |        8 |        13 |       20     <- starts late
 AMZN   |       20 |         1 |       20
 ...
 MSFT   |       19 |         2 |       20     <- missing day 1
```

**One session for a chart:**

```sql
SELECT timestamp, open, high, low, close, volume
FROM v_bars WHERE ticker = 'NVDA' AND day = 1 ORDER BY timestamp;
```
```
       timestamp        |    open     |    high     |     low     |  close  | volume
------------------------+-------------+-------------+-------------+---------+---------
 2026-07-20 13:30:00+00 | 205.9750061 | 207.3399963 | 205.0800018 |  206.75 | 6712839
```

**Daily OHLC rolled up from the 5-minute bars:**

```sql
SELECT ticker, day,
       (array_agg(open  ORDER BY timestamp))[1]              AS day_open,
       max(high)                                             AS day_high,
       min(low)                                              AS day_low,
       (array_agg(close ORDER BY timestamp DESC))[1]         AS day_close,
       sum(volume)                                           AS day_volume
FROM v_bars GROUP BY ticker, day ORDER BY ticker, day;
```

**Fundamentals without the duplication:**

```sql
SELECT ticker, day, si_market_cap, si_pe_trailing, si_beta
FROM v_fundamentals WHERE ticker = 'NVDA' ORDER BY day;
```

**Bar-to-bar returns, in SQL:**

```sql
SELECT ticker, day, timestamp, close,
       close / lag(close) OVER (PARTITION BY ticker, day ORDER BY timestamp) - 1 AS ret
FROM v_bars WHERE ticker = 'AAPL' AND day = 13 ORDER BY timestamp;
```

Note the `PARTITION BY ticker, day` — without it, `lag` reaches across a session
boundary and computes an overnight gap as if it were a 5-minute move. The same
reasoning is why no window in `load_stocki` spans two days.

### Rules of the road

- **The table is read-mostly.** `stocki ingest` owns writes and is idempotent
  (`ON CONFLICT (ticker, timestamp) DO UPDATE`), so re-running it is safe. Do not
  hand-edit rows — `stocki verify` will fail and you will not know which file is
  the truth.
- **The API connects as `stocki_ro`**, which holds `SELECT` and nothing else. Use
  those credentials for anything exposed.
- **Fix data in the CSV, then re-ingest.** The files are the source of truth; the
  table is a queryable copy of them.
- **`timestamp` is `timestamptz` in UTC.** The market session runs 13:30–19:55
  UTC. If your client renders local time, that is your client, not the data.
- **Prices are `double precision`**, which round-trips the CSV floats exactly
  (`309.3599853516` comes back bit-identical). Do not cast to `numeric` and
  expect the same bytes.

### Direct SQL from Python

If you want a query the loaders do not expose:

```python
import pandas as pd
from stocki.db.session import connect

with connect() as conn:
    df = pd.read_sql(
        "SELECT ticker, day, timestamp, close FROM v_bars WHERE ticker = %s",
        conn, params=("NVDA",),
    )
```

`connect()` reads the same settings as everything else and raises a readable
error if Postgres is down. Always pass parameters — never f-string user input
into SQL.

The API connects as `stocki_ro`, which holds `SELECT` and nothing else.

---

## Running the tests

```bash
pip install -r requirements.txt
docker compose up -d postgres
pytest
```

Tests marked `db` need Postgres and skip cleanly without it. The suite includes a
golden test that ingests all 187 real files and checks the counts, the coverage
gaps, and the round-trip against disk.

---

## Layout

```
backend/stocki/
├── config.py       environment-driven settings
├── errors.py       errors that say what to do next
├── cli.py          the stocki command
├── db/             schema.sql, connections, the read-only role
├── live/           Alpha Vantage -> data/<TICKER>/day<N>.csv
├── ingest/         columns, reader, validation, loading
├── datasets/       labels, windowing, loaders   <- the model imports this
└── api/            FastAPI routes, validation, hardening
```

`datasets/` never imports `api/`, so training and notebooks do not need FastAPI
installed. `live/` is imported by nothing except `cli.py`, and only inside the
`fetch` handler — a clone with no API key behaves exactly as it did before.

```
live/
├── client.py     the HTTP client: budget, throttle, cache, error envelopes
├── fields.py     provider JSON -> the 118 columns (pure, no I/O)
├── sessions.py   the market clock and the shared day numbering
└── fetch.py      orchestration: fetch, validate, write
```
