# Training and running on live data

**Scope:** how to pull fresh market data and train, tune, or run the model on
it. Picks up where [`model-training-and-inference.md`](model-training-and-inference.md)
leaves off.

**Audience:** Cason, or anyone retraining the model on data newer than the 20
days committed to the repo.

---

## What changed for you

Nothing in `model_training/` or `model/`. Not one line.

Live data enters at the **front** of the pipeline as `data/<TICKER>/day<N>.csv`
files — the same 118 columns, the same 78 bars, the same UTC stamps, the same
day numbering as the files already committed. Everything downstream reads them
the way it always did:

```
stocki fetch            <- new: writes data/<TICKER>/day<N>.csv from Alpha Vantage
        |
stocki ingest           <- unchanged
        v
bars_raw -> /api/v1/bars
        |
python main.py data     <- unchanged; the cache notices the new bars by itself
        v
python main.py train
```

So there is one new command upstream of you, and your existing workflow is
otherwise identical.

---

## One-time setup

You need an Alpha Vantage key in `backend/.env`:

```bash
ALPHA_ADVANTAGE_KEY=your-key-here
```

Free keys: <https://www.alphavantage.co/support/#api-key>. `.env` is gitignored —
never commit it. Nothing else in the stack reads this key, so if you only ever
train on the committed data you can skip it entirely.

---

## The loop

```bash
# 1. pull the most recent session for every ticker in data/
stocki fetch --days 1

# 2. load it into Postgres (idempotent, safe to re-run)
stocki ingest

# 3. confirm the API sees it
curl localhost:8000/health

# 4. your normal workflow, from here on unchanged
cd model_training
python main.py data
python main.py train --fresh
python main.py evaluate --subset test
python main.py export
```

Step 3 is worth doing every time. `bar_count` is what tells you the new session
actually landed, and it is also what invalidates your bar cache — see below.

### You do not need `--refresh`

`api_client.fetch_bars` checks the cached `bars.npz` against `/health`'s
`bar_count` on every run. New sessions change that number, so the cache
invalidates itself:

```
fetching bars from http://localhost:8000 (cache holds 14586, API has 15366)
  fetched 15366/15366 bars
cached 15366 bars to model_training/.cache/bars.npz
```

`python main.py data --refresh` still works if you want to force it.

---

## The numbers move — here is the arithmetic

`model-training-and-inference.md` tells you `bar_count` must be **14,586** and
to stop if `python main.py data` prints anything else. That is the right check
for the committed dataset and the wrong one the moment you fetch. Use the
arithmetic instead:

| quantity | formula | 187 sessions | +1 day × 10 tickers |
| --- | --- | --- | --- |
| sessions | `(ticker, day)` pairs that exist | 187 | 197 |
| `bar_count` | `sessions × 78` | 14,586 | 15,366 |
| windows | `sessions × 46` | 8,602 | 9,062 |

46 is `78 - window - horizon + 1` with the defaults (`window=32`, `horizon=1`).
Change either and it changes.

`stocki stats` prints the session count and the coverage gaps directly, which is
the quickest way to check the left column.

---

## Two traps

### 1. The test set moves, so metrics are not comparable across refreshes

Splits are computed from the **last** day present, not from fixed day numbers:

```python
test_cutoff = day.max() - TEST_DAYS      # days above this are test
val_cutoff  = test_cutoff - VAL_DAYS
```

Fetch one day and everything shifts by one:

| | days 1–20 (committed) | days 1–21 (after one fetch) |
| --- | --- | --- |
| train | 1–13 · 5,382 windows | 1–14 · 5,842 windows |
| val | 14–16 · 1,380 | 15–17 · 1,380 |
| test | **17–20** · 1,840 | **18–21** · 1,840 |

A test accuracy of 0.53 on days 17–20 and 0.51 on days 18–21 are numbers about
**different held-out data**. Do not put them in the same row of a results table.

Record the day range with every result — `loader.summary()` prints it, and
`main.py data` shows it in the `days` column.

`--test-days N` will not pin an old test set for you. It always means "the last
N days", so on days 1–24, `--test-days 8` holds out days 17–**24**, not 17–20.
There is no flag for a window in the middle. To score a checkpoint on a fixed
day range, mask on the day metadata yourself — it lines up with the windows row
for row:

```python
import numpy as np
from dataloader import StockDataLoader

loader = StockDataLoader()
days = loader.windows.day
fixed = np.nonzero((days >= 17) & (days <= 20))[0]   # always days 17-20

x, y = loader.get_batch(fixed)                        # however much data arrives later
```

### 2. Use `--fresh` after a data refresh

`python main.py train` resumes from `checkpoints/latest.pt` by default, and the
checkpoint carries `best_val_loss` with it:

```python
best_val_loss = checkpoint.get("best_val_loss", float("inf"))
is_best = val_metrics["loss"] < best_val_loss
```

That threshold was measured on the **old** validation days. After a refresh the
validation set is different data, so if the new days are even slightly harder,
`is_best` never fires, `best.pt` is never rewritten, and `python main.py export`
quietly ships the model you trained last week.

```bash
python main.py train --fresh          # after any fetch
```

Resuming across a refresh is not leakage — the old model never saw the new test
days, they did not exist yet. It is a broken best-checkpoint tracker and a
training history that no longer matches the split underneath it. Both are enough
to throw the run away.

---

## Running the model on today's market

A live session is only complete after the close. Before then, `fetch` refuses to
write a short file unless you ask:

```bash
stocki fetch --tickers NVDA --allow-partial
```

That writes however many bars have printed so far. It will **not** auto-ingest —
`stocki ingest` validates every file against one exact bar count, so a directory
holding both 78-bar history and a 40-bar live session cannot satisfy it either
way. Point ingest at a directory of its own:

```bash
stocki fetch --tickers NVDA --allow-partial --data-dir /tmp/live
stocki ingest --data-dir /tmp/live --expected-bars 40
```

### Predicting from the trailing 32 bars

For inference you do not need a label, so you do not need `build_windows` — it
returns nothing for a 32-bar input, because `78 - 32 - 1 + 1` arithmetic leaves
no bar ahead to label against.

Derive the channels on the **whole session** and slice afterwards. That order
matters: `log_return` reads the previous bar, so deriving on a 32-bar slice
would zero the first return instead of computing it, and the input would not
match what the model was trained on.

```python
import numpy as np
import requests

import api_client
import config
from dataloader import derive_channels, normalize_windows

bars = api_client.fetch_bars(refresh=True)
nvda = bars[bars["ticker"] == "NVDA"]
session = nvda[nvda["day"] == nvda["day"].max()]        # the newest session

# Derive on the full session, exactly as build_windows does, then take the tail.
channels = derive_channels(session)
matrix = np.column_stack([channels[name] for name in config.FEATURE_CHANNELS])
window = matrix[-config.SEQUENCE_LENGTH:][None, ...]     # (1, 32, 8)
x = normalize_windows(window)[0]                         # window-z, (32, 8)

response = requests.post(
    "http://localhost:8001/predict", json={"sequence": x.tolist()}
)
print(response.json())        # {"prediction": [0.5231...]}
```

`/predict` wants `(timesteps, features)` and transposes to channels-first
itself. The number it returns is P(the next close is higher).

Needs at least `SEQUENCE_LENGTH` bars in the session — 32 by default, so about
2h40m after the open.

---

## Known limitation: 5-minute bars need a paid plan

`TIME_SERIES_INTRADAY` is a **premium** Alpha Vantage endpoint. On a free key
every form of the request comes back with:

```
This is a premium endpoint. You may subscribe to any of the premium plans
at https://www.alphavantage.co/premium/ to instantly unlock all premium endpoints
```

What this means for you:

| | free key | paid key |
| --- | --- | --- |
| fundamentals, news, quotes | work | work |
| 5-minute bars | **unavailable** | work |
| what `fetch` writes | 105 of 118 columns, bars reported unavailable | complete sessions |

`fetch` raises `StockiPlanError` naming the endpoint rather than failing
obscurely, and no code changes when the key is upgraded.

Daily bars (`TIME_SERIES_DAILY`) *are* free, but a daily bar is not a 5-minute
bar, and `fetch` will not manufacture 78 intraday bars from one daily OHLC. That
would put prices that never traded into your training set, and every metric
downstream would be measuring fiction.

**So until the key is upgraded, train on the committed 20 days.** The plumbing
is in place and tested; it is one API plan away from producing new sessions.

---

## Request budget

The free plan allows **25 requests a day**, which is why the fetch is shaped the
way it is:

| call | cost | cached |
| --- | --- | --- |
| `TIME_SERIES_INTRADAY` | 1 per ticker per calendar month covered | no |
| `NEWS_SENTIMENT` | **1 for all tickers at once** | no |
| `OVERVIEW`, `INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW` | 4 per ticker | 24h |

Ten tickers cost ~51 requests the first time and **~11 every day after**, since
company reports change quarterly and are served from `.cache/alphavantage/`.

A run that would exceed the budget stops with a message instead of half-writing
a dataset. `stocki fetch --dry-run` fetches and validates without writing;
`--budget N` changes the ceiling.

---

## Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `no Alpha Vantage API key found` | no key in `backend/.env` | add `ALPHA_ADVANTAGE_KEY=` |
| `does not serve TIME_SERIES_INTRADAY on this API key's plan` | free key | see the limitation above |
| `budget of 25 Alpha Vantage calls is spent` | daily quota | fetch fewer tickers, or wait for the reset |
| `skipped: 40 of 78 bars` | market still open | wait for the close, or `--allow-partial` |
| `skipped: no bars returned` | market holiday, or ticker not traded | expected; the session is simply absent |
| `skipped: day3.csv already exists` | that date is already collected | `--overwrite` if you mean to replace it |
| `falls inside the range already collected` | fetching a date older than the newest on disk | fetch newer dates; renumbering `data/` is a deliberate act |
| `data` prints the old bar count | backend not re-ingested | `stocki ingest`, then check `/health` |
| `best.pt` never updates after a refresh | resumed `best_val_loss` | `python main.py train --fresh` |

---

## Reference

| command | does |
| --- | --- |
| `stocki fetch` | every ticker in `data/`, most recent weekday |
| `stocki fetch --tickers NVDA,AAPL --days 5` | named tickers, last 5 weekdays |
| `stocki fetch --date 2026-08-17` | one specific session |
| `stocki fetch --dry-run` | fetch and validate, write nothing |
| `stocki fetch --days 1 --ingest` | fetch, then ingest if everything validated |
| `stocki fetch --no-fundamentals --no-news` | bars only; those columns become `NULL` |
| `stocki stats` | session count, coverage, gaps |
| `stocki verify` | prove `bars_raw` still matches every CSV |

Full flag list and the column-by-column mapping:
[`backend/README.md#live-data`](../backend/README.md).
