# model_training

Training framework for the stocki 1D-CNN. Reads its data from the backend API
over HTTP — no database driver, no CSV parsing, no repo-internal imports.

| file | what it holds |
| --- | --- |
| `config.py` | all tunables: feature channels, window, horizon, split sizes, API URL |
| `api_client.py` | the only file that touches the network — paged bar fetch + on-disk cache |
| `dataloader.py` | windowing, labelling, normalisation, day-based splits, batching |
| `model.py` | `StockCNN1D` — the base model |
| `train.py` | training loop, metrics, checkpointing, resume, `evaluate()` |
| `export_onnx.py` | checkpoint → `.onnx` for the model service |
| `main.py` | CLI: `data` / `train` / `evaluate` / `export` |

## Running it

The backend has to be up, because that is where the data comes from:

```bash
docker compose up -d          # from the repo root -- API on :8000
cd model_training
pip install -r requirements.txt

python main.py data                      # fetch bars, print the data card, verify the transform
python main.py train                     # resumes from checkpoints/latest.pt if present
python main.py train --fresh --epochs 20 # ignore checkpoints, start from random weights
python main.py evaluate --subset test    # score checkpoints/best.pt on the held-out days
python main.py export                    # checkpoints/best.pt -> artifacts/model.onnx

python dataloader.py                     # data card + verification against the API
python api_client.py                      # coverage per ticker, straight from the API
python model.py                          # architecture + a forward-pass shape check
```

If the API is on another host or port, set `STOCKI_API_URL` (default
`http://localhost:8000`) or pass `--api-url`.

`python main.py data` should print this — different numbers mean the backend
ingest is incomplete, so run `stocki verify` over there:

```
8602 windows of shape (8, 32) [8.4 MB]
channels: open, high, low, close, volume, log_return, hl_range, co_range
tickers:  AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, V

subset   windows   up rate  days
train       5382    0.4892  1-13
val         1380    0.5159  14-16
test        1840    0.5125  17-20

verified 500 windows against /api/v1/dataset/windows: labels identical, max |local - api| = 0.000e+00
```

## Where the data comes from

```
GET /api/v1/bars?limit=1000&offset=N   ->  14,586 bars (15 pages)
        |  cached at model_training/.cache/bars.npz
        v
  derive channels -> window per session -> label -> window-z normalise
        v
  8,602 windows of (8, 32), float32, channels-first
```

`dataloader.py` deliberately mirrors the transform in
`backend/stocki/datasets/{windows,labels}.py` — same formulas, same
`(ticker, day)` grouping, same ordering. **Why repeat it instead of asking the
API for finished windows:** `/api/v1/dataset/windows` takes no `offset` and is
capped at 500, so it can only ever return the first 500 of 8,602. It is used as
a *check* instead: `verify_against_api()` compares the local tensors against the
backend's own, index for index, and raises if they disagree. Currently the match
is exact (`max |local - api| = 0.0`, every label identical), so the two cannot
silently drift apart.

The bar cache is validated against `/health`'s `bar_count` on every run — one
cheap request — so a re-ingest on the backend invalidates it automatically.
Delete `.cache/` or pass `--refresh` to force a refetch. If the API is
unreachable but a cache exists, the run continues on the cache with a warning.

### Leakage controls

Three, inherited from the backend design:

1. **No window crosses a session boundary.** Day 13 may be a week after day 12,
   so a window spanning two sessions would describe a move that never happened.
   Windowing runs per `(ticker, day)`.
2. **Splits are by trading day, never random.** Windows overlap by 31 of their 32
   bars, so a random split scores the model on near-copies of its training data.
   There is no random-split option in the loader — this was the one real bug in
   the old `split_indices`.
3. **Normalisation is per window.** Each channel is centred at 0 with unit spread
   using only the 32 bars inside that window; dataset-wide statistics would fold
   the test set into training.

### Splits

| subset | days | windows | set by |
| --- | --- | --- | --- |
| train | 1–13 | 5,382 | whatever is left |
| val | 14–16 | 1,380 | `config.VAL_DAYS = 3` |
| test | 17–20 | 1,840 | `config.TEST_DAYS = 4` |

`test` uses the same cutoff as the backend's own `subset="test"`, so it is the
same 1,840 windows the rest of the team evaluates on and the numbers stay
comparable. `val` is carved out of what the backend calls `train`.

`train.py` reads `val` every epoch and never touches `test`. Scoring the test
set is a separate command (`main.py evaluate`), run once at the end — tuning
against it turns it into a second validation set.

`StockDataLoader.splits()` asserts
`train.timestamps.max() < val.min() < test.min()` before returning, so a bad
`TEST_DAYS`/`VAL_DAYS` fails loudly instead of quietly leaking.

> **Caveat for per-ticker results:** AAPL was only collected from day 13, so it
> contributes 1 training day against 3 val and 4 test days, while every other
> ticker contributes ~13. MSFT is missing day 1.

## Tensor layout

```
input  (batch, NUM_INPUT_FEATURES, SEQUENCE_LENGTH)   channels-first, as Conv1d wants
output (batch, NUM_OUTPUTS)                            sigmoid -> P(up)
```

`NUM_INPUT_FEATURES` is derived from `config.FEATURE_CHANNELS` (8 channels), so
adding or dropping a channel is a one-line edit and the model, dataloader and
ONNX exporter all follow. `SEQUENCE_LENGTH` is 32, the backend's documented
default; windows-per-session is `78 - SEQUENCE_LENGTH - HORIZON + 1`, so raising
it costs examples.

The network pools over time, so a trained model accepts sequence lengths other
than the one it trained on; `sequence_length` is exported as a dynamic ONNX axis.

## Metrics

`train.py` reports loss and accuracy per epoch plus precision / recall / F1, and
prints the **majority-class baseline** next to them. On this dataset the baseline
sits at ~0.51 accuracy, so anything under about 0.52 is noise, not signal — the
task is close to a coin flip by construction.

## Checkpoints

`checkpoints_old/epoch_XXX.pt` at the end of every epoch, mirrored to
`checkpoints_old/latest.pt` (the resume point), plus `checkpoints_old/best.pt` whenever
validation loss improves. `best.pt` is what `export` uses by default.

Each file carries the weights, the optimizer state, the epoch number, the
architecture config, the data contract (`feature_channels`, `sequence_length`,
`horizon`, `normalize`) and that epoch's metrics — so `export_onnx.py` can
rebuild the right architecture from the checkpoint alone, and you can tell what
a stale checkpoint was trained on.
