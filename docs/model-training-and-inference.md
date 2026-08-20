# Training and inference guide

**Scope:** everything from "the backend is running" to "the prediction API is
answering" — the training scripts in `model_training/`, the ONNX export, and
loading those weights into ONNX Runtime.

**Audience:** anyone on the team who needs to train, retrain, or serve the model.

---

## The pipeline

```
docker compose up -d                     backend API on :8000
        |
        |  GET /api/v1/bars  (paginated, 15 requests)
        v
model_training/.cache/bars.npz           14,586 bars, cached
        |
        |  derive channels -> window per session -> label -> window-z normalise
        v
8,602 windows of (8, 32) float32         5,382 train / 1,380 val / 1,840 test
        |
        |  python main.py train
        v
model_training/checkpoints/best.pt       PyTorch weights + optimizer + metrics
        |
        |  python main.py export
        v
model_training/artifacts/model.onnx      ~641 KB, opset 17
        |
        |  ONNX Runtime
        v
model/  ->  POST /predict                P(up) in [0, 1]
```

Each arrow is one command. Nothing is implicit and nothing is manual.

---

## Prerequisites

```bash
# 1. the backend, because that is where the data comes from
docker compose up -d                 # from the repo root
curl localhost:8000/health           # {"status":"ok","database":"ok","bar_count":14586}

# 2. the training environment
cd model_training
pip install -r requirements.txt       # torch, numpy, onnx, onnxruntime
```

`bar_count` must be **14,586**. A different number means the backend ingest is
incomplete — run `stocki verify` on that side before training on it.

The training scripts need no database driver and no credentials. They read the
public API over `urllib` from the standard library, so the only thing that has to
be reachable is `http://localhost:8000`. Override that with `STOCKI_API_URL` if
the backend is elsewhere.

---

# Part 1 — the data

## Check it before you train

```bash
python main.py data
```

```
using 14586 cached bars from model_training/.cache/bars.npz

8602 windows of shape (8, 32) [8.4 MB]
channels: open, high, low, close, volume, log_return, hl_range, co_range
tickers:  AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, V

subset   windows   up rate  days
train       5382    0.4892  1-13
val         1380    0.5159  14-16
test        1840    0.5125  17-20

verified 500 windows against /api/v1/dataset/windows: labels identical, max |local - api| = 0.000e+00
```

Run this first, every time. If the numbers differ from the above, stop and find
out why before spending epochs on it.

| flag | effect |
| --- | --- |
| `--refresh` | ignore `.cache/bars.npz` and refetch all 14,586 bars |
| `--api-url URL` | read from another backend instance |
| `--no-verify` | skip the cross-check against the API's own windows |

## Where the numbers come from

One **window** is 32 consecutive 5-minute bars from a single trading session, and
its **label** is whether the close one bar later is higher. A session is 78 bars,
so each session yields `78 - 32 - 1 + 1 = 46` windows, and 187 sessions give
8,602.

The 8 channels per bar:

| channel | source |
| --- | --- |
| `open`, `high`, `low`, `close`, `volume` | straight from the bar |
| `log_return` | `log(close / previous close)` |
| `hl_range` | `(high - low) / close` |
| `co_range` | `(close - open) / open` |

The last three read the current bar and the one immediately before it, never a
later one.

## Why the windowing happens client-side

The backend has a route that serves finished windows —
`GET /api/v1/dataset/windows` — but it **accepts no `offset` and is capped at
500**, so it can only ever return the first 500 of 8,602. It cannot feed
training.

So `model_training/dataloader.py` mirrors the transform in
`backend/stocki/datasets/{windows,labels}.py`: same formulas, same
`(ticker, day)` grouping, same ordering. Two copies of a transform is a real
risk, so it is checked rather than trusted — `verify_against_api()` pulls 500
windows from the backend and compares index for index:

```
verified 500 windows against /api/v1/dataset/windows: labels identical, max |local - api| = 0.000e+00
```

Bit-exact. This runs on every `main.py data` and raises `AssertionError` if the
two ever drift apart.

## The cache

Bars land in `model_training/.cache/bars.npz` (gitignored). On every run the
cache is validated against `/health`'s `bar_count` — one cheap request — so a
re-ingest on the backend invalidates it automatically.

If the API is unreachable but a cache exists, the run continues on the cache with
a warning rather than failing. Delete `.cache/` or pass `--refresh` to force a
refetch.

## Splits, and the one rule that matters

| subset | days | windows | up rate | set by |
| --- | --- | --- | --- | --- |
| train | 1–13 | 5,382 | 0.4892 | whatever is left |
| val | 14–16 | 1,380 | 0.5159 | `config.VAL_DAYS = 3` |
| test | 17–20 | 1,840 | 0.5125 | `config.TEST_DAYS = 4` |

**Never split these randomly.** Consecutive windows overlap by 31 of their 32
bars, so a random split puts near-identical windows on both sides of the boundary
and the validation score becomes fiction. `StockDataLoader` has no random-split
option for exactly this reason.

Three layered leakage controls, inherited from the backend design:

1. **No window crosses a session boundary.** Day 13 may be a week after day 12,
   so a window spanning two sessions would describe a move that never happened.
   Windowing runs per `(ticker, day)`.
2. **Splits are by trading day.** No window straddles the boundary because none
   straddles a day. `splits()` asserts
   `train.timestamps.max() < val.min() < test.min()` before returning, so a bad
   `TEST_DAYS`/`VAL_DAYS` fails loudly instead of quietly leaking.
3. **Normalisation is per window.** Each channel is centred at 0 with unit spread
   using only the 32 bars inside that window. Dataset-wide statistics would fold
   the test set into training.

`test` uses the same cutoff as the backend's own `subset="test"`, so it is the
same 1,840 windows the rest of the team evaluates on and the numbers stay
comparable.

> **Caveat for per-ticker results:** AAPL was only collected from day 13, so it
> contributes 1 training day against 3 val and 4 test days, while every other
> ticker contributes ~13. MSFT is missing day 1.

## Using the loader directly

For a notebook or a one-off experiment:

```python
from dataloader import StockDataLoader

loader = StockDataLoader()          # fetches (or reuses the cache) on construction
split  = loader.splits()            # .train / .val / .test index arrays

loader.windows.X.shape              # (8602, 8, 32) float32, channels-first
loader.windows.y.shape              # (8602,) int8, 1 = UP
loader.windows.ticker               # (8602,) str
loader.windows.day                  # (8602,) int16
loader.windows.timestamps           # (8602,) datetime64[ns] -- last bar *inside* each window

x, y = loader.get_batch(split.train[:64])     # torch (64, 8, 32), (64, 1)

for x, y in loader.iter_batches(indices=split.train):
    ...

loader.up_rate(split.test)          # 0.5125 -- the baseline to beat
print(loader.summary(split))        # the data card above
```

Every metadata array lines up with `X` row for row, so slicing works:

```python
mask = (loader.windows.ticker == "NVDA") & (loader.windows.day == 20)
loader.windows.X[mask].shape        # (46, 8, 32)
```

---

# Part 2 — training

## The commands

```bash
python main.py train                      # resume from checkpoints/latest.pt if it exists
python main.py train --fresh              # ignore checkpoints, random weights
python main.py train --fresh --epochs 20
python main.py train --lr 3e-4 --batch-size 128
python main.py train --device cpu         # default: cuda if available
python main.py train --refresh-data       # refetch bars first
```

`train` is **resumable by default**. It restores the weights, the optimizer
state, the epoch counter and the best-so-far validation loss, so
`--epochs 20` after a 30-epoch run trains epochs 31–50. Use `--fresh` when you
have changed the architecture or the feature set — a checkpoint from a different
`FEATURE_CHANNELS` will fail to load, which is the intended behaviour.

## Reading the output

```
8602 windows of shape (8, 32) [8.4 MB]
...
validation baseline (always predict UP): accuracy 0.5159, f1 0.6807
created a new model with random weights (no checkpoint at checkpoints/latest.pt)
model: 159,425 trainable parameters on cuda
epoch   1 | train loss 0.70650 acc 0.4931 | val loss 0.69300 acc 0.5051 f1 0.5873 | epoch_001.pt  <- best
epoch   2 | train loss 0.69191 acc 0.5206 | val loss 0.71814 acc 0.4841 f1 0.0000 | epoch_002.pt
```

(Two epochs of a smoke run — the format, not a result.)

**Read the baseline line first.** Always predicting UP scores 0.5159 accuracy on
the validation days. Next-bar direction on 5-minute bars is close to a coin flip
by construction, so:

- **under ~0.52** is noise, not signal, whatever the loss curve looks like;
- **`f1 0.0000`** means the model collapsed to predicting DOWN for everything —
  common in early epochs and not by itself a bug;
- a model can beat the baseline on accuracy and lose on F1, or the reverse.
  `main.py evaluate` prints both next to the baseline for this reason.

`<- best` marks an improvement in validation loss, which is when `best.pt` is
rewritten.

## Scoring a checkpoint

```bash
python main.py evaluate                      # best.pt on the test days
python main.py evaluate --subset val         # while iterating
python main.py evaluate --weights checkpoints/epoch_012.pt --subset val
```

```
test (1840 windows, days 17-20)
  loss       0.69290
  accuracy   0.5076   baseline 0.5125
  precision  0.5165   baseline 0.5125
  recall     0.6151   baseline 1.0000
  f1         0.5615   baseline 0.6777
  predicted UP on 61.0% of windows (actual 51.2%)
```

> The numbers above come from a 2-epoch smoke run, so they are the *shape* of the
> output, not a result. That model loses to the baseline on every metric — which
> is what an undertrained model should look like.

**Run the test subset once, at the end.** `train.py` reads `val` every epoch and
never touches `test`; scoring the test set is deliberately a separate command so
a test number cannot leak into a training decision by accident. Tuning against it
turns it into a second validation set and the reported figure stops meaning
anything.

## Configuration

Everything lives in `model_training/config.py`. There is no second place to edit.

**Data shape** — changing any of these invalidates existing checkpoints:

| setting | default | notes |
| --- | --- | --- |
| `FEATURE_CHANNELS` | the 8 channels | `NUM_INPUT_FEATURES` is derived from it, so model, loader and exporter all follow one edit |
| `SEQUENCE_LENGTH` | `32` | bars per window. Windows-per-session is `78 - SEQUENCE_LENGTH - HORIZON + 1`, so raising it costs examples |
| `HORIZON` | `1` | how many bars ahead the label looks |
| `LABEL_THRESHOLD` | `0.0` | fractional move required for UP; `0.001` = 0.1% |
| `NORMALIZE` | `"window-z"` | or `None` for raw prices (see the warning in Part 4) |

**Splits:**

| setting | default | notes |
| --- | --- | --- |
| `TEST_DAYS` | `4` | matches the backend's `subset="test"` — changing it breaks comparability with the team |
| `VAL_DAYS` | `3` | carved out of what the backend calls `train` |

**Model and optimiser:** `CONV_CHANNELS`, `KERNEL_SIZE`, `DROPOUT`, `EPOCHS`,
`BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `GRAD_CLIP_NORM`.

**API:** `API_BASE_URL` (env `STOCKI_API_URL`), `API_TIMEOUT` (env
`STOCKI_API_TIMEOUT`), `API_PAGE_SIZE`, `API_MAX_RETRIES`.

## What a checkpoint holds

`checkpoints/epoch_XXX.pt` every epoch, mirrored to `latest.pt` (the resume
point), plus `best.pt` on every validation-loss improvement. `best.pt` is what
`export` uses by default.

```python
{
  "epoch": 12,
  "model_state_dict": {...},
  "optimizer_state_dict": {...},
  "model_config": {"num_input_features": 8, "num_outputs": 1,
                   "conv_channels": [64, 128, 128], "kernel_size": 5, "dropout": 0.1},
  "sequence_length": 32,
  "feature_channels": ["open", "high", ..., "co_range"],   # the data contract
  "horizon": 1,
  "normalize": "window-z",
  "metrics": {"train": {...}, "val": {...}, "val_baseline": {...}},
  "best_val_loss": 0.6912,
}
```

The architecture travels with the weights, so `export_onnx.py` rebuilds the right
model from the file alone. The data contract travels too, so you can tell what a
stale checkpoint was trained on:

```python
import torch
ck = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
print(ck["epoch"], ck["feature_channels"], ck["sequence_length"], ck["metrics"]["val"])
```

## The model

`model_training/model.py` — a 1D CNN, 159,425 parameters:

```
input (batch, 8, 32)
  -> ConvBlock(8 -> 64)     Conv1d(k=5, same padding) -> BatchNorm -> GELU -> Dropout
  -> ConvBlock(64 -> 128)
  -> ConvBlock(128 -> 128)
  -> concat(AdaptiveAvgPool1d(1), AdaptiveMaxPool1d(1))    -> (batch, 256)
  -> Linear(256 -> 128) -> GELU -> Dropout -> Linear(128 -> 1) -> Sigmoid
output (batch, 1)          P(up)
```

Trained with `BCELoss` against the sigmoid output. Because it pools over the time
axis, **a trained model accepts sequence lengths other than 32** — which is why
`sequence_length` is a dynamic axis in the export.

```bash
python model.py     # print the architecture and run a forward-pass shape check
```

---

# Part 3 — exporting to ONNX

```bash
python main.py export                                    # best.pt -> artifacts/model.onnx
python main.py export --weights checkpoints/latest.pt
python main.py export --output /tmp/model.onnx
python main.py export --sequence-length 60               # only fixes the tracing input
python main.py export --no-verify
```

```
exported checkpoints/best.pt -> artifacts/model.onnx
verified: output shape (1, 1), max |onnx - torch| = 0.000e+00
```

The verification step runs the exported graph and the PyTorch model on the same
input and compares. It needs `onnxruntime`; without it the step is skipped with a
message. A `max |onnx - torch|` above `1e-4` prints a warning — treat that as an
export bug, not a rounding artefact.

## The exported graph

```
input   "input"   (batch, 8, sequence_length)   float32
output  "output"  (batch, 1)                    float32   P(up), sigmoid applied
opset 17, ~641 KB
```

`batch` and `sequence_length` are **dynamic**; the feature axis is **fixed at 8**.
So the same file serves batches of any size and windows of any length:

```python
>>> for T in (16, 32, 60):
...     predictor.predict(np.random.randn(T, 8).tolist())
[0.531642]      # T=16
[0.506577]      # T=32
[0.501379]      # T=60
```

The sigmoid is **inside** the graph. The output is already a probability — do not
apply another one.

---

# Part 4 — loading the weights in ONNX Runtime

## ⚠️ The input must be normalised the same way it was in training

This is the one mistake that produces confidently wrong predictions instead of an
error. The model was trained on **window-z normalised** input: each of the 8
channels centred at 0 with unit spread, using only the bars inside that window.
Feed it raw prices and it saturates:

```python
P(up) on window-z normalised input =  0.5048      # sensible
P(up) on raw OHLCV prices          =  1.0000      # saturated, meaningless
```

Both are valid float32 arrays of shape `(32, 8)`, so nothing raises. Volume is in
the millions while `log_return` is around `1e-4`; without normalisation the first
convolution sees numbers seven orders of magnitude apart and the sigmoid pins to
1.0.

Nothing downstream can detect this for you. Normalise on the way in.

## Recommended: reuse the training transform

The safest preprocessing is the code that built the training data. It is
importable and has no torch dependency for this path:

```python
import json
import urllib.request

import numpy as np
import onnxruntime as ort

import config
from api_client import BAR_DTYPE
from dataloader import derive_channels, normalize_windows

# 1. the most recent session of bars, straight from the backend
rows = json.load(urllib.request.urlopen("http://localhost:8000/api/v1/bars/NVDA/20"))
bars = np.array(
    [
        (r["ticker"], r["day"], np.datetime64(r["timestamp"].rstrip("Z"), "ns"),
         r["open"], r["high"], r["low"], r["close"], r["volume"])
        for r in rows
    ],
    dtype=BAR_DTYPE,
)

# 2. the last SEQUENCE_LENGTH bars -> 8 channels -> window-z normalised
channels = derive_channels(bars[-config.SEQUENCE_LENGTH:])
window = np.column_stack([channels[name] for name in config.FEATURE_CHANNELS])[None, ...]
window = normalize_windows(window, config.NORMALIZE).astype(np.float32)   # (1, 32, 8)

# 3. channels-last -> channels-first, which is what the graph wants
batch = np.ascontiguousarray(window.transpose(0, 2, 1))                   # (1, 8, 32)

session = ort.InferenceSession("artifacts/model.onnx", providers=["CPUExecutionProvider"])
probability = float(session.run(None, {"input": batch})[0][0, 0])
print(f"P(up) = {probability:.4f}")        # P(up) = 0.5048
```

(That value is from an undertrained smoke-run model; what matters here is that
the call succeeds and the number is a plausible probability rather than a
saturated 0 or 1.)

Two shape conversions are easy to get wrong, so both are called out:

| | layout | who uses it |
| --- | --- | --- |
| the API and `derive_channels` | `(timesteps, features)` | CSVs, `/api/v1/bars`, `/api/v1/dataset/windows` |
| the ONNX graph and Conv1d | `(features, timesteps)` | the model |

`normalize_windows` expects a **3D** `(n_windows, timesteps, channels)` array and
normalises along axis 1 — hence the `[None, ...]`. Passing a 2D array silently
normalises the wrong axis.

## Using `OnnxPredictor`

`model/inference.py` wraps all of the above except the normalisation, and takes
the natural row-per-timestep layout:

```python
from inference import OnnxPredictor

predictor = OnnxPredictor("../model_training/artifacts/model.onnx")
predictor.info()
# {'model_path': '...', 'input_features': 8, 'sequence_length': 'dynamic',
#  'providers': ['CPUExecutionProvider']}

predictor.predict(window[0].tolist())            # (timesteps, features) -> [0.5048]
predictor.predict_many([w.tolist() for w in windows])
```

It validates what it can — rectangular 2D input, no NaN or infinity, and the
feature count against the graph — and raises `ValueError` with a message naming
the mismatch. Note what it **cannot** check:

- **`sequence_length` is dynamic in the export, so `predictor.sequence_length` is
  `None` and timestep count is not validated.** Any length is accepted.
- Whether the values were normalised. See the warning above.

For GPU inference, pass providers explicitly:

```python
OnnxPredictor(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

## Serving it

```bash
cd model
pip install -r requirements.txt
uvicorn main:app --reload --port 8001        # http://127.0.0.1:8001/docs
```

**Use a port other than 8000.** `model/main.py` defaults to 8000, which is where
the backend API already is. Pass `--port` to uvicorn or set `PORT`.

The service looks for `../model_training/artifacts/model.onnx` by default;
override with `MODEL_PATH`:

```bash
MODEL_PATH=/path/to/model.onnx uvicorn main:app --port 8001
```

The model is loaded once at startup. A load failure does **not** crash the
service — `/health` reports it instead, which is why the first thing to check is
always:

```bash
curl localhost:8001/health
```

```json
{"model_loaded": true,
 "model_path": ".../model_training/artifacts/model.onnx",
 "detail": {"model_path": ".../model_training/artifacts/model.onnx",
            "input_features": 8,
            "sequence_length": "dynamic",
            "providers": ["CPUExecutionProvider"]},
 "error": null}
```

`200` with `model_loaded: true` when the model loaded, `503` with `false` and an
`error` string otherwise.

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | load status and graph info |
| POST | `/predict` | `{"sequence": [[...], ...]}` | `{"prediction": [0.5048]}` |
| POST | `/predict-many` | `{"sequences": [[[...], ...], ...]}` | `{"predictions": [[...], ...], "count": N}` |

`sequence` is `(timesteps, features)` — one row per timestep, **normalised**.
Because `sequence_length` is dynamic, a short sequence is accepted, which makes a
hand-written smoke test practical:

```bash
curl -X POST localhost:8001/predict -H "Content-Type: application/json" -d '{
  "sequence": [[ 0.1,-1.2, 0.4, 0.9,-0.3, 1.1,-0.7, 0.2],
               [-0.5, 0.8,-1.1, 0.3, 1.4,-0.9, 0.6,-0.2],
               [ 1.2, 0.1, 0.7,-1.3,-0.8, 0.5, 0.3, 1.0],
               [-0.8, 0.3, 0.0, 0.1,-0.3,-0.7,-0.2,-1.0]]}'
```

```json
{"prediction": [0.5273711681365967]}
```

```bash
curl -X POST localhost:8001/predict-many -H "Content-Type: application/json" -d '{
  "sequences": [[[0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]],
                [[0.5,0.1,-0.2,0.3,0.4,-0.1,0.2,0.0], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]]}'
```

```json
{"predictions": [[0.5252357125282288], [0.5255650281906128]], "count": 2}
```

Status codes: `422` for malformed or wrong-shaped input, `503` when the model is
not loaded. The `422` `detail` names the mismatch:

```bash
curl -X POST localhost:8001/predict -H "Content-Type: application/json" \
     -d '{"sequence": [[1, 2, 3]]}'
```

```json
{"detail": "sequence: model expects 8 features per timestep, got 3"}
```

Sequences of different lengths in one `predict-many` call cannot share a batch
dimension, so `OnnxPredictor` runs those one at a time rather than rejecting
them — correct, but slower than a uniform batch.

---

## Full retrain, start to finish

```bash
docker compose up -d                          # backend on :8000
cd model_training
python main.py data --refresh                 # refetch, verify against the API
python main.py train --fresh --epochs 30      # train from random weights
python main.py evaluate --subset val          # iterate against this
python main.py evaluate --subset test         # once, at the end
python main.py export                         # best.pt -> artifacts/model.onnx
cd ../model && uvicorn main:app --port 8001   # serve it
curl localhost:8001/health
```

---

## Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `StockiAPIError: cannot reach the stocki API at ...` | backend is down | `docker compose up -d`, or set `STOCKI_API_URL` |
| `warning: could not reach the API ...; using cached bars` | backend down, cache present | fine for offline work; the data may be stale |
| `bar_count` is not 14,586 | incomplete backend ingest | `stocki verify` on the backend side |
| `AssertionError: local windows disagree with /api/v1/dataset/windows` | the two transforms drifted | reconcile `dataloader.py` against `backend/stocki/datasets/windows.py` — do not train through it |
| `built windows of shape (N, 32) but config expects (8, 32)` | `FEATURE_CHANNELS` and `NUM_INPUT_FEATURES` disagree | `NUM_INPUT_FEATURES` is derived; you probably edited it by hand |
| `no session had enough bars for window=..., horizon=...` | `SEQUENCE_LENGTH + HORIZON > 78` | lower `SEQUENCE_LENGTH` |
| `days 1-20 cannot give up N test + M val days ...` | `TEST_DAYS + VAL_DAYS` too large | lower them |
| `train windows overlap val in time` | bad split config | do not work around this — it means leakage |
| `RuntimeError: Error(s) in loading state_dict` on resume | checkpoint predates an architecture or feature change | `--fresh`, or point `--weights` at the matching checkpoint |
| `f1 0.0000` for many epochs | model collapsed to all-DOWN | lower the learning rate, or check the class balance in `main.py data` |
| accuracy stuck near 0.51 | that is the baseline | see "Reading the output" — this task is near a coin flip |
| `no weights at checkpoints/best.pt` | never trained, or no epoch improved | train first, or `--weights checkpoints/latest.pt` |
| `WARNING: outputs diverge more than expected` on export | genuine export bug | do not ship the file; report it |
| `/health` returns `503` with `model_loaded: false` | ONNX file missing or unreadable | check `MODEL_PATH`; the `error` field says which |
| `422 model expects 8 features per timestep, got N` | wrong channel count | build the input from `config.FEATURE_CHANNELS` |
| predictions all ~1.0 or all ~0.0 | **input not normalised** | see the warning in Part 4 |

---

## File reference

**`model_training/`**

| file | what it holds |
| --- | --- |
| `config.py` | all tunables — feature channels, window, horizon, split sizes, API URL |
| `api_client.py` | the only file that touches the network: paged bar fetch, cache, `/health` validation |
| `dataloader.py` | windowing, labelling, normalisation, day-based splits, batching, API verification |
| `model.py` | `StockCNN1D` |
| `train.py` | training loop, metrics, baseline, checkpointing, resume, `evaluate()` |
| `export_onnx.py` | checkpoint → `.onnx`, with a PyTorch-equivalence check |
| `main.py` | CLI: `data` / `train` / `evaluate` / `export` |
| `.cache/bars.npz` | fetched bars (gitignored, regenerated on demand) |
| `checkpoints/` | `epoch_XXX.pt`, `latest.pt`, `best.pt` (gitignored) |
| `artifacts/model.onnx` | the export (gitignored) |

**`model/`**

| file | what it holds |
| --- | --- |
| `inference.py` | `OnnxPredictor` — ONNX Runtime session, layout transpose, input validation |
| `main.py` | FastAPI service: `/health`, `/predict`, `/predict-many` |

## See also

- `model_training/README.md` — the same ground, condensed
- `backend/README.md` — the data pipeline and the full API surface
- `docs/superpowers/specs/2026-08-19-stocki-backend-design.md` — why the backend
  is shaped the way it is, including the leakage controls this guide inherits
