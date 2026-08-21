"""Central configuration for the stock-prediction training framework.

Everything the training run depends on lives here, so there is exactly one place
to edit. The data-shape values are no longer placeholders: they are pinned to
what the backend API actually serves (see ``backend/README.md``).
"""

from pathlib import Path
import os

# =====================================================================
# DATA SOURCE -- the backend API
# ---------------------------------------------------------------------
# The dataloader reads bars over HTTP from the stocki backend
# (`docker compose up -d` in the repo root). Nothing here touches Postgres
# or the CSVs directly.
API_BASE_URL = os.environ.get("STOCKI_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.environ.get("STOCKI_API_TIMEOUT", "30"))

# /api/v1/bars is capped at 1000 rows per request; 14,586 bars is 15 pages.
API_PAGE_SIZE = 1000
# The dataset routes are rate-limited (20/min), the rest 120/min. Retries are
# for transient 429/5xx only.
API_MAX_RETRIES = 4
API_RETRY_BACKOFF = 1.5  # seconds, doubled per attempt

# =====================================================================
# FEATURES
# ---------------------------------------------------------------------
# The channels the backend derives from each bar. `open..volume` come straight
# from the CSV; the last three are scale-free shapes computed from the current
# bar and the one before it (never later ones -- no look-ahead).
#
# This tuple is the single source of truth: NUM_INPUT_FEATURES is derived from
# it, and the model / dataloader / ONNX exporter all read that.
FEATURE_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "hl_range",
    "co_range",
)
NUM_INPUT_FEATURES = len(FEATURE_CHANNELS)  # 8

# Number of consecutive 5-minute bars in one training example. A session is 78
# bars, so this must stay well under that: windows-per-session is
# `78 - SEQUENCE_LENGTH - HORIZON + 1`. 32 is the backend's documented default
# and yields 8,602 windows across the 187 sessions.
#
# The model pools over time, so a trained network accepts other lengths; this
# only fixes what training and the ONNX export use.
SEQUENCE_LENGTH = 32

# Number of values the model predicts per example.
# 1 == probability of next-period up-move (binary classification).
NUM_OUTPUTS = 1

# =====================================================================
# LABELS
# ---------------------------------------------------------------------
# 1 (UP) when the close `HORIZON` bars after the window ends is more than
# LABEL_THRESHOLD (a fractional return) above the close at the window end. The
# bar the label reads is never inside the window.
HORIZON = 1
LABEL_THRESHOLD = 0.0

# Per-window, per-channel z-scoring: each channel is centred at 0 with unit
# spread using only the bars inside that window. Dataset-wide statistics would
# fold the test set into training. Set to None for raw prices.
NORMALIZE = "window-z"

# =====================================================================
# TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------
# Splits are by *trading day*, never random. Windows overlap by
# SEQUENCE_LENGTH-1 bars, so a random split puts near-identical windows on both
# sides of the boundary and the validation score becomes fiction.
#
# With days 1-20 collected, the defaults below give:
#     train  days 1-13   (5,382 windows)
#     val    days 14-16  (1,380 windows)
#     test   days 17-20  (1,840 windows)
#
# TEST_DAYS matches the backend's own `subset="test"`, so `test` here is exactly
# the API's test set and results stay comparable across the team.
TEST_DAYS = 4
VAL_DAYS = 3

# --- Paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = MODULE_DIR / "checkpoints"
ARTIFACT_DIR = MODULE_DIR / "artifacts"

# Bars fetched from the API are cached here so repeated runs do not re-page the
# service. Safe to delete at any time; the loader refetches.
CACHE_DIR = MODULE_DIR / ".cache"

# Checkpoint written/overwritten at the end of every epoch; also the file the
# training loop resumes from when it exists.
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
# Best-validation-loss checkpoint, which is the one worth exporting.
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# --- Model hyperparameters -------------------------------------------
CONV_CHANNELS = (64, 128, 128)
KERNEL_SIZE = 5
DROPOUT = 0.1

# --- Training hyperparameters ----------------------------------------
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0  # set to None to disable

# --- ONNX export -----------------------------------------------------
ONNX_OPSET = 17
