"""Central configuration for the stock-prediction training framework.

Everything that is still undecided about the data lives here so there is exactly
one place to edit once the feature set / data layout is finalized.
"""

from pathlib import Path

# =====================================================================
# >>> CHANGE ME <<<  INPUT FEATURE COUNT
# ---------------------------------------------------------------------
# Number of features per timestep (i.e. the number of Conv1d input
# channels). This is the ONLY place the value should be written down --
# the model, the dataloader and the ONNX exporter all read it from here.
#
# The raw CSVs in data/ currently carry ~120 columns, most of which are
# metadata or constant-per-day fundamentals, so this is a placeholder
# until the feature selection is settled.
NUM_INPUT_FEATURES = 8
# =====================================================================

# Number of consecutive timesteps in one training example (the length of the
# contiguous segment fed to the CNN). The model itself is length-agnostic
# (it pools over time), so this only fixes what training/export use.
SEQUENCE_LENGTH = 60

# Number of values the model predicts per example. 1 == next close price.
NUM_OUTPUTS = 1

# --- Paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# Checkpoint written/overwritten at the end of every epoch; also the file the
# training loop resumes from when it exists.
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

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
