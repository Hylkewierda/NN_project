"""Central configuration for all experiments."""

import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
TRAIN_IMG_DIR = DATA_DIR / "train_set" / "train_set" / "train_set"
TEST_IMG_DIR = DATA_DIR / "test_set" / "test_set" / "test_set"
TRAIN_LABELS_CSV = DATA_DIR / "train_labels.csv"
CLASS_LIST = DATA_DIR / "class_list_food.txt"
SAMPLE_CSV = DATA_DIR / "sample.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────
NUM_CLASSES = 80
IMG_SIZE = 128  # start small, increase for later experiments
DATASET_MEAN = [0.6310, 0.5407, 0.4424]
DATASET_STD = [0.2259, 0.2413, 0.2647]

# ── Training defaults ──────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42

# ── Device ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
