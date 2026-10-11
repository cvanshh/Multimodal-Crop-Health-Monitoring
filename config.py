"""Central configuration for AgriSpectra.AI"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Dataset paths (change if your folders are elsewhere) ----
DATA_DIR = BASE_DIR / "archive" / "train"        # train/<Class>/*.tif
TEST_DIR = BASE_DIR / "archive" / "test"         # optional labelled test folder
VAL_DIR = BASE_DIR / "archive" / "val" / "val"   # (usually unlabelled)

# ---- Model artefacts ----
MODEL_DIR = BASE_DIR / "models"
CNN_PATH = MODEL_DIR / "cnn_hsi.keras"
LSTM_PATH = MODEL_DIR / "lstm_iot.keras"
FUSION_PATH = MODEL_DIR / "fusion.keras"
META_PATH = MODEL_DIR / "meta.json"
RESULTS_PATH = MODEL_DIR / "results.json"
MODEL_PATH = CNN_PATH
CLASSES_PATH = MODEL_DIR / "classes.json"
HISTORY_PATH = MODEL_DIR / "history.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
REFSPEC_PATH = MODEL_DIR / "reference_spectra.json"

# ---- Imaging ----
IMG_SIZE = 128         # paper: 128x128 (files are 64x64, so this upsamples; use --img_size 64 to skip)
NUM_BANDS = 125        # expected bands in the competition TIFFs
WL_MIN, WL_MAX = 450.0, 950.0   # nm, assumed spectral range (verify with your data!)

DEFAULT_CLASS_NAMES = ["Health", "Other", "Rust"]
SEED = 42
