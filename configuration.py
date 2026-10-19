from pathlib import Path

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATASET_DIR = PROJECT_ROOT / "original_dataset"
OUTPUT_DATASET_DIR = PROJECT_ROOT / "dataset_split"
TRAIN_DIR = OUTPUT_DATASET_DIR / "train"
VALID_DIR = OUTPUT_DATASET_DIR / "valid"
TEST_DIR = OUTPUT_DATASET_DIR / "test"
CLASS_NAMES_PATH = OUTPUT_DATASET_DIR / "class_names.json"
MANIFEST_PATH = OUTPUT_DATASET_DIR / "split_manifest.csv"
TFRECORD_DIR = PROJECT_ROOT / "tfrecords"

# Models and reports
LATEST_MODEL_DIR = PROJECT_ROOT / "saved_model"
LATEST_WEIGHTS_PATH = PROJECT_ROOT / "checkpoints" / "latest.weights.h5"
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
TRAINING_STATE_PATH = PROJECT_ROOT / "checkpoints" / "training_state.json"
INSPECT_WORSE_VALID_DIR = PROJECT_ROOT / "inspect_worse_valid"
HARD_POOL_DIR = PROJECT_ROOT / "hard_pool"
REPORT_DIR = PROJECT_ROOT / "reports"

# -----------------------------------------------------------------------------
# Dataset split
# -----------------------------------------------------------------------------
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 20250809
COPY_MODE = "copy"  # "copy" or "move"
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 0_nocard and 54_zerocard are merged into one class.
NON_CARD_OUTPUT_NAME = "0_nocard"
NON_CARD_ALIASES = (
    "0", "0_nocard", "nocard", "no_card", "noncard", "non_card",
    "54", "54_zerocard", "zerocard", "zero_card",
)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
PYTHON_TARGET = "3.10.12"
EPOCHS = 2000
BATCH_SIZE = 16
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
CHANNELS = 3
LEARNING_RATE = 1e-4
DROPOUT_KEEP_PROB = 0.80

# Shared preprocessing. Every Step3/Step4 path uses the same settings.
RESIZE_MODE = "stretch"  # "stretch" or "pad"
NORMALIZE_MODE = "minus_one_to_one"  # Inception style: [0,255] -> [-1,1]
DATA_AUGMENTATION = True

# Resume / pretrained behavior
RESUME_LATEST = True
LOAD_PRETRAINED = False
PRETRAINED_WEIGHTS_PATH = PROJECT_ROOT / "pretrained" / "inception_v4.weights.h5"

# Saving behavior
SAVE_LATEST_EVERY_EPOCH = True
SNAPSHOT_EVERY_EPOCHS = 10
ENABLE_F12_MANUAL_SAVE = True

# Validation inspection
DUMP_WHEN_VALID_WORSE = True
MAX_WORSE_VALID_IMAGES = 500

# Hard-pool behavior. Any hard-pool image whose SHA-1 already appears in
# valid/test is automatically excluded, preventing train/validation leakage.
USE_HARD_POOL = True
OVERSAMPLE_TRAIN = False
MAX_OVERSAMPLE_PER_CLASS = 0  # 0 = oversample to current largest class

# Step4
SMOKE_MAX_PER_CLASS = 0  # 0 = all files; e.g. 10 for smoke10 reports
STEP4_SCORE_DECIMALS = 6
STEP4_WORKERS = 1
