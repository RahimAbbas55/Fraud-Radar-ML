from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"

RANDOM_STATE = 42
TARGET_COL = "Class"
TEST_SIZE = 0.2
STRATIFY_SPLIT = True