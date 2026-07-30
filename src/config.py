from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"

RANDOM_STATE = 42
TARGET_COL = "Class"
# Modelling
TEST_SIZE = 0.2
# Stratify on the target since fraud is ~0.17% of transactions —
# a random split without stratification risks a test set with
# very few (or zero) fraud cases.
STRATIFY_SPLIT = True