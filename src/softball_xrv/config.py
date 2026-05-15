"""Project-wide paths and constants for the pre-modeling pipeline."""

from pathlib import Path

# config.py lives in src/<package>/, so parents[2] points back to the repo root.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR = PROJECT_DIR / "models"


RAW_CSV_PATH = RAW_DIR / "Trackman_Master_Data_final.csv"
CLEAN_DATA_PATH = INTERIM_DIR / "trackman_cleaned.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"


TARGET_COL = "xRV Of Event (Count Ind)"
GROUP_COL = "GameID"
RANDOM_STATE = 42
TEST_SIZE = 0.20
