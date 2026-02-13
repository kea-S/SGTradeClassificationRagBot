from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = (REPO_ROOT / "data" / "raw").resolve()
INTERMEDIATE_DATA_DIR = (REPO_ROOT / "data" / "intermediate").resolve()
PROCESSED_DATA_DIR = (REPO_ROOT / "data" / "processed").resolve()
