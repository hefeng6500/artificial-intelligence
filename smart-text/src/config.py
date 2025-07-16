from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

SEQ_LEN = 5
