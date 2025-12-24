from pathlib import Path

# Get project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths (relative to project root)
BASE_PATH = PROJECT_ROOT / "data" / "raw"
hosp_path = BASE_PATH / "hosp"
icu_path = BASE_PATH / "icu"
ed_path = BASE_PATH / "ed"
note_path = BASE_PATH / "note"
path = hosp_path
dataset_path = PROJECT_ROOT / "data" / "bronze" / "bronze_dataset.csv"