"""Side-effect-free project path configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# This extracted directory is source data and must remain read-only.
SOURCE_DATASET_DIR = (
    PROJECT_ROOT
    / "Sample_Ecommerce_Capstone_Dataset"
    / "Sample_Ecommerce_Capstone_Dataset"
)

WORKING_DATA_DIR = PROJECT_ROOT / "data"
CHROMA_STORE_DIR = PROJECT_ROOT / "chroma_store"
REPORT_DIR = PROJECT_ROOT / "report"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"
