"""Project paths and brand settings for Step 1."""

from pathlib import Path

# Repo root: hiver-sde-assignment/
ROOT_DIR = Path(__file__).resolve().parent.parent

BRAND = "AppleSupport"

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_CSV_PATH = RAW_DIR / "twcs.csv"
PROCESSED_CONVERSATIONS_PATH = PROCESSED_DIR / "applesupport_conversations.csv"
PROCESSED_SUPPORT_PAIRS_PATH = PROCESSED_DIR / "applesupport_support_pairs.csv"
GOLDEN_SET_CANDIDATES_PATH = PROCESSED_DIR / "golden_set_candidates.csv"
GOLDEN_SET_FINAL_PATH = PROCESSED_DIR / "golden_set_final.csv"
GOLDEN_SET_PREDICTIONS_PATH = PROCESSED_DIR / "golden_set_predictions.csv"
WEAK_TRAINING_PATH = PROCESSED_DIR / "applesupport_weak_training.csv"

MODELS_DIR = ROOT_DIR / "models"
BASELINE1_DIR = MODELS_DIR / "baseline1"
BASELINE1_MODEL_PATH = BASELINE1_DIR / "tfidf_logreg.joblib"
BASELINE1_META_PATH = BASELINE1_DIR / "tfidf_logreg_meta.joblib"

BASELINE2_DIR = MODELS_DIR / "baseline2"
BASELINE2_MODEL_PATH = BASELINE2_DIR / "tfidf_linearsvc.joblib"
BASELINE2_META_PATH = BASELINE2_DIR / "tfidf_linearsvc_meta.joblib"

RETRIEVAL_DIR = MODELS_DIR / "retrieval"
RETRIEVAL_INDEX_PATH = RETRIEVAL_DIR / "tfidf_cosine_index.joblib"

# Approved preliminary taxonomy (not ground-truth labels on the full dataset).
INTENT_TAXONOMY = [
    "battery_drain",
    "charging_issue",
    "software_update_issue",
    "hardware_issue",
    "network_connectivity",
    "account_access",
    "app_store_issue",
    "app_issue",
    "payment_or_refund",
    "subscription_management",
    "storage_issue",
    "howto_or_feature",
    "other_or_unclear",
]
