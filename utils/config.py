"""
Config & Logging
=================
Central configuration and logging setup for CineMatch.
"""

import logging
import os
from dataclasses import dataclass, field


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class Config:
    """Global runtime configuration (override via environment variables)."""

    # Data
    data_dir: str           = os.getenv("DATA_DIR", "data")
    dataset:  str           = os.getenv("DATASET",  "movielens-100k")

    # Model hyper-parameters
    n_svd_factors: int      = int(os.getenv("N_SVD_FACTORS", "50"))
    cf_weight:     float    = float(os.getenv("CF_WEIGHT", "0.6"))
    cb_weight:     float    = float(os.getenv("CB_WEIGHT", "0.4"))
    tfidf_features: int     = int(os.getenv("TFIDF_FEATURES", "500"))

    # Evaluation
    eval_threshold: float   = float(os.getenv("EVAL_THRESHOLD", "3.0"))
    eval_k:         int     = int(os.getenv("EVAL_K", "10"))

    # API
    api_host: str           = os.getenv("API_HOST", "0.0.0.0")
    api_port: int           = int(os.getenv("API_PORT", "8000"))

    # GenAI
    anthropic_api_key: str  = os.getenv("ANTHROPIC_API_KEY", "")
    genai_model: str        = os.getenv("GENAI_MODEL", "claude-sonnet-4-20250514")


# Singleton
config = Config()
