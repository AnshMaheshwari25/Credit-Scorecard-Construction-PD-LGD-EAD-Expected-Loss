"""Configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML configuration (default: configs/config.yaml)."""
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_data_path(cfg: dict) -> Path:
    """Find the loan CSV: env var CREDIT_RISK_DATA first, then the candidate paths in the config."""
    candidates = []
    env = os.environ.get("CREDIT_RISK_DATA")
    if env:
        candidates.append(Path(env))
    for p in cfg["data"]["candidate_paths"]:
        p = Path(p)
        candidates.append(p if p.is_absolute() else ROOT / p)
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Loan data not found. Download 'loan_data_2015.csv' from Kaggle into data/raw/ "
        "or set the CREDIT_RISK_DATA environment variable. Looked in:\n  " + tried
    )
