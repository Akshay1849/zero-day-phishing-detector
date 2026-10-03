"""Dataset preparation helpers for the raw-URL training pipeline.

This module extracts deterministic URL-string features from the raw URL dataset
and groups URLs by registrable domain for domain-aware train/test splitting.
"""

from __future__ import annotations

import ipaddress
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

# Allow this module to be run from the training/ directory or the project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.url_feature_extraction import InvalidURL, URL_ONLY_FEATURE_ORDER, extract_url_only_row  # noqa: E402

_SECOND_LEVEL_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "com.au", "co.jp", "co.in", "com.br"}


def registrable_domain(url: str) -> str:
    """Derive a deterministic grouping key from the parsed hostname only."""
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    labels = [label for label in hostname.split(".") if label]
    if len(labels) <= 2:
        return hostname
    suffix = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix in _SECOND_LEVEL_SUFFIXES else suffix


def engineer_url_dataset(
    raw: pd.DataFrame,
    url_column: str = "URL",
    label_column: str = "label",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract URL-only features, remove invalid/duplicate URLs, and preserve labels."""
    rows = []
    invalid_count = 0
    for url, label in raw[[url_column, label_column]].itertuples(index=False, name=None):
        try:
            row = extract_url_only_row(url)
        except (InvalidURL, TypeError, ValueError):
            invalid_count += 1
            continue
        row = {"URL": url, **row, "label": int(label), "domain_group": registrable_domain(url)}
        rows.append(row)
    engineered = pd.DataFrame(rows)
    before = len(engineered)
    engineered = engineered.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    duplicate_url_count = before - len(engineered)
    if engineered.empty:
        raise ValueError("No valid URL rows remain after extraction.")
    if engineered["label"].isin([0, 1]).all() is False:
        raise ValueError("The raw URL label column must contain only 0 (Phishing) and 1 (Legitimate).")
    numeric_columns = list(URL_ONLY_FEATURE_ORDER)
    if engineered[numeric_columns].isna().any().any():
        raise ValueError("URL-only extraction produced missing numeric values.")
    if not engineered[numeric_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce").notna().all()
    ).all():
        raise ValueError("URL-only extraction produced nonnumeric values.")
    return engineered, {
        "invalid_rows_removed": invalid_count,
        "duplicate_urls_removed": duplicate_url_count,
        "valid_rows": len(engineered),
        "unique_domains": int(engineered["domain_group"].nunique()),
    }
