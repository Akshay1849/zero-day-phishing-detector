"""Validation helpers for the benchmark's encoded feature representation.

The benchmark does not contain raw URLs. This module intentionally refuses to
invent a URL-to-feature mapping and validates only supplied encoded values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

ALL_FEATURE_ORDER = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick", "UsingPopupWindow",
    "IframeRedirection", "AgeofDomain", "DNSRecording", "WebsiteTraffic",
    "PageRank", "GoogleIndex", "LinksPointingToPage", "StatsReport",
]

FINAL_FEATURE_ORDER = [
    "HTTPS", "AnchorURL", "WebsiteTraffic", "SubDomains",
    "PrefixSuffix-", "LinksInScriptTags", "ServerFormHandler", "RequestURL",
    "LinksPointingToPage", "DomainRegLen", "AgeofDomain", "GoogleIndex",
    "DNSRecording", "UsingIP", "PageRank", "LongURL", "HTTPSDomainURL",
    "InfoEmail", "ShortURL",
]

_ALLOWED_VALUES = {
    feature: {-1, 0, 1}
    for feature in ALL_FEATURE_ORDER
    if feature not in {"UsingIP", "ShortURL", "Symbol@", "Redirecting//", "PrefixSuffix-", "DomainRegLen", "Favicon", "NonStdPort", "HTTPSDomainURL", "RequestURL", "InfoEmail", "AbnormalURL", "StatusBarCust", "DisableRightClick", "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording", "PageRank", "GoogleIndex", "StatsReport"}
}
for feature in set(ALL_FEATURE_ORDER) - set(_ALLOWED_VALUES):
    _ALLOWED_VALUES[feature] = {-1, 1}
_ALLOWED_VALUES["WebsiteForwarding"] = {0, 1}


class FeatureExtractionNotSupported(ValueError):
    """Raised when a raw URL cannot produce the benchmark feature vector."""


def extract_features_from_url(url: str) -> dict[str, Any]:
    """Reject raw URL extraction because the complete benchmark mapping is unavailable."""
    if not isinstance(url, str) or not url.strip():
        raise FeatureExtractionNotSupported("A raw URL was not provided.")
    raise FeatureExtractionNotSupported(
        "Raw URL extraction is unavailable for the benchmark feature format. "
        "Use the static URL model endpoint instead."
    )


def validate_feature_vector(
    features: Mapping[str, Any],
    feature_order: list[str] | tuple[str, ...] = ALL_FEATURE_ORDER,
) -> dict[str, float | int]:
    """Validate and return a complete feature vector in the requested order."""
    if not isinstance(features, Mapping):
        raise ValueError("features must be a JSON object.")
    expected = list(feature_order)
    missing = [feature for feature in expected if feature not in features]
    unexpected = [feature for feature in features if feature not in ALL_FEATURE_ORDER and feature not in {"Index"}]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected feature(s): {', '.join(unexpected)}")

    validated: dict[str, float | int] = {}
    for feature in expected:
        value = features[feature]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Feature {feature!r} must be numeric.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Feature {feature!r} must be finite.")
        if numeric_value not in _ALLOWED_VALUES[feature]:
            allowed = sorted(_ALLOWED_VALUES[feature])
            raise ValueError(f"Feature {feature!r} must be one of {allowed}.")
        else:
            validated[feature] = int(numeric_value)
    return validated


def select_final_features(validated_features: Mapping[str, Any]) -> list[float | int]:
    """Return the validated vector in the exact order expected by the final model."""
    validated = validate_feature_vector(validated_features, ALL_FEATURE_ORDER)
    return [validated[feature] for feature in FINAL_FEATURE_ORDER]


def allowed_values(feature: str) -> list[int]:
    """Return the benchmark values accepted for a feature."""
    if feature not in _ALLOWED_VALUES:
        raise KeyError(feature)
    return sorted(_ALLOWED_VALUES[feature])
