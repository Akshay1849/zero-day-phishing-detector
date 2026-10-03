"""Static, URL-string-only feature extraction.

This module intentionally does not map these measurements to the benchmark's
encoded feature names. The benchmark has no raw URLs and several of its fields
require external page, DNS, registration, traffic, or reputation data.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections.abc import Mapping
from urllib.parse import parse_qsl, urlsplit, urlunsplit


class InvalidURL(ValueError):
    """Raised when an input is not a supported absolute HTTP(S) URL."""


_SUSPICIOUS_CHARS = set("<>\\\"'`{}[]|^")
_SHORTENER_HOSTS = {
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at",
}

URL_ONLY_FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length", "query_length", "fragment_length",
    "dot_count", "hyphen_count", "underscore_count", "digit_count", "special_character_count",
    "slash_count", "query_parameter_count", "url_depth", "subdomain_count", "uses_https",
    "uses_ip_address", "contains_at_symbol", "encoded_character_count", "suspicious_character_count",
    "url_entropy", "hostname_entropy", "path_entropy", "letter_count", "numeric_count",
    "digit_ratio", "special_character_ratio", "looks_like_shortener",
]


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {character: value.count(character) for character in set(value)}
    length = len(value)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def _validate_and_normalize(url: str) -> tuple[str, object]:
    if not isinstance(url, str) or not url.strip():
        raise InvalidURL("Enter a URL.")
    candidate = url.strip()
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise InvalidURL("Only absolute http:// and https:// URLs are supported.")
    if not parsed.netloc or parsed.username or parsed.password:
        raise InvalidURL("The URL must contain a host and must not contain credentials.")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise InvalidURL("The URL contains an invalid host or port.") from error
    if not hostname or any(character.isspace() for character in candidate):
        raise InvalidURL("The URL contains an invalid host or whitespace.")
    if port is not None and not 1 <= port <= 65535:
        raise InvalidURL("The URL port must be between 1 and 65535.")
    normalized = urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, parsed.fragment))
    return normalized, parsed


def extract_url_features(url: str) -> dict[str, object]:
    """Return deterministic measurements derived only from a URL string.

    No DNS lookup, HTTP request, redirect, HTML download, JavaScript execution,
    or external reputation query is performed.
    """
    normalized, parsed = _validate_and_normalize(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_without_fragment = normalized
    labels = [label for label in hostname.split(".") if label]
    domain_without_port = hostname
    query_pairs = parse_qsl(query, keep_blank_values=True)
    special_character_count = sum(not character.isalnum() for character in full_without_fragment)
    encoded_character_count = len(re.findall(r"%[0-9A-Fa-f]{2}", full_without_fragment))
    letter_count = sum(character.isalpha() for character in full_without_fragment)
    numeric_count = sum(character.isdigit() for character in full_without_fragment)
    feature_values = {
        "url_length": len(full_without_fragment),
        "hostname_length": len(domain_without_port),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),
        "dot_count": full_without_fragment.count("."),
        "hyphen_count": full_without_fragment.count("-"),
        "underscore_count": full_without_fragment.count("_"),
        "digit_count": numeric_count,
        "special_character_count": special_character_count,
        "slash_count": full_without_fragment.count("/"),
        "query_parameter_count": len(query_pairs),
        "url_depth": len([part for part in path.split("/") if part]),
        "subdomain_count": max(len(labels) - 2, 0),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "uses_ip_address": int(_is_ip_address(hostname)),
        "contains_at_symbol": int("@" in full_without_fragment),
        "encoded_character_count": encoded_character_count,
        "suspicious_character_count": sum(character in _SUSPICIOUS_CHARS for character in full_without_fragment),
        "url_entropy": _entropy(full_without_fragment),
        "hostname_entropy": _entropy(hostname),
        "path_entropy": _entropy(path),
        "letter_count": letter_count,
        "numeric_count": numeric_count,
        "digit_ratio": numeric_count / len(full_without_fragment) if full_without_fragment else 0.0,
        "special_character_ratio": special_character_count / len(full_without_fragment) if full_without_fragment else 0.0,
        "looks_like_shortener": int(hostname in _SHORTENER_HOSTS),
    }
    return {
        "url": normalized,
        "scheme": parsed.scheme.lower(),
        "hostname": hostname,
        "uses_https": parsed.scheme.lower() == "https",
        "uses_ip_address": _is_ip_address(hostname),
        "url_length": len(full_without_fragment),
        "hostname_length": len(domain_without_port),
        **feature_values,
        "contains_double_slash_in_path": "//" in path,
        "has_prefix_suffix_hyphen": any(label.startswith("-") or label.endswith("-") for label in labels),
        "has_suspicious_characters": bool(_SUSPICIOUS_CHARS.intersection(full_without_fragment)),
        "looks_like_shortener": hostname in _SHORTENER_HOSTS,
        "has_fragment": bool(parsed.fragment),
        "port": parsed.port,
        "safe_static_analysis_only": True,
    }


def extract_url_only_row(url: str) -> dict[str, float | int]:
    """Return only the documented numeric URL-string features for modeling."""
    features = extract_url_features(url)
    return {name: features[name] for name in URL_ONLY_FEATURE_ORDER}


def validate_url_features(features: Mapping[str, object]) -> dict[str, object]:
    """Validate the output shape of :func:`extract_url_features`."""
    if not isinstance(features, Mapping) or not features.get("safe_static_analysis_only"):
        raise ValueError("Features must come from static URL analysis.")
    for key in ("url", "scheme", "hostname"):
        if not isinstance(features.get(key), str) or not features[key]:
            raise ValueError(f"URL feature {key!r} is invalid.")
    for key, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            raise ValueError(f"URL feature {key!r} must be finite.")
    return dict(features)
