"""Canonical naming helpers for Indian state and district identifiers.

The datasets in this project were produced in different years and by different
publishers, so the same state appears under several spellings. Every alias in
``OBSERVED_STATE_ALIASES`` was observed in this workspace and verified against
the census extract; aliases are additive and never silently rename a district.

Joins must always use ``canonical_key`` (state + district). District names on
their own are not unique: Aurangabad, Raigarh, Hamirpur, Bilaspur, Bijapur and
Pratapgarh each occur in two states.
"""

from __future__ import annotations

import re

# Verified in this project's data:
#   ORISSA      (D1)            <-> ODISHA      (D2)
#   PONDICHERRY (D1)            <-> PUDUCHERRY  (D2, D4)
#   NCT OF DELHI(D1)            <-> DELHI       (D4)
OBSERVED_STATE_ALIASES: dict[str, str] = {
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "DELHI": "NCT OF DELHI",
    "UTTARANCHAL": "UTTARAKHAND",
}

# Orthographic district variants. Only entries verified in this workspace.
#   PONDICHERRY (D1 district name) <-> PUDUCHERRY (D2, geography.csv district name)
DISTRICT_ALIASES: dict[str, str] = {
    "PONDICHERRY": "PUDUCHERRY",
}


def normalise_name(value: object) -> str:
    """Upper-case and strip punctuation so names compare reliably."""
    if value is None:
        return ""
    text = str(value).upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_state(value: object) -> str:
    """Map a state spelling to one canonical label."""
    name = normalise_name(value)
    seen: set[str] = set()
    while name in OBSERVED_STATE_ALIASES and name not in seen:
        seen.add(name)
        name = OBSERVED_STATE_ALIASES[name]
    return name


def canonical_district(value: object) -> str:
    """Map a district spelling to one canonical label (within a state)."""
    name = normalise_name(value)
    return DISTRICT_ALIASES.get(name, name)


def canonical_key(state: object, district: object) -> str:
    """Stable join key: canonical state '||' canonical district."""
    return f"{canonical_state(state)}||{canonical_district(district)}"


def strip_area_prefix(value: object) -> str:
    """Remove the 'District - ' / 'State - ' prefixes used by census extracts."""
    text = str(value).strip()
    return re.sub(r"^(District|State)\s*-\s*", "", text, flags=re.IGNORECASE).strip()
