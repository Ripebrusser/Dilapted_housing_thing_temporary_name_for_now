"""Target construction for Phase 2.

Decision D-001 (approved): the sole final target is the Census-computed share of
households living in dilapidated census houses, taken from the HLPCA district
``Total`` row (D2). T1 (D1 dilapidated / D1 Households) and T3 (D1 dilapidated /
D5 households) are **not** used as alternative targets in the modelling pipeline.

The target is a district-level prevalence measure. It is not a homelessness rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_NAME = "target_deprivation_rate_pct"
SOURCE_DATASET = "D2 (Census 2011 Houselisting & Housing Census Abstract)"
SOURCE_COLUMN = "Total Number of Dilapidated"
SOURCE_SUBSET = "Rural/Urban == 'Total'"
UNIT = "percent of households in the district"
N_EXPECTED = 640


def build_target(d2_total: pd.DataFrame) -> pd.DataFrame:
    """Return ``key`` + target + provenance columns, with hard validation."""
    values = pd.to_numeric(d2_total[SOURCE_COLUMN], errors="coerce")
    frame = pd.DataFrame(
        {
            "key": d2_total["key"].to_numpy(),
            "target_deprivation_rate_pct": values.to_numpy(),
        }
    )

    if len(frame) != N_EXPECTED:
        raise AssertionError(f"target: expected {N_EXPECTED} districts, got {len(frame)}")
    if not frame["key"].is_unique:
        raise AssertionError("target: duplicate district keys")
    if frame["target_deprivation_rate_pct"].isna().any():
        raise AssertionError("target: missing values present")
    if frame["target_deprivation_rate_pct"].duplicated().all():
        raise AssertionError("target: constant column")

    t = frame["target_deprivation_rate_pct"]
    if t.min() < 0 or t.max() > 100:
        raise AssertionError(f"target: outside 0-100 range ({t.min()}, {t.max()})")

    # Internal consistency of the D2 cross-tabulation: Good + Livable +
    # Dilapidated must be ~100 for the same row set, otherwise the share we use
    # is not a share of all households.
    comps = ["Total Number of Good", "Total Number of Livable", SOURCE_COLUMN]
    present = [c for c in comps if c in d2_total.columns]
    if len(present) == 3:
        total = sum(pd.to_numeric(d2_total[c], errors="coerce") for c in present)
        if not ((total >= 99.5) & (total <= 100.5)).all():
            raise AssertionError("target: condition components do not sum to ~100")

    return frame


def target_metadata() -> dict:
    """Machine-readable provenance for the feature manifest and report."""
    return {
        "target_name": TARGET_NAME,
        "source_dataset": SOURCE_DATASET,
        "source_column": SOURCE_COLUMN,
        "source_subset": SOURCE_SUBSET,
        "formula": (
            "100 x (households living in dilapidated census houses) / (total "
            "households in the census-house condition table)"
        ),
        "unit": UNIT,
        "n_districts": N_EXPECTED,
        "interpretation": (
            "District-level prevalence of dilapidated census houses as a share of "
            "households. A structural-condition measure of severe housing "
            "deprivation, not a homelessness measure."
        ),
        "rejected_alternatives": {
            "T1": "D1 dilapidated / D1 Households - invalid denominator (census houses)",
            "T3": "D1 dilapidated / D5 households - robustness check only, not a target",
        },
    }


def describe_target(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(len(s)),
        "min": round(float(s.min()), 3),
        "p05": round(float(s.quantile(0.05)), 3),
        "p25": round(float(s.quantile(0.25)), 3),
        "median": round(float(s.median()), 3),
        "mean": round(float(s.mean()), 3),
        "p75": round(float(s.quantile(0.75)), 3),
        "p95": round(float(s.quantile(0.95)), 3),
        "max": round(float(s.max()), 3),
        "sd": round(float(s.std()), 3),
        "skew": round(float(s.skew()), 3),
        "n_unique": int(s.nunique()),
        "n_null": int(pd.to_numeric(series, errors="coerce").isna().sum()),
        "n_negative": int((s < 0).sum()),
        "n_above_100": int((s > 100).sum()),
    }
