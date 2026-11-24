"""Typed loaders for the Phase 2 pipeline.

Every loader asserts the invariants established in Phase 1, so a corrupted or
swapped file fails loudly instead of silently producing a different dataset.

Raw files are opened read-only. Nothing here writes to ``datasets/raw``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))

from naming import canonical_district, canonical_key, canonical_state, strip_area_prefix  # noqa: E402

RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
METADATA = ROOT / "datasets" / "metadata"

D1_PATH = RAW / "india_census" / "india-districts-census-2011.csv"
D2_PATH = RAW / "india_census" / "india_census_housing-hlpca-full.csv"
D5_PATH = RAW / "india_pca" / "census_2011.csv"

N_DISTRICTS = 640
N_DISTRICTS_WITH_D5 = 628


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing raw dataset: {path.relative_to(ROOT)}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    dupes = int(frame.duplicated(subset=columns).sum())
    if dupes:
        raise AssertionError(f"{label}: {dupes} duplicate rows on {columns}")


def load_d1() -> pd.DataFrame:
    """District census 2011, one row per district, with canonical keys."""
    df = _read(D1_PATH)
    out = df.copy()
    out["key"] = [
        canonical_key(s, d) for s, d in zip(df["State name"], df["District name"])
    ]
    out["state_key"] = df["State name"].map(canonical_state)
    out["district_key"] = df["District name"].map(canonical_district)

    if len(out) != N_DISTRICTS:
        raise AssertionError(f"D1: expected {N_DISTRICTS} districts, found {len(out)}")
    if not out["key"].is_unique:
        raise AssertionError("D1: canonical district keys are not unique")
    if out["Population"].isna().any():
        raise AssertionError("D1: null population")
    return out


def load_d2_total() -> pd.DataFrame:
    """HLPCA district totals (Rural/Urban == 'Total'), one row per district."""
    df = _read(D2_PATH)
    df["Rural/Urban"] = df["Rural/Urban"].astype(str).str.strip()
    total = df[df["Rural/Urban"] == "Total"].copy()
    total["area_clean"] = total["Area Name"].map(strip_area_prefix)
    total["key"] = [
        canonical_key(s, a) for s, a in zip(total["State Name"], total["area_clean"])
    ]
    total["District Code"] = pd.to_numeric(total["District Code"], errors="coerce")

    if len(total) != N_DISTRICTS:
        raise AssertionError(
            f"D2: expected {N_DISTRICTS} district Total rows, found {len(total)}"
        )
    if not total["key"].is_unique:
        raise AssertionError("D2: duplicate district keys in Total rows")
    return total


def load_d5_households(d1: pd.DataFrame) -> pd.Series:
    """Census 2011 PCA household counts per district, aligned to ``d1``.

    D5 has no state column and no district code, so a district name is only
    usable when it identifies exactly one D1 district. The 12 districts whose
    names occur in two states stay missing: no borrowing, no imputation.
    """
    df = _read(D5_PATH)
    district = df[df["LEVEL"].astype(str).str.strip() == "DISTRICT"].copy()
    district["district_key"] = district["NAME"].map(canonical_district)

    agg = district.groupby("district_key", as_index=False).agg(
        no_hh=("NO_HH", "sum"), tot_p=("TOT_P", "sum"), n_rows=("AREA", "size")
    )

    name_freq = d1["district_key"].value_counts()
    resolvable = set(name_freq[name_freq == 1].index) & set(
        agg.loc[agg["n_rows"] == 2, "district_key"]
    )

    lookup = dict(zip(agg["district_key"], agg["no_hh"]))
    series = pd.Series(
        [lookup.get(name) if name in resolvable else np.nan for name in d1["district_key"]],
        index=d1["key"],
        name="d5_no_hh",
        dtype=float,
    )

    if int(series.notna().sum()) != N_DISTRICTS_WITH_D5:
        raise AssertionError(
            f"D5: expected {N_DISTRICTS_WITH_D5} resolvable household counts, "
            f"found {int(series.notna().sum())}"
        )
    if (series.dropna() <= 0).any():
        raise AssertionError("D5: non-positive household counts")
    return series


def assert_merge(frame: pd.DataFrame, expected_rows: int, label: str) -> pd.DataFrame:
    """Explicit row-count guard for every join."""
    if len(frame) != expected_rows:
        raise AssertionError(f"{label}: expected {expected_rows} rows, got {len(frame)}")
    if not frame["key"].is_unique:
        raise AssertionError(f"{label}: key not unique after merge")
    return frame
