"""Phase 1 contract tests.

These lock in the invariants the audit established, so a later refactor cannot
silently break the target definition or the geographic matching.

Run with plain Python (no pytest required):

    py tests/test_phase1_contracts.py

Also compatible with pytest if it is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))

from naming import (  # noqa: E402
    canonical_district,
    canonical_key,
    canonical_state,
    strip_area_prefix,
)

RAW = ROOT / "datasets" / "raw"
META = ROOT / "datasets" / "metadata"

NUM = "Condition_of_occupied_census_houses_Dilapidated_Households"


def d1() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "india_census" / "india-districts-census-2011.csv", encoding="utf-8-sig"
    )


def d2() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "india_census" / "india_census_housing-hlpca-full.csv", encoding="utf-8-sig"
    )


def d4() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "slum_population" / "Slum Population Of India.csv", encoding="utf-8-sig"
    )


# ---------------------------------------------------------------------------
# Naming / aliases
# ---------------------------------------------------------------------------
def test_state_aliases_collapse() -> None:
    assert canonical_state("Orissa") == canonical_state("ODISHA") == "ODISHA"
    assert canonical_state("Pondicherry") == canonical_state("Puducherry") == "PUDUCHERRY"
    assert canonical_state("Delhi") == canonical_state("NCT of Delhi") == "NCT OF DELHI"
    assert canonical_state("Jammu & Kashmir") == canonical_state("Jammu and Kashmir")


def test_district_alias() -> None:
    assert canonical_district("Pondicherry") == canonical_district("Puducherry") == "PUDUCHERRY"


def test_area_prefix_stripped() -> None:
    assert strip_area_prefix("District - Kupwara") == "Kupwara"
    assert strip_area_prefix("State - LAKSHADWEEP") == "LAKSHADWEEP"


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
def test_district_keys_unique_and_complete() -> None:
    frame = d1()
    keys = [canonical_key(s, d) for s, d in zip(frame["State name"], frame["District name"])]
    assert len(keys) == 640, len(keys)
    assert len(set(keys)) == 640, "district keys are not unique"


def test_district_codes_are_globally_unique_in_d1() -> None:
    """D1 codes districts 1..640 across the whole country, not per state."""
    codes = d1()["District code"]
    assert codes.is_unique, "District code repeats within D1"
    assert codes.nunique() == 640
    assert int(codes.min()) == 1 and int(codes.max()) == 640


def test_district_codes_align_between_d1_and_d2() -> None:
    """The two census files use the same 1..640 district coding."""
    total = d2()[d2()["Rural/Urban"].astype(str).str.strip() == "Total"]
    assert total["District Code"].is_unique, "D2 district codes repeat"
    merged = d1()[["District code", "State name", "District name"]].merge(
        total[["District Code", "State Name", "Area Name"]],
        left_on="District code",
        right_on="District Code",
    )
    assert len(merged) == 640, f"code join produced {len(merged)} rows"
    mismatched = [
        (s1, d1_, s2, strip_area_prefix(a2))
        for s1, d1_, s2, a2 in zip(
            merged["State name"], merged["District name"], merged["State Name"], merged["Area Name"]
        )
        if canonical_key(s1, d1_) != canonical_key(s2, strip_area_prefix(a2))
    ]
    assert not mismatched, f"district codes disagree on {len(mismatched)} rows: {mismatched[:5]}"


def test_six_district_names_are_duplicated_across_states() -> None:
    frame = d1()
    names = frame["District name"].map(canonical_district)
    duplicated = names[names.duplicated(keep=False)].unique()
    assert len(duplicated) == 6, sorted(duplicated)
    assert int(names.duplicated().sum()) == 6


def test_d1_d2_districts_match_completely() -> None:
    frame1 = d1()
    frame2 = d2()
    keys1 = {canonical_key(s, d) for s, d in zip(frame1["State name"], frame1["District name"])}
    total = frame2[frame2["Rural/Urban"].astype(str).str.strip() == "Total"]
    keys2 = {
        canonical_key(s, strip_area_prefix(a))
        for s, a in zip(total["State Name"], total["Area Name"])
    }
    assert keys1 - keys2 == set(), f"D1 districts missing from D2: {sorted(keys1 - keys2)}"
    assert keys2 - keys1 == set(), f"D2 districts missing from D1: {sorted(keys2 - keys1)}"


# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------
def test_target_columns_present_and_compatible() -> None:
    frame = d1()
    num = pd.to_numeric(frame[NUM], errors="coerce")
    den = pd.to_numeric(frame["Households"], errors="coerce")
    assert num.notna().all() and den.notna().all(), "target columns contain nulls"
    assert (den > 0).all(), "zero denominators present"
    assert (num <= den).all(), "numerator exceeds denominator"


def test_d1_households_is_not_the_household_universe() -> None:
    """Guard against someone restoring the rejected denominator."""
    frame = d1()
    total = float(pd.to_numeric(frame["Households"], errors="coerce").sum())
    india_households = 246_740_228
    india_census_houses = 330_983_625
    assert abs(total - india_census_houses) / india_census_houses < 0.01
    assert total > india_households * 1.25, "Households column unexpectedly equals households"


def test_household_assets_are_household_counts() -> None:
    """The asset columns really are households, which is why only the denominator is wrong."""
    frame = d1()
    electricity = float(
        pd.to_numeric(frame["Housholds_with_Electric_Lighting"], errors="coerce").sum()
    )
    share = electricity / 246_740_228 * 100
    assert 66.0 <= share <= 68.5, f"electricity share {share:.2f}% outside published 67.2%"


# ---------------------------------------------------------------------------
# Housing abstract (D2)
# ---------------------------------------------------------------------------
def test_d2_is_percentage_normalised() -> None:
    frame = d2()
    total = pd.to_numeric(frame["Total Number of households"], errors="coerce")
    assert len(total) == 1908
    assert (total.round(6) == 100).all(), "D2 is not uniformly percentage-normalised"


def test_d2_condition_components_sum_to_100() -> None:
    frame = d2()
    comps = ["Total Number of Good", "Total Number of Livable", "Total Number of Dilapidated"]
    total = sum(pd.to_numeric(frame[c], errors="coerce") for c in comps)
    assert ((total >= 99.5) & (total <= 100.5)).all()


def test_every_d2_district_has_a_total_row() -> None:
    frame = d2()
    total = frame[frame["Rural/Urban"].astype(str).str.strip() == "Total"]
    assert len(total) == 640
    assert total["Area Name"].map(strip_area_prefix).nunique() == 634


# ---------------------------------------------------------------------------
# Slum dataset (D4) and artefacts
# ---------------------------------------------------------------------------
def test_slum_dataset_is_state_level_and_incomplete() -> None:
    frame = d4()
    assert len(frame) == 32, "expected 31 states/UTs plus an India row"
    assert "District" not in " ".join(map(str, frame.columns))
    states = {canonical_state(s) for s in frame["State"]}
    assert "INDIA" in states
    missing = {canonical_state(s) for s in d1()["State name"]} - states
    assert missing == {
        "DADRA AND NAGAR HAVELI",
        "DAMAN AND DIU",
        "LAKSHADWEEP",
        "MANIPUR",
    }, sorted(missing)


def test_metadata_artifacts_exist() -> None:
    expected = [
        "dataset_manifest.csv",
        "column_dictionary.csv",
        "data_quality_report.csv",
        "district_crosswalk.csv",
        "state_alias_map.csv",
        "target_candidates.csv",
        "audit_findings.json",
        "target_feasibility.json",
        "geography_findings.json",
        "denominator_resolution.json",
    ]
    missing = [name for name in expected if not (META / name).exists()]
    assert not missing, f"missing metadata artefacts: {missing}"


def test_crosswalk_covers_all_districts() -> None:
    frame = pd.read_csv(META / "district_crosswalk.csv")
    assert len(frame) == 640
    assert int(frame["d2_matched"].sum()) == 640
    # 12 districts share a name with a district in another state
    assert int(frame["d5_match_reliable"].sum()) == 628


def main() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures: list[tuple[str, BaseException]] = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except BaseException as exc:  # noqa: BLE001 - report every failure, keep going
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
