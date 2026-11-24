"""Phase 1 target-feasibility test.

The proposed target is

    target_deprivation_rate = dilapidated households / total households * 100

with numerator ``Condition_of_occupied_census_houses_Dilapidated_Households``
and denominator ``Households`` from D1. This script tests whether those two
columns describe the same population, by benchmarking both against the
independent PCA household counts in D5 and the Census-computed dilapidated
share in D2.

Outputs:
    reports/dataset_report/target_feasibility_report.txt
    datasets/metadata/target_feasibility.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from naming import canonical_district, canonical_key, canonical_state, strip_area_prefix  # noqa: E402

RAW = ROOT / "datasets" / "raw"
META = ROOT / "datasets" / "metadata"
REP = ROOT / "reports" / "dataset_report"

NUM = "Condition_of_occupied_census_houses_Dilapidated_Households"
DEN = "Households"

# India, Census 2011 published aggregates (used only as an external sanity anchor).
REFERENCE_TOTAL_HOUSEHOLDS = 246_740_228
REFERENCE_TOTAL_CENSUS_HOUSES = 330_983_625


def main() -> int:
    d1 = pd.read_csv(RAW / "india_census" / "india-districts-census-2011.csv", encoding="utf-8-sig")
    d2 = pd.read_csv(RAW / "india_census" / "india_census_housing-hlpca-full.csv", encoding="utf-8-sig")
    d5 = pd.read_csv(RAW / "india_pca" / "census_2011.csv", encoding="utf-8-sig")

    out: dict = {}

    d1["key"] = [canonical_key(s, d) for s, d in zip(d1["State name"], d1["District name"])]
    d1["name"] = d1["District name"].map(canonical_district)
    d1["num"] = pd.to_numeric(d1[NUM], errors="coerce")
    d1["den"] = pd.to_numeric(d1[DEN], errors="coerce")
    d1["pop"] = pd.to_numeric(d1["Population"], errors="coerce")

    # ------------------------------------------------------------------
    # D5: district totals from the PCA extract (Rural + Urban)
    # ------------------------------------------------------------------
    d5d = d5[d5["LEVEL"].astype(str).str.strip() == "DISTRICT"].copy()
    d5d["name"] = d5d["NAME"].map(canonical_district)
    rows_per_name = d5d.groupby("name").size().rename("n_rows")
    d5_agg = d5d.groupby("name").agg(no_hh=("NO_HH", "sum"), tot_p=("TOT_P", "sum")).reset_index()
    d5_agg["n_rows"] = d5_agg["name"].map(rows_per_name)

    shared = d1.groupby("name")["key"].nunique()
    shared_names = sorted(shared[shared > 1].index.tolist())
    out["shared_district_names"] = {
        "names": shared_names,
        "n_d1_districts_involved": int(shared[shared > 1].sum()),
        "d5_rows_per_shared_name": {
            n: int(rows_per_name.get(n, 0)) for n in shared_names
        },
        "reading": (
            "A district name carried by more than one state cannot be joined by name. "
            "D5 has no state or district code column, so these districts need the "
            "count-based target to be built without D5, or a coded crosswalk."
        ),
    }

    xw = d1.merge(d5_agg, on="name", how="left")
    xw["ambiguous_name"] = xw["key"].map(shared) > 1

    # ------------------------------------------------------------------
    # 1. Is D1 'Households' the household universe?
    # ------------------------------------------------------------------
    out["national_aggregates"] = {
        "d1_households_sum": int(d1["den"].sum()),
        "d1_dilapidated_sum": int(d1["num"].sum()),
        "d1_population_sum": int(d1["pop"].sum()),
        "d5_no_hh_sum": int(d5_agg["no_hh"].sum()),
        "d5_tot_p_sum": int(d5_agg["tot_p"].sum()),
        "reference_total_households_2011": REFERENCE_TOTAL_HOUSEHOLDS,
        "reference_total_census_houses_2011": REFERENCE_TOTAL_CENSUS_HOUSES,
        "d1_households_over_reference_households": float(
            d1["den"].sum() / REFERENCE_TOTAL_HOUSEHOLDS
        ),
        "d1_households_over_d5_no_hh": float(d1["den"].sum() / d5_agg["no_hh"].sum()),
        "d1_population_over_d5_tot_p": float(d1["pop"].sum() / d5_agg["tot_p"].sum()),
    }

    valid = xw.dropna(subset=["no_hh", "den", "pop", "tot_p"]).copy()
    valid["hh_ratio"] = valid["den"] / valid["no_hh"]
    valid["pop_ratio"] = valid["pop"] / valid["tot_p"]
    out["denominator_test"] = {
        "n_districts": int(len(valid)),
        "household_ratio_median": float(valid["hh_ratio"].median()),
        "household_ratio_p05": float(valid["hh_ratio"].quantile(0.05)),
        "household_ratio_p95": float(valid["hh_ratio"].quantile(0.95)),
        "household_ratio_min": float(valid["hh_ratio"].min()),
        "household_ratio_max": float(valid["hh_ratio"].max()),
        "share_ratio_within_2pct_of_1": float((valid["hh_ratio"].sub(1).abs() <= 0.02).mean()),
        "population_ratio_median": float(valid["pop_ratio"].median()),
        "population_ratio_p05": float(valid["pop_ratio"].quantile(0.05)),
        "population_ratio_p95": float(valid["pop_ratio"].quantile(0.95)),
        "share_population_ratio_within_0_1pct": float((valid["pop_ratio"].sub(1).abs() <= 0.001).mean()),
        "corr_households_vs_no_hh": float(valid["den"].corr(valid["no_hh"])),
    }

    # ------------------------------------------------------------------
    # 2. Does D2's dilapidated share imply D1's dilapidated count?
    # ------------------------------------------------------------------
    d2["Rural/Urban"] = d2["Rural/Urban"].astype(str).str.strip()
    d2["area_clean"] = d2["Area Name"].map(strip_area_prefix)
    d2["key"] = [canonical_key(s, a) for s, a in zip(d2["State Name"], d2["area_clean"])]
    d2t = d2[d2["Rural/Urban"] == "Total"].drop_duplicates("key")

    cols = ["Total Number of Dilapidated", "Residence_Dilapidated"]
    cols = [c for c in cols if c in d2t.columns]
    implied = d2t[["key"] + cols].copy()
    for c in cols:
        implied[c] = pd.to_numeric(implied[c], errors="coerce")
    m = d1[["key", "num"]].merge(implied, on="key", how="inner")
    m["name"] = m["key"].map(lambda k: k.split("||")[1])
    m = m.merge(d5_agg[["name", "no_hh"]], on="name", how="left")
    m["ambiguous"] = m["name"].map(shared).fillna(0) > 1
    m = m[~m["ambiguous"]]

    implied_block = {}
    for c in cols:
        est = m[c] / 100 * m["no_hh"]
        rel = (est - m["num"]) / m["num"] * 100
        implied_block[c] = {
            "n": int(est.notna().sum()),
            "median_abs_rel_diff_pct": float(rel.abs().median()),
            "median_signed_rel_diff_pct": float(rel.median()),
            "share_within_2pct": float((rel.abs() <= 2).mean()),
            "corr": float(est.corr(m["num"])),
            "implied_national_total": float(est.sum()),
        }
    out["implied_dilapidated_count"] = {
        "d1_dilapidated_national": int(d1["num"].sum()),
        "candidates": implied_block,
    }

    # ------------------------------------------------------------------
    # 3. Candidate target definitions
    # ------------------------------------------------------------------
    cand = d1[["key", "num", "den"]].copy()
    cand["name"] = cand["key"].map(lambda k: k.split("||")[1])
    cand = cand.merge(d5_agg[["name", "no_hh"]], on="name", how="left")
    cand["ambiguous"] = cand["name"].map(shared).fillna(0) > 1
    # A name shared by two states takes the same D5 household count for both
    # districts, so the count-based target is undefined for those rows.
    cand.loc[cand["ambiguous"], "no_hh"] = np.nan
    cand = cand.merge(implied, on="key", how="left")

    cand["T1_d1_counts"] = cand["num"] / cand["den"] * 100
    if "Total Number of Dilapidated" in cand.columns:
        cand["T2_housing_abstract_pct"] = pd.to_numeric(
            cand["Total Number of Dilapidated"], errors="coerce"
        )
    else:
        cand["T2_housing_abstract_pct"] = np.nan
    cand["T3_d1_num_over_pca_households"] = np.where(
        cand["no_hh"] > 0, cand["num"] / cand["no_hh"] * 100, np.nan
    )

    def describe(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors="coerce").dropna()
        return {
            "n": int(len(s)),
            "min": round(float(s.min()), 3),
            "p05": round(float(s.quantile(0.05)), 3),
            "median": round(float(s.median()), 3),
            "mean": round(float(s.mean()), 3),
            "p95": round(float(s.quantile(0.95)), 3),
            "max": round(float(s.max()), 3),
            "sd": round(float(s.std()), 3),
            "national_pooled": None,
        }

    target_defs = {}
    for col in ["T1_d1_counts", "T2_housing_abstract_pct", "T3_d1_num_over_pca_households"]:
        target_defs[col] = describe(cand[col])
    target_defs["T1_d1_counts"]["national_pooled"] = round(
        float(cand["num"].sum() / cand["den"].sum() * 100), 3
    )
    target_defs["T3_d1_num_over_pca_households"]["national_pooled"] = round(
        float(cand.loc[~cand["ambiguous"], "num"].sum() / cand.loc[~cand["ambiguous"], "no_hh"].sum() * 100),
        3,
    )

    pairwise = {}
    for a in ["T1_d1_counts", "T2_housing_abstract_pct", "T3_d1_num_over_pca_households"]:
        for b in ["T1_d1_counts", "T2_housing_abstract_pct", "T3_d1_num_over_pca_households"]:
            if a < b:
                sub = cand[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
                pairwise[f"{a} vs {b}"] = {
                    "n": int(len(sub)),
                    "pearson": round(float(sub[a].corr(sub[b])), 4),
                    "spearman": round(float(sub[a].corr(sub[b], method="spearman")), 4),
                    "median_abs_diff_pp": round(float((sub[a] - sub[b]).abs().median()), 3),
                }

    out["candidate_targets"] = target_defs
    out["candidate_target_pairwise"] = pairwise
    out["ambiguous_districts_excluded_from_count_join"] = int(cand["ambiguous"].sum())

    # ------------------------------------------------------------------
    # 4. Optional offline extract integrity (D7/D8/D9 duplicate diagnosis)
    # ------------------------------------------------------------------
    dup_report = {}
    for ds, fname, keys in [
        ("D7", "education_2011.csv", ["state_code", "dist_code", "area_name", "total", "age_group"]),
        ("D8", "marital_status_2011.csv", ["state_code", "dist_code", "area_name", "total", "age_group"]),
        ("D9", "age_group_2011.csv", None),
    ]:
        df = pd.read_csv(RAW / "india_pca" / fname, encoding="utf-8-sig")
        exact = int(df.duplicated().sum())
        entry = {"rows": int(len(df)), "exact_duplicates": exact}
        if keys and all(k in df.columns for k in keys):
            key_dup = int(df.duplicated(subset=keys).sum())
            entry["duplicates_on_natural_key"] = key_dup
            entry["natural_key"] = keys
            entry["verdict"] = (
                "duplicates are redundant repeats" if key_dup == exact else
                "duplicates repeat distinct records (multiple measures per key)"
            )
        dup_report[ds] = entry
    out["optional_extract_duplicates"] = dup_report

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    (META / "target_feasibility.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    cand.to_csv(META / "target_candidates.csv", index=False)

    L = ["PHASE 1 TARGET FEASIBILITY REPORT", "=" * 70, ""]
    L.append("1. NATIONAL AGGREGATES")
    for k, v in out["national_aggregates"].items():
        L.append(f"   {k}: {v:,}" if isinstance(v, int) else f"   {k}: {v}")
    L.append("")
    L.append("2. IS D1 'Households' THE HOUSEHOLD UNIVERSE?  (tested vs D5 PCA NO_HH)")
    for k, v in out["denominator_test"].items():
        L.append(f"   {k}: {v}")
    L.append("")
    L.append("3. DOES D2's SHARE IMPLY D1's DILAPIDATED COUNT?")
    L.append(f"   D1 dilapidated national total: {out['implied_dilapidated_count']['d1_dilapidated_national']:,}")
    for c, v in implied_block.items():
        L.append(f"   from {c}: {v}")
    L.append("")
    L.append("4. CANDIDATE TARGET DISTRIBUTIONS")
    for name, d in target_defs.items():
        L.append(f"   {name}: {d}")
    L.append("")
    L.append("5. PAIRWISE AGREEMENT")
    for k, v in pairwise.items():
        L.append(f"   {k}: {v}")
    L.append("")
    L.append("6. SHARED DISTRICT NAMES (block count-based joins by name)")
    L.append(f"   {shared_names}")
    L.append(f"   D5 rows per shared name: {out['shared_district_names']['d5_rows_per_shared_name']}")
    L.append("")
    L.append("7. OPTIONAL EXTRACT DUPLICATES")
    for k, v in dup_report.items():
        L.append(f"   {k}: {v}")
    (REP / "target_feasibility_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
