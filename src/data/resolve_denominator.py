"""Resolve the D1 denominator question and the remaining join gaps.

Phase 1 must state whether a valid household denominator exists. This script:

1. tests ``Rural_Households + Urban_Households`` in D1 as an alternative
   household universe, against D5 PCA household counts;
2. tests whether D1's household-asset columns are genuine household counts by
   benchmarking a few national totals against published 2011 shares;
3. resolves the six district names that occur in two states by matching
   population, so we know whether D5 actually holds 634 or 640 districts;
4. diagnoses the geography.csv vintage mismatch (Telangana, New Delhi).

Writes reports/dataset_report/denominator_resolution_report.txt and
datasets/metadata/denominator_resolution.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from naming import canonical_district, canonical_key, canonical_state  # noqa: E402

RAW = ROOT / "datasets" / "raw"
META = ROOT / "datasets" / "metadata"
REP = ROOT / "reports" / "dataset_report"

NUM = "Condition_of_occupied_census_houses_Dilapidated_Households"


def main() -> int:
    d1 = pd.read_csv(RAW / "india_census" / "india-districts-census-2011.csv", encoding="utf-8-sig")
    d5 = pd.read_csv(RAW / "india_pca" / "census_2011.csv", encoding="utf-8-sig")
    geo = pd.read_csv(RAW / "india_pca" / "geography.csv", encoding="utf-8-sig")

    out: dict = {}
    d1["key"] = [canonical_key(s, d) for s, d in zip(d1["State name"], d1["District name"])]
    d1["name"] = d1["District name"].map(canonical_district)

    num = pd.to_numeric(d1[NUM], errors="coerce")
    hh_labeled = pd.to_numeric(d1["Households"], errors="coerce")
    rural = pd.to_numeric(d1["Rural_Households"], errors="coerce")
    urban = pd.to_numeric(d1["Urban_Households"], errors="coerce")
    den_alt = rural + urban

    d5d = d5[d5["LEVEL"].astype(str).str.strip() == "DISTRICT"].copy()
    d5d["name"] = d5d["NAME"].map(canonical_district)
    d5_agg = d5d.groupby("name").agg(no_hh=("NO_HH", "sum"), tot_p=("TOT_P", "sum")).reset_index()

    m = d1[["key", "name", "Population"]].copy()
    m["num"] = num
    m["den_labeled"] = hh_labeled
    m["den_alt"] = den_alt
    m = m.merge(d5_agg, on="name", how="left")

    # ------------------------------------------------------------------
    # 1. Alternative denominator
    # ------------------------------------------------------------------
    out["alternative_denominator"] = {
        "d1_households_labeled_sum": int(hh_labeled.sum()),
        "d1_rural_plus_urban_sum": int(den_alt.sum()),
        "d1_rural_sum": int(rural.sum()),
        "d1_urban_sum": int(urban.sum()),
        "d5_no_hh_sum": int(d5_agg["no_hh"].sum()),
        "published_total_households_2011": 246_740_228,
        "alt_over_d5": float(den_alt.sum() / d5_agg["no_hh"].sum()),
        "alt_over_published": float(den_alt.sum() / 246_740_228),
        "exact_equals_d5_count": int((den_alt == m["no_hh"]).sum()),
        "within_0_5pct_of_d5": int(((den_alt - m["no_hh"]).abs() / m["no_hh"] <= 0.005).sum()),
        "n_districts": int(len(m)),
        "corr_alt_vs_d5": float(den_alt.corr(m["no_hh"])),
        "median_abs_pct_diff_vs_d5": float((((den_alt - m["no_hh"]).abs() / m["no_hh"]) * 100).median()),
    }

    # ------------------------------------------------------------------
    # 2. Are D1 household-asset columns household counts?
    # ------------------------------------------------------------------
    probe = {
        "LPG_or_PNG_Households": 0.58,
        "Housholds_with_Electric_Lighting": 0.67,
        "Households_with_Television": 0.47,
        "Having_latrine_facility_within_the_premises_Total_Households": 0.47,
    }
    probe_block = {}
    for col, published_share in probe.items():
        if col not in d1.columns:
            continue
        total = float(pd.to_numeric(d1[col], errors="coerce").sum())
        probe_block[col] = {
            "national_sum": int(total),
            "share_of_published_households": round(total / 246_740_228 * 100, 2),
            "published_2011_share_pct": round(published_share * 100, 1),
            "share_of_d1_households_labeled": round(total / float(hh_labeled.sum()) * 100, 2),
            "share_of_alt_denominator": round(total / float(den_alt.sum()) * 100, 2),
        }
    out["household_asset_probe"] = probe_block

    # ------------------------------------------------------------------
    # 3. Shared district names: which state does D5 actually hold?
    # ------------------------------------------------------------------
    shared_names = ["AURANGABAD", "BIJAPUR", "BILASPUR", "HAMIRPUR", "PRATAPGARH", "RAIGARH"]
    resolved = []
    for name in shared_names:
        rows = m[m["name"] == name].copy()
        hit = rows[rows["Population"] == rows["tot_p"]]
        resolved.append(
            {
                "district_name": name,
                "d1_candidates": [
                    {
                        "state": str(r["key"]).split("||")[0],
                        "population": int(r["Population"]),
                        "no_hh": None if pd.isna(r["no_hh"]) else int(r["no_hh"]),
                    }
                    for _, r in rows.iterrows()
                ],
                "d5_population": None if rows["tot_p"].isna().all() else int(rows["tot_p"].dropna().iloc[0]),
                "matched_district": None if hit.empty else str(hit.iloc[0]["key"]),
                "unmatched_d1_districts": [str(r["key"]) for _, r in rows.iterrows() if pd.isna(r["no_hh"]) or r["Population"] != r["tot_p"]],
            }
        )
    out["shared_name_resolution"] = resolved

    # ------------------------------------------------------------------
    # 4. geography.csv vintage
    # ------------------------------------------------------------------
    geo["level"] = geo["geo_level"].astype(str).str.strip()
    gs = geo[geo["level"] == "state"][["geo_code", "name"]]
    out["geography_vintage"] = {
        "years": sorted(geo["year"].dropna().astype(str).unique().tolist()),
        "n_states": int(len(gs)),
        "state_names": sorted(gs["name"].astype(str).tolist()),
        "has_telangana": bool(gs["name"].astype(str).str.upper().str.contains("TELANGANA").any()),
        "delhi_entry": gs[gs["name"].astype(str).str.upper().str.contains("DELHI")].astype(str).to_dict(orient="records"),
        "district_rows_named_new_delhi": geo[
            (geo["level"] == "district") & (geo["name"].astype(str).str.upper() == "NEW DELHI")
        ].astype(str).to_dict(orient="records"),
    }

    # ------------------------------------------------------------------
    # 5. Final candidate target with the alternative denominator
    # ------------------------------------------------------------------
    t4 = np.where(den_alt > 0, num / den_alt * 100, np.nan)
    d1["T4_num_over_rural_plus_urban"] = t4
    d2 = pd.read_csv(RAW / "india_census" / "india_census_housing-hlpca-full.csv", encoding="utf-8-sig")
    d2t = d2[d2["Rural/Urban"].astype(str).str.strip() == "Total"].drop_duplicates("key" if "key" in d2 else None)
    # rebuild the D2 key without relying on a pre-existing column
    from naming import strip_area_prefix
    d2t = d2[d2["Rural/Urban"].astype(str).str.strip() == "Total"].copy()
    d2t["key"] = [
        canonical_key(s, strip_area_prefix(a)) for s, a in zip(d2t["State Name"], d2t["Area Name"])
    ]
    d2t["d2_pct"] = pd.to_numeric(d2t["Total Number of Dilapidated"], errors="coerce")
    cmp = d1[["key", "T4_num_over_rural_plus_urban"]].merge(d2t[["key", "d2_pct"]], on="key")
    cmp = cmp.dropna()
    diff = cmp["T4_num_over_rural_plus_urban"] - cmp["d2_pct"]
    out["T4_validation"] = {
        "n": int(len(cmp)),
        "national_pooled_pct": round(float(num.sum() / den_alt.sum() * 100), 3),
        "median_abs_diff_pp": round(float(diff.abs().median()), 3),
        "share_within_0_5pp": round(float((diff.abs() <= 0.5).mean()), 3),
        "corr": round(float(cmp["T4_num_over_rural_plus_urban"].corr(cmp["d2_pct"])), 4),
        "distribution": {
            "min": round(float(cmp["T4_num_over_rural_plus_urban"].min()), 3),
            "median": round(float(cmp["T4_num_over_rural_plus_urban"].median()), 3),
            "mean": round(float(cmp["T4_num_over_rural_plus_urban"].mean()), 3),
            "max": round(float(cmp["T4_num_over_rural_plus_urban"].max()), 3),
            "sd": round(float(cmp["T4_num_over_rural_plus_urban"].std()), 3),
        },
    }

    (META / "denominator_resolution.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    L = ["PHASE 1 DENOMINATOR RESOLUTION REPORT", "=" * 70, "", "1. ALTERNATIVE DENOMINATOR (Rural_Households + Urban_Households)"]
    for k, v in out["alternative_denominator"].items():
        L.append(f"   {k}: {v:,}" if isinstance(v, int) and v > 1000 else f"   {k}: {v}")
    L.append("")
    L.append("2. HOUSEHOLD ASSET PROBE (national sums vs published 2011 shares)")
    for k, v in out["household_asset_probe"].items():
        L.append(f"   {k}: {v}")
    L.append("")
    L.append("3. SHARED DISTRICT NAME RESOLUTION VIA POPULATION")
    for r in resolved:
        L.append(f"   {r['district_name']}: d5_population={r['d5_population']} matched={r['matched_district']}")
        L.append(f"      unmatched: {r['unmatched_d1_districts']}")
    L.append("")
    L.append("4. GEOGRAPHY.csv VINTAGE")
    g = out["geography_vintage"]
    L.append(f"   years={g['years']} n_states={g['n_states']} has_telangana={g['has_telangana']}")
    L.append(f"   delhi entries: {g['delhi_entry']}")
    L.append(f"   New Delhi district rows: {g['district_rows_named_new_delhi']}")
    L.append("")
    L.append("5. T4 (dilapidated / Rural+Urban households) vs D2 share")
    for k, v in out["T4_validation"].items():
        L.append(f"   {k}: {v}")
    (REP / "denominator_resolution_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
