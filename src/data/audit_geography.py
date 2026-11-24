"""Phase 1 geographic reconciliation and target validation.

Answers three questions the audit report must be able to defend:

1. Do D1 (district census) and D2 (HLPCA housing) describe the same districts?
2. Can the optional PCA extracts (D5-D9) be joined to districts at all, and on
   what key?
3. Does the target computed from D1 reproduce the independent dilapidated share
   in D2, and if it differs, by how much and where?

Outputs:
    datasets/metadata/state_alias_map.csv
    datasets/metadata/district_crosswalk.csv
    reports/dataset_report/geographic_matching_report.txt
    reports/dataset_report/target_validation_report.txt
    datasets/metadata/geography_findings.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from naming import (  # noqa: E402
    OBSERVED_STATE_ALIASES,
    canonical_district,
    canonical_key,
    canonical_state,
    normalise_name,
    strip_area_prefix,
)

RAW = ROOT / "datasets" / "raw"
META = ROOT / "datasets" / "metadata"
REP = ROOT / "reports" / "dataset_report"

NUM = "Condition_of_occupied_census_houses_Dilapidated_Households"
DEN = "Households"


def load() -> dict[str, pd.DataFrame]:
    return {
        "D1": pd.read_csv(RAW / "india_census" / "india-districts-census-2011.csv", encoding="utf-8-sig"),
        "D2": pd.read_csv(RAW / "india_census" / "india_census_housing-hlpca-full.csv", encoding="utf-8-sig"),
        "D5": pd.read_csv(RAW / "india_pca" / "census_2011.csv", encoding="utf-8-sig"),
        "D7": pd.read_csv(RAW / "india_pca" / "education_2011.csv", encoding="utf-8-sig"),
        "GEO": pd.read_csv(RAW / "india_pca" / "geography.csv", encoding="utf-8-sig"),
    }


def main() -> int:
    f = load()
    d1, d2, d5, d7, geo = f["D1"], f["D2"], f["D5"], f["D7"], f["GEO"]
    out: dict = {}

    # ------------------------------------------------------------------
    # D1 base table
    # ------------------------------------------------------------------
    d1 = d1.copy()
    d1["key"] = [canonical_key(s, d) for s, d in zip(d1["State name"], d1["District name"])]
    d1["num"] = pd.to_numeric(d1[NUM], errors="coerce")
    d1["den"] = pd.to_numeric(d1[DEN], errors="coerce")
    d1["target_deprivation_rate"] = d1["num"] / d1["den"] * 100

    # ------------------------------------------------------------------
    # D2 district totals
    # ------------------------------------------------------------------
    d2 = d2.copy()
    d2["Rural/Urban"] = d2["Rural/Urban"].astype(str).str.strip()
    d2["area_clean"] = d2["Area Name"].map(strip_area_prefix)
    d2["key"] = [canonical_key(s, a) for s, a in zip(d2["State Name"], d2["area_clean"])]
    d2_total = d2[d2["Rural/Urban"] == "Total"].copy()

    rows_per_key = d2.groupby("key")["Rural/Urban"].apply(lambda s: tuple(sorted(s))).to_dict()
    out["d2_rows_per_key"] = {
        "n_keys": len(rows_per_key),
        "patterns": {
            "|".join(k): sum(1 for v in rows_per_key.values() if v == k)
            for k in {v for v in rows_per_key.values()}
        },
        "keys_missing_total": sorted(k for k, v in rows_per_key.items() if "Total" not in v),
        "keys_not_three_rows": sorted(k for k, v in rows_per_key.items() if len(v) != 3),
    }

    # D2 tail structure: the extract carries trailing columns whose header labels
    # are duplicated / unnamed. Quantify before trusting any of them.
    out["d2_column_structure"] = {
        "n_columns": int(d2.shape[1]),
        "duplicate_labels": sorted(
            {str(c) for c in d2.columns if list(d2.columns).count(c) > 1}
        ),
        "unnamed_columns": [str(c) for c in d2.columns if str(c).startswith("Unnamed")],
        "tail_columns": [str(c) for c in d2.columns[145:]],
        "tail_sample_values": d2.iloc[0, 145:].tolist(),
    }

    # ------------------------------------------------------------------
    # D5 district universe (aggregate Rural + Urban)
    # ------------------------------------------------------------------
    d5d = d5[d5["LEVEL"].astype(str).str.strip() == "DISTRICT"].copy()
    d5d["district_norm"] = d5d["NAME"].map(canonical_district)
    # Every district should contribute exactly two rows (Rural, Urban). A name
    # with 4 rows would mean two same-named districts from different states are
    # both present; a name with 1 row would mean one of the pair is missing.
    rows_per_name = d5d.groupby("district_norm").size()
    ambiguous = sorted(rows_per_name[rows_per_name != 2].index.tolist())
    d5_agg = (
        d5d.groupby("district_norm")
        .agg(no_hh=("NO_HH", "sum"), tot_p=("TOT_P", "sum"), n_rows=("AREA", "size"))
        .reset_index()
    )
    out["d5_universe"] = {
        "n_district_rows": int(len(d5d)),
        "n_unique_district_names": int(d5d["district_norm"].nunique()),
        "names_with_nonstandard_row_counts": ambiguous,
        "rows_per_name_distribution": {
            str(k): int(v) for k, v in rows_per_name.value_counts().sort_index().items()
        },
        "areas": d5d["AREA"].astype(str).str.strip().value_counts().to_dict(),
        "districts_with_single_area": d5d.groupby("district_norm")["AREA"]
        .nunique()
        .loc[lambda s: s < 2]
        .index.tolist(),
        "has_state_column": bool({"STATE", "State", "state"} & set(d5.columns)),
        "has_district_code": bool({"DISTRICT", "DIST_CODE", "dist_code"} & set(d5.columns)),
    }

    # ------------------------------------------------------------------
    # D7 level encoding (education): state_code + dist_code present?
    # ------------------------------------------------------------------
    d7 = d7.copy()
    d7["is_state_row"] = pd.to_numeric(d7["dist_code"], errors="coerce").fillna(0).eq(0)
    out["d7_universe"] = {
        "columns_present": [c for c in ["state_code", "dist_code", "area_name"] if c in d7.columns],
        "n_rows": int(len(d7)),
        "n_state_rows": int(d7["is_state_row"].sum()),
        "n_district_rows": int((~d7["is_state_row"]).sum()),
        "n_states": int(d7.loc[d7["is_state_row"], "state_code"].nunique()),
        "n_district_codes": int(d7.loc[~d7["is_state_row"], "dist_code"].nunique()),
    }

    # ------------------------------------------------------------------
    # geography.csv reference
    # ------------------------------------------------------------------
    geo = geo.copy()
    geo["level"] = geo["geo_level"].astype(str).str.strip()
    geo_state = geo[geo["level"] == "state"][["geo_code", "name"]].rename(
        columns={"geo_code": "state_code", "name": "state_name"}
    )
    geo_dist = geo[geo["level"] == "district"].copy()
    geo_dist["state_name"] = geo_dist["parent_code"].map(dict(zip(geo_state["state_code"], geo_state["state_name"])))
    geo_dist["key"] = [canonical_key(s, n) for s, n in zip(geo_dist["state_name"], geo_dist["name"])]
    out["geography_reference"] = {
        "n_states": int(len(geo_state)),
        "n_districts": int(len(geo_dist)),
        "years": sorted(geo["year"].dropna().astype(str).unique().tolist()),
        "district_name_duplicates": int(geo_dist["name"].map(normalise_name).duplicated().sum()),
        "state_codes": int(geo_state["state_code"].nunique()),
    }

    # ------------------------------------------------------------------
    # Crosswalk: D1 is the spine
    # ------------------------------------------------------------------
    d2_cols = d2_total.drop_duplicates("key")[
        ["key", "Total Number of Dilapidated", "Total Number of Good", "Total Number of Livable"]
    ].copy()
    d2_cols = d2_cols.rename(
        columns={
            "Total Number of Dilapidated": "d2_dilapidated_pct",
            "Total Number of Good": "d2_good_pct",
            "Total Number of Livable": "d2_livable_pct",
        }
    )

    xw = d1[["State name", "District name", "District code", "key", "num", "den", "target_deprivation_rate"]].copy()
    xw = xw.merge(d2_cols, on="key", how="left")
    xw = xw.merge(d5_agg.rename(columns={"district_norm": "d5_name"}), left_on=xw["District name"].map(canonical_district), right_on="d5_name", how="left")
    xw = xw.merge(
        geo_dist[["key", "geo_code", "square_kms"]].rename(
            columns={"geo_code": "geography_district_id", "square_kms": "geography_square_kms"}
        ),
        on="key",
        how="left",
    )
    xw["d2_matched"] = xw["d2_dilapidated_pct"].notna()
    xw["d5_matched"] = xw["d5_name"].notna()
    # D5 has no state/code column, so a name shared by two states cannot be
    # resolved and the household figure is not attributable to a single district.
    _name_freq = xw["District name"].map(canonical_district).value_counts()
    xw["d5_name_shared_in_d1"] = (
        xw["District name"].map(canonical_district).map(_name_freq).fillna(0) > 1
    )
    xw["d5_match_reliable"] = xw["d5_matched"] & ~xw["d5_name_shared_in_d1"]
    xw["geography_matched"] = xw["geography_district_id"].notna()
    xw["d2_minus_target_pp"] = pd.to_numeric(xw["d2_dilapidated_pct"], errors="coerce") - xw["target_deprivation_rate"]

    out["crosswalk"] = {
        "n_d1_districts": int(len(xw)),
        "d2_matched": int(xw["d2_matched"].sum()),
        "d2_unmatched": xw.loc[~xw["d2_matched"], ["State name", "District name"]].astype(str).values.tolist(),
        "d5_matched": int(xw["d5_matched"].sum()),
        "d5_matched_reliable": int(xw["d5_match_reliable"].sum()),
        "d5_unmatched": xw.loc[~xw["d5_matched"], ["State name", "District name"]].astype(str).values.tolist(),
        "geography_matched": int(xw["geography_matched"].sum()),
        "geography_unmatched": xw.loc[~xw["geography_matched"], ["State name", "District name"]]
        .astype(str)
        .values.tolist(),
        "d2_keys_extra": sorted(set(d2_total["key"]) - set(d1["key"])),
    }

    # ------------------------------------------------------------------
    # Target validation D1 vs D2, overall and per state
    # ------------------------------------------------------------------
    tv = xw.dropna(subset=["d2_dilapidated_pct"]).copy()
    tv["signed_diff"] = tv["d2_minus_target_pp"]
    tv["abs_diff"] = tv["signed_diff"].abs()
    per_state = (
        tv.groupby(tv["State name"].map(canonical_state))
        .agg(
            n=("abs_diff", "size"),
            median_abs_diff_pp=("abs_diff", "median"),
            mean_signed_diff_pp=("signed_diff", "mean"),
            max_abs_diff_pp=("abs_diff", "max"),
            target_mean=("target_deprivation_rate", "mean"),
            d2_mean=("d2_dilapidated_pct", "mean"),
        )
        .sort_values("median_abs_diff_pp", ascending=False)
        .round(3)
    )
    out["target_validation"] = {
        "n": int(len(tv)),
        "mean_signed_diff_pp": float(tv["signed_diff"].mean()),
        "median_signed_diff_pp": float(tv["signed_diff"].median()),
        "median_abs_diff_pp": float(tv["abs_diff"].median()),
        "max_abs_diff_pp": float(tv["abs_diff"].max()),
        "share_within_0_5pp": float((tv["abs_diff"] <= 0.5).mean()),
        "share_within_1pp": float((tv["abs_diff"] <= 1.0).mean()),
        "corr": float(tv["target_deprivation_rate"].corr(tv["d2_dilapidated_pct"])),
        "spearman": float(tv["target_deprivation_rate"].corr(tv["d2_dilapidated_pct"], method="spearman")),
        "ratio_target_over_d2_median": float((tv["target_deprivation_rate"] / tv["d2_dilapidated_pct"]).median()),
        "per_state": per_state.reset_index().rename(columns={"State name": "state"}).to_dict(orient="records"),
    }

    # Is the gap explained by the dilapidated *residence-only* share in D2?
    resid_candidates = [c for c in d2.columns if str(c).startswith("Residence_Dilapidated")]
    alt_block: dict = {"candidates": [str(c) for c in resid_candidates]}
    for candidate in resid_candidates:
        alt = d2_total.drop_duplicates("key")[["key", candidate]].copy()
        alt["alt"] = pd.to_numeric(alt[candidate], errors="coerce")
        merged = tv[["key", "target_deprivation_rate"]].merge(alt[["key", "alt"]], on="key", how="inner").dropna()
        if len(merged) > 10:
            alt_block[candidate] = {
                "n": int(len(merged)),
                "median_abs_diff_pp": float((merged["target_deprivation_rate"] - merged["alt"]).abs().median()),
                "corr": float(merged["target_deprivation_rate"].corr(merged["alt"])),
            }
    out["d2_alternative_columns"] = alt_block

    # D5 denominator check: does SUM(NO_HH) match D1 Households?
    d5check = xw.dropna(subset=["no_hh"])
    if len(d5check):
        diff = d5check["no_hh"] - d5check["den"]
        rel = diff / d5check["den"] * 100
        out["d5_denominator_check"] = {
            "n": int(len(d5check)),
            "exact_matches": int((diff == 0).sum()),
            "median_rel_diff_pct": float(rel.median()),
            "p95_abs_rel_diff_pct": float(rel.abs().quantile(0.95)),
            "max_abs_diff": float(diff.abs().max()),
        }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    pd.DataFrame(
        [
            {"alias": k, "canonical": v, "scope": "state", "source": "observed in this workspace"}
            for k, v in OBSERVED_STATE_ALIASES.items()
        ]
    ).to_csv(META / "state_alias_map.csv", index=False)

    xw[
        [
            "State name", "District name", "District code", "key", "num", "den",
            "target_deprivation_rate", "d2_dilapidated_pct", "d2_minus_target_pp",
            "d2_matched", "no_hh", "d5_matched", "d5_match_reliable", "geography_district_id",
            "geography_matched", "geography_square_kms",
        ]
    ].rename(
        columns={
            "State name": "state_name",
            "District name": "district_name",
            "District code": "district_code",
            "num": "dilapidated_households",
            "den": "total_households",
        }
    ).to_csv(META / "district_crosswalk.csv", index=False)

    (META / "geography_findings.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    geo_lines = ["PHASE 1 GEOGRAPHIC MATCHING REPORT", "=" * 70, ""]
    geo_lines.append(f"D1 district rows ................ {out['crosswalk']['n_d1_districts']}")
    geo_lines.append(f"D2 matched (state+district) ...... {out['crosswalk']['d2_matched']}")
    geo_lines.append(f"D2 unmatched ..................... {len(out['crosswalk']['d2_unmatched'])}")
    geo_lines.append(f"D5 matched (district name only) .. {out['crosswalk']['d5_matched']}")
    geo_lines.append(f"geography.csv matched ............ {out['crosswalk']['geography_matched']}")
    geo_lines.append("")
    geo_lines.append("D2 rows per district key:")
    geo_lines.append(f"  keys: {out['d2_rows_per_key']['n_keys']}, patterns: {out['d2_rows_per_key']['patterns']}")
    geo_lines.append(f"  keys missing a Total row: {out['d2_rows_per_key']['keys_missing_total']}")
    geo_lines.append(f"  keys without exactly 3 rows: {out['d2_rows_per_key']['keys_not_three_rows']}")
    geo_lines.append(f"  D2 keys absent from D1: {out['crosswalk']['d2_keys_extra']}")
    geo_lines.append("")
    geo_lines.append("D5 (PCA 2011) universe:")
    for k, v in out["d5_universe"].items():
        geo_lines.append(f"  {k}: {v}")
    geo_lines.append("")
    geo_lines.append(f"D1 districts not matched inside D5: {out['crosswalk']['d5_unmatched']}")
    geo_lines.append(f"D1 districts not matched in geography.csv: {out['crosswalk']['geography_unmatched']}")
    geo_lines.append("")
    geo_lines.append("D5 denominator cross-check vs D1 Households:")
    geo_lines.append(f"  {out.get('d5_denominator_check')}")
    (REP / "geographic_matching_report.txt").write_text("\n".join(geo_lines), encoding="utf-8")

    tv_lines = ["PHASE 1 TARGET VALIDATION REPORT", "=" * 70, ""]
    tv_lines.append("Target under test: target_deprivation_rate = dilapidated households / total households * 100")
    tv_lines.append("Independent comparison: D2 (HLPCA) district 'Total' dilapidated share.")
    tv_lines.append("")
    for k, v in out["target_validation"].items():
        if k != "per_state":
            tv_lines.append(f"{k}: {v}")
    tv_lines.append("")
    tv_lines.append("Worst states by median absolute difference:")
    for row in out["target_validation"]["per_state"][:10]:
        tv_lines.append(
            f"  {row['state']:<32} n={row['n']:>3} median|d|={row['median_abs_diff_pp']:>6.3f} pp "
            f"mean signed={row['mean_signed_diff_pp']:>7.3f} pp max={row['max_abs_diff_pp']:>6.3f} pp"
        )
    tv_lines.append("")
    tv_lines.append(f"D2 alternative dilapidated columns tested: {out['d2_alternative_columns']}")
    (REP / "target_validation_report.txt").write_text("\n".join(tv_lines), encoding="utf-8")

    print("\n".join(geo_lines))
    print()
    print("\n".join(tv_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
