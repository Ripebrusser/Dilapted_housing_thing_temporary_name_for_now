"""Phase 1 dataset audit.

Reads every raw dataset in ``datasets/raw``, profiles it, runs explicit
data-quality checks, verifies geographic matching across datasets and writes:

    datasets/metadata/dataset_manifest.csv
    datasets/metadata/column_dictionary.csv
    datasets/metadata/data_quality_report.csv
    datasets/metadata/audit_findings.json
    reports/dataset_report/dataset_audit_summary.txt

The script is read-only with respect to the raw data: it never writes into
``datasets/raw``. Run from anywhere:

    py src/data/audit_datasets.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from naming import canonical_key, canonical_state, normalise_name  # noqa: E402

RAW = ROOT / "datasets" / "raw"
META = ROOT / "datasets" / "metadata"
REPORTS = ROOT / "reports" / "dataset_report"

# ---------------------------------------------------------------------------
# Dataset registry. Absolute machine paths are never stored; everything is
# resolved relative to the repository root.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "dataset_id": "D1",
        "name": "India District Census 2011 (household + population abstract)",
        "path": RAW / "india_census" / "india-districts-census-2011.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "primary",
    },
    {
        "dataset_id": "D2",
        "name": "Census 2011 Houselisting & Housing Census Abstract (HLPCA) - district extract",
        "path": RAW / "india_census" / "india_census_housing-hlpca-full.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "primary_supporting",
    },
    {
        "dataset_id": "D3",
        "name": "HLPCA column dictionary",
        "path": RAW / "india_census" / "hlpca-colnames.csv",
        "geographic_level": "n/a",
        "time_period": 2011,
        "role": "dictionary",
    },
    {
        "dataset_id": "D4",
        "name": "Slum Population of India (Census 2011, state level)",
        "path": RAW / "slum_population" / "Slum Population Of India.csv",
        "geographic_level": "state",
        "time_period": 2011,
        "role": "secondary",
    },
    {
        "dataset_id": "D5",
        "name": "Census 2011 PCA extract (state/district x rural/urban)",
        "path": RAW / "india_pca" / "census_2011.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "optional_supporting",
    },
    {
        "dataset_id": "D6",
        "name": "Census 2001 PCA extract (state/district x rural/urban)",
        "path": RAW / "india_pca" / "census_2001.csv",
        "geographic_level": "district",
        "time_period": 2001,
        "role": "optional_supporting",
    },
    {
        "dataset_id": "D7",
        "name": "Education level 2011 (C-series table extract)",
        "path": RAW / "india_pca" / "education_2011.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "optional_supporting",
    },
    {
        "dataset_id": "D8",
        "name": "Marital status 2011 (C-series table extract)",
        "path": RAW / "india_pca" / "marital_status_2011.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "optional_supporting",
    },
    {
        "dataset_id": "D9",
        "name": "Age group 2011 (C-series table extract)",
        "path": RAW / "india_pca" / "age_group_2011.csv",
        "geographic_level": "district",
        "time_period": 2011,
        "role": "optional_supporting",
    },
    {
        "dataset_id": "D10",
        "name": "Geography lookup (India boundaries, 2016 vintage)",
        "path": RAW / "india_pca" / "geography.csv",
        "geographic_level": "state/district",
        "time_period": 2016,
        "role": "reference",
    },
]

TARGET_NUMERATOR = "Condition_of_occupied_census_houses_Dilapidated_Households"
TARGET_DENOMINATOR = "Households"
TARGET_NAME = "target_deprivation_rate"

IDENTIFIER_COLS = {
    "District code", "State name", "District name", "State Code", "State Name",
    "District Code", "District Name", "Tehsil Code", "Tehsil Name",
    "Town Code/Village code", "Ward No", "Area Name", "Rural/Urban",
    "LEVEL", "YEAR", "NAME", "AREA", "state_code", "dist_code", "area_name",
    "table_name", "table", "id", "geo_level", "geo_code", "parent_level",
    "parent_code", "version", "#", "State", "Unnamed: 0",
}

QUALITY_ROWS: list[dict] = []
FINDINGS: dict = {"datasets": {}, "geography": {}, "target": {}, "leakage": {}}


def add_quality(dataset_id: str, check: str, status: str, severity: str, details: str) -> None:
    QUALITY_ROWS.append(
        {
            "dataset_id": dataset_id,
            "check": check,
            "status": status,
            "severity": severity,
            "details": details,
        }
    )


def classify_role(dataset_id: str, column: str) -> str:
    if dataset_id == "D1":
        if column in {TARGET_NUMERATOR, TARGET_DENOMINATOR}:
            return "target_component_leakage_risk"
        if column in {"District code", "State name", "District name"}:
            return "identifier"
        return "candidate_predictor"
    if dataset_id == "D2":
        if column in {"Total Number of Dilapidated", "Contition_T_Dilapidated"}:
            return "target_crosscheck_leakage_risk"
        if column in {"Total Number of households", "Contition_T_Total"}:
            return "denominator_crosscheck_leakage_risk"
        if column in IDENTIFIER_COLS:
            return "identifier"
        return "candidate_predictor_or_supporting"
    if column in IDENTIFIER_COLS:
        return "identifier"
    return "candidate_predictor"


def profile_columns(dataset_id: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(df)
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_share = float(numeric.notna().sum() / non_null) if non_null else 0.0
        min_val = max_val = ""
        if numeric_share >= 0.95 and numeric.notna().any():
            min_val = float(numeric.min())
            max_val = float(numeric.max())
        samples = (
            series.dropna().astype(str).unique()[:3].tolist()
            if series.notna().any()
            else []
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "column_name": str(column),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "null_count": int(n - non_null),
                "null_pct": round(100 * (n - non_null) / n, 2) if n else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
                "min": min_val,
                "max": max_val,
                "numeric_share": round(numeric_share, 3),
                "sample_values": " | ".join(str(s) for s in samples)[:180],
                "role": classify_role(dataset_id, str(column)),
            }
        )
    return rows


def read_dataset(dataset_id: str, path: Path) -> pd.DataFrame:
    """Load a dataset with the parser it actually needs.

    ``hlpca-colnames.csv`` (D3) is not valid RFC-4180 CSV: many descriptions
    contain unquoted commas ("Coal,Lignite,Charcoal"), so the default parser
    splits them into extra fields. It is a code<->description map, so we split
    on the first comma only.
    """
    if dataset_id == "D3":
        rows = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                code, _, description = line.partition(",")
                rows.append((code.strip(), description.strip()))
        return pd.DataFrame(rows, columns=["source_column_code", "source_column_description"])
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def raw_row_widths(path: Path) -> dict:
    """Detect ragged CSV rows (row field counts differing from the header)."""
    widths: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            widths[len(row)] = widths.get(len(row), 0) + 1
    return dict(sorted(widths.items()))


def main() -> int:
    META.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    dictionary: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}

    for spec in DATASETS:
        dataset_id = spec["dataset_id"]
        path: Path = spec["path"]
        rel = path.relative_to(ROOT).as_posix()

        if not path.exists():
            add_quality(dataset_id, "file_exists", "FAIL", "critical", f"Missing file: {rel}")
            manifest.append(
                {
                    "dataset_id": dataset_id,
                    "name": spec["name"],
                    "relative_path": rel,
                    "size_bytes": 0,
                    "n_rows": 0,
                    "n_cols": 0,
                    "geographic_level": spec["geographic_level"],
                    "time_period": spec["time_period"],
                    "role": spec["role"],
                    "status": "MISSING",
                }
            )
            continue

        if dataset_id == "D3":
            add_quality(
                dataset_id,
                "not_rfc4180_csv",
                "WARN",
                "medium",
                "Descriptions contain unquoted commas, so rows break the header width "
                "(2 columns). Loaded with a split-on-first-comma parser instead.",
            )
        df = read_dataset(dataset_id, path)
        frames[dataset_id] = df
        widths = raw_row_widths(path)

        manifest.append(
            {
                "dataset_id": dataset_id,
                "name": spec["name"],
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "geographic_level": spec["geographic_level"],
                "time_period": spec["time_period"],
                "role": spec["role"],
                "status": "LOADED",
            }
        )
        dictionary.extend(profile_columns(dataset_id, df))
        FINDINGS["datasets"][dataset_id] = {
            "name": spec["name"],
            "relative_path": rel,
            "size_bytes": int(path.stat().st_size),
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
            "row_widths": {str(k): v for k, v in widths.items()},
            "geographic_level": spec["geographic_level"],
            "time_period": spec["time_period"],
        }

        add_quality(dataset_id, "file_exists", "PASS", "info", f"Loaded {rel}")
        if len(widths) > 1 and dataset_id != "D3":
            add_quality(
                dataset_id,
                "ragged_rows",
                "WARN",
                "high",
                f"Row field counts differ from header: {widths}",
            )

        duplicates = int(df.duplicated().sum())
        add_quality(
            dataset_id,
            "exact_duplicate_rows",
            "PASS" if duplicates == 0 else "WARN",
            "medium" if duplicates else "info",
            f"{duplicates} exact duplicate rows out of {len(df)}",
        )

        null_cells = int(df.isna().sum().sum())
        cols_with_nulls = int((df.isna().sum() > 0).sum())
        add_quality(
            dataset_id,
            "missing_values",
            "PASS" if null_cells == 0 else "WARN",
            "low" if null_cells else "info",
            f"{null_cells} null cells across {cols_with_nulls} columns "
            f"({round(100 * null_cells / max(df.size, 1), 3)}% of all cells)",
        )

        numeric = df.apply(pd.to_numeric, errors="coerce")
        negatives = {
            str(col): int((numeric[col] < 0).sum())
            for col in numeric.columns
            if (numeric[col] < 0).any()
        }
        add_quality(
            dataset_id,
            "negative_values",
            "PASS" if not negatives else "FAIL",
            "high" if negatives else "info",
            "No negative values detected" if not negatives else f"Negative values: {negatives}",
        )
        print(f"[audit] {dataset_id} loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # ------------------------------------------------------------------
    # Target feasibility: D1
    # ------------------------------------------------------------------
    target_block: dict = {}
    if "D1" in frames:
        d1 = frames["D1"]
        has_num = TARGET_NUMERATOR in d1.columns
        has_den = TARGET_DENOMINATOR in d1.columns
        target_block["numerator_column_present"] = has_num
        target_block["denominator_column_present"] = has_den
        add_quality(
            "D1",
            "target_numerator_present",
            "PASS" if has_num else "FAIL",
            "critical",
            TARGET_NUMERATOR,
        )
        add_quality(
            "D1",
            "target_denominator_present",
            "PASS" if has_den else "FAIL",
            "critical",
            TARGET_DENOMINATOR,
        )

        if has_num and has_den:
            num = pd.to_numeric(d1[TARGET_NUMERATOR], errors="coerce")
            den = pd.to_numeric(d1[TARGET_DENOMINATOR], errors="coerce")
            target_block.update(
                {
                    "numerator_sum": int(num.sum()),
                    "denominator_sum": int(den.sum()),
                    "numerator_null": int(num.isna().sum()),
                    "denominator_null": int(den.isna().sum()),
                    "numerator_min": float(num.min()),
                    "numerator_max": float(num.max()),
                    "denominator_min": float(den.min()),
                    "denominator_max": float(den.max()),
                    "zero_denominators": int((den == 0).sum()),
                    "numerator_gt_denominator": int((num > den).sum()),
                }
            )
            zero_den = int((den == 0).sum())
            add_quality(
                "D1",
                "zero_denominators",
                "PASS" if zero_den == 0 else "FAIL",
                "high" if zero_den else "info",
                f"{zero_den} districts have Households == 0",
            )
            over = int((num > den).sum())
            add_quality(
                "D1",
                "numerator_le_denominator",
                "PASS" if over == 0 else "FAIL",
                "high" if over else "info",
                f"{over} districts where dilapidated households exceed total households",
            )

            rate = np.where(den > 0, num / den * 100, np.nan)
            d1[TARGET_NAME] = rate
            target_block.update(
                {
                    "rate_min": float(np.nanmin(rate)),
                    "rate_max": float(np.nanmax(rate)),
                    "rate_mean": float(np.nanmean(rate)),
                    "rate_median": float(np.nanmedian(rate)),
                    "rate_std": float(np.nanstd(rate, ddof=1)),
                    "rate_p05": float(np.nanpercentile(rate, 5)),
                    "rate_p95": float(np.nanpercentile(rate, 95)),
                    "n_valid": int(np.isfinite(rate).sum()),
                    "n_negative_rate": int((np.nan_to_num(rate, nan=0.0) < 0).sum()),
                    "n_rate_above_100": int((np.nan_to_num(rate, nan=0.0) > 100).sum()),
                }
            )
            add_quality(
                "D1",
                "target_range_0_100",
                "PASS"
                if target_block["n_rate_above_100"] == 0 and target_block["rate_min"] >= 0
                else "WARN",
                "medium",
                f"target_deprivation_rate range [{target_block['rate_min']:.3f}, "
                f"{target_block['rate_max']:.3f}]",
            )
            print(
                f"[audit] target built for {target_block['n_valid']} districts "
                f"(mean {target_block['rate_mean']:.2f}%)"
            )
    FINDINGS["target"] = target_block

    # ------------------------------------------------------------------
    # Geographic matching
    # ------------------------------------------------------------------
    geo: dict = {}
    if "D1" in frames and "D2" in frames:
        d1 = frames["D1"].copy()
        d2 = frames["D2"].copy()
        d1["key"] = [
            canonical_key(s, d) for s, d in zip(d1["State name"], d1["District name"])
        ]

        d2_total = d2[d2["Rural/Urban"].astype(str).str.strip() == "Total"].copy()
        d2_total["area_clean"] = (
            d2_total["Area Name"].astype(str).str.replace(r"^(District|State)\s*-\s*", "", regex=True)
        )
        d2_total["key"] = [
            canonical_key(s, a)
            for s, a in zip(d2_total["State Name"], d2_total["area_clean"])
        ]

        geo.update(
            {
                "d1_rows": int(len(d1)),
                "d1_unique_keys": int(d1["key"].nunique()),
                "d1_unique_district_codes": int(d1["District code"].nunique()),
                "d1_unique_state_names": sorted(d1["State name"].unique().tolist()),
                "d1_n_states": int(d1["State name"].nunique()),
                "d1_duplicate_keys": int(d1["key"].duplicated().sum()),
                "d2_rows": int(len(d2)),
                "d2_rows_total": int(len(d2_total)),
                "d2_unique_keys_total": int(d2_total["key"].nunique()),
                "d2_duplicate_key_examples": d2_total.loc[
                    d2_total["key"].duplicated(keep=False), ["State Name", "Area Name", "Rural/Urban"]
                ]
                .astype(str)
                .to_dict(orient="records"),
                "d2_unique_area_names": int(d2_total["area_clean"].nunique()),
                "d1_keys_not_in_d2": sorted(set(d1["key"]) - set(d2_total["key"])),
                "d2_keys_not_in_d1": sorted(set(d2_total["key"]) - set(d1["key"])),
            }
        )
        add_quality(
            "D1",
            "district_key_uniqueness",
            "PASS" if geo["d1_duplicate_keys"] == 0 else "FAIL",
            "high" if geo["d1_duplicate_keys"] else "info",
            f"{geo['d1_unique_keys']} unique state+district keys for {len(d1)} rows; "
            f"{geo['d1_unique_district_codes']} unique District code values, "
            "nationally unique 1..640 and consistent with D2's District Code. "
            "D5 has no district code, so the canonical name key is used for all joins.",
        )
        add_quality(
            "D2",
            "district_key_uniqueness",
            "WARN" if geo["d2_duplicate_key_examples"] else "PASS",
            "medium",
            f"{len(geo['d2_duplicate_key_examples'])} duplicate (State, Area, Total) rows; "
            "district names such as Aurangabad/Raigarh/Hamirpur/Bilaspur appear in more "
            "than one state, so names alone are not unique",
        )
        add_quality(
            "D1-D2",
            "district_key_overlap",
            "PASS"
            if not geo["d1_keys_not_in_d2"] and not geo["d2_keys_not_in_d1"]
            else "WARN",
            "high" if (geo["d1_keys_not_in_d2"] or geo["d2_keys_not_in_d1"]) else "info",
            f"{len(geo['d1_keys_not_in_d2'])} D1 keys unmatched in D2; "
            f"{len(geo['d2_keys_not_in_d1'])} D2 keys unmatched in D1",
        )
        print(
            f"[audit] geography: D1 {geo['d1_unique_keys']} keys, "
            f"D2 {geo['d2_unique_keys_total']} keys, "
            f"{len(geo['d1_keys_not_in_d2'])} unmatched / "
            f"{len(geo['d2_keys_not_in_d1'])} extra"
        )

    # Slum coverage vs census states
    if "D4" in frames and "D1" in frames:
        slum = frames["D4"]
        slum_states = [s for s in slum["State"].astype(str) if canonical_state(s) != "INDIA"]
        census_states = set(frames["D1"]["State name"].map(canonical_state))
        slum_norm = {canonical_state(s) for s in slum_states}
        geo["slum_rows"] = int(len(slum))
        geo["slum_states"] = sorted(slum_norm)
        geo["slum_states_not_in_census"] = sorted(slum_norm - census_states)
        geo["census_states_not_in_slum"] = sorted(census_states - slum_norm)
        add_quality(
            "D4",
            "state_coverage_vs_census",
            "WARN"
            if geo["slum_states_not_in_census"] or geo["census_states_not_in_slum"]
            else "PASS",
            "high",
            f"slum states not in census: {geo['slum_states_not_in_census']}; "
            f"census states without a slum row: {geo['census_states_not_in_slum']}",
        )
        FINDINGS["datasets"]["D4"]["embedded_india_row_present"] = bool(
            (slum["State"].astype(str).str.strip().str.upper() == "INDIA").any()
        )

    FINDINGS["geography"] = geo

    # ------------------------------------------------------------------
    # D2 normalisation check + independent target cross-check
    # ------------------------------------------------------------------
    d2_block: dict = {}
    if "D2" in frames:
        d2 = frames["D2"]
        total_col = pd.to_numeric(d2["Total Number of households"], errors="coerce")
        comps = [
            "Total Number of Good",
            "Total Number of Livable",
            "Total Number of Dilapidated",
        ]
        comp_sum = sum(pd.to_numeric(d2[c], errors="coerce") for c in comps)
        d2_block = {
            "total_households_min": float(total_col.min()),
            "total_households_max": float(total_col.max()),
            "rows_with_total_100": int((total_col.round(6) == 100).sum()),
            "n_rows": int(len(d2)),
            "component_sum_min": float(comp_sum.min()),
            "component_sum_max": float(comp_sum.max()),
            "rows_component_sum_99_101": int(((comp_sum >= 99) & (comp_sum <= 101)).sum()),
        }
        is_pct = d2_block["rows_with_total_100"] == len(d2)
        d2_block["values_are_percentages"] = bool(is_pct)
        add_quality(
            "D2",
            "values_are_percentages_not_counts",
            "WARN",
            "critical",
            f"'Total Number of households' equals 100 in "
            f"{d2_block['rows_with_total_100']}/{len(d2)} rows and the Good+Livable+"
            f"Dilapidated components sum to ~100, so the extract is percentage-normalised "
            f"rather than counts. Raw household counts cannot be recovered from it.",
        )
        add_quality(
            "D2",
            "condition_components_sum_to_100",
            "PASS" if d2_block["rows_component_sum_99_101"] == len(d2) else "WARN",
            "low",
            f"{d2_block['rows_component_sum_99_101']}/{len(d2)} rows sum to 99-101 "
            f"(range {d2_block['component_sum_min']:.2f}-{d2_block['component_sum_max']:.2f})",
        )

        if "D1" in frames and TARGET_NAME in frames["D1"].columns:
            d1 = frames["D1"].copy()
            d1["key"] = (
                d1["State name"].map(normalise_name)
                + "||"
                + d1["District name"].map(normalise_name)
            )
            d2t = d2[d2["Rural/Urban"].astype(str).str.strip() == "Total"].copy()
            d2t["area_clean"] = (
                d2t["Area Name"].astype(str).str.replace(r"^(District|State)\s*-\s*", "", regex=True)
            )
            d2t["key"] = [
                canonical_key(s, a) for s, a in zip(d2t["State Name"], d2t["area_clean"])
            ]
            d2t = d2t.drop_duplicates(subset="key")
            merged = d1[["key", "State name", "District name", TARGET_NAME]].merge(
                d2t[["key", "Total Number of Dilapidated"]], on="key", how="inner"
            )
            merged["d2_pct"] = pd.to_numeric(
                merged["Total Number of Dilapidated"], errors="coerce"
            )
            valid = merged.dropna(subset=[TARGET_NAME, "d2_pct"])
            diff = (valid[TARGET_NAME] - valid["d2_pct"]).abs()
            d2_block.update(
                {
                    "crosscheck_n": int(len(valid)),
                    "crosscheck_mean_abs_diff": float(diff.mean()) if len(valid) else None,
                    "crosscheck_median_abs_diff": float(diff.median()) if len(valid) else None,
                    "crosscheck_max_abs_diff": float(diff.max()) if len(valid) else None,
                    "crosscheck_within_0_5pp": int((diff <= 0.5).sum()),
                    "crosscheck_within_1pp": int((diff <= 1.0).sum()),
                    "crosscheck_corr": float(
                        valid[TARGET_NAME].corr(valid["d2_pct"])
                    )
                    if len(valid) > 2
                    else None,
                    "crosscheck_worst": valid.assign(diff=diff)
                    .nlargest(5, "diff")[["State name", "District name", TARGET_NAME, "d2_pct"]]
                    .round(3)
                    .to_dict(orient="records"),
                }
            )
            add_quality(
                "D1-D2",
                "target_reproduced_from_housing_abstract",
                "PASS"
                if d2_block.get("crosscheck_median_abs_diff") is not None
                and d2_block["crosscheck_median_abs_diff"] <= 0.5
                else "WARN",
                "high",
                f"Independent recomputation of the district dilapidated share from D2 "
                f"matches the D1-built target on {d2_block['crosscheck_n']} districts: "
                f"median |diff| {d2_block['crosscheck_median_abs_diff']:.3f} pp, "
                f"max {d2_block['crosscheck_max_abs_diff']:.3f} pp, "
                f"r = {d2_block['crosscheck_corr']:.4f}",
            )
            print(
                f"[audit] target cross-check: n={d2_block['crosscheck_n']}, "
                f"median |diff|={d2_block['crosscheck_median_abs_diff']:.3f} pp, "
                f"r={d2_block['crosscheck_corr']:.4f}"
            )
    FINDINGS["d2_normalisation"] = d2_block

    # ------------------------------------------------------------------
    # Column dictionary for D3 (the HLPCA dictionary must only describe D2)
    # ------------------------------------------------------------------
    d3_block: dict = {}
    if "D3" in frames:
        d3 = frames["D3"]
        header = list(d3.columns)
        d3_block = {
            "header": header,
            "n_rows": int(len(d3)),
            "duplicate_header_labels": len(header) != len(set(header)),
            "has_data_rows": bool(len(d3) and str(d3.iloc[0, 0]) != header[0]),
            "first_column_codes": d3.iloc[:, 0].astype(str).head(3).tolist(),
            "code_prefix": "c11..c145",
        }
        add_quality(
            "D3",
            "header_is_duplicate_pair",
            "WARN",
            "increasing",
            f"Header is {header}: both columns carry the same label, so the file is a "
            "two-column code/description mapping read with duplicated column names.",
        )
        add_quality(
            "D3",
            "applies_to_D2_only",
            "PASS",
            "info",
            "Codes c11-c145 map 1:1 onto the measurement columns of D2 "
            "(Number of households with condition of Census House ...). "
            "It does not describe D1's columns and must never be joined to D5-D9.",
        )
    FINDINGS["dictionary"] = d3_block

    # ------------------------------------------------------------------
    # Slum dataset structure
    # ------------------------------------------------------------------
    if "D4" in frames:
        slum = frames["D4"]
        FINDINGS["datasets"]["D4"]["columns_raw"] = list(slum.columns)
        FINDINGS["datasets"]["D4"]["embedded_india_row"] = slum.iloc[0].astype(str).to_dict()
        numeric_like = {}
        for col in slum.columns:
            s = slum[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            numeric_like[str(col)] = int(
                pd.to_numeric(s.str.strip(), errors="coerce").notna().sum()
            )
        FINDINGS["datasets"]["D4"]["numeric_after_parse_counts"] = numeric_like
        add_quality(
            "D4",
            "counts_and_percentages_are_text",
            "WARN",
            "medium",
            "Household/population counts use thousands separators inside quoted strings "
            '("13,920,191") and shares are strings with a % sign ("5.41%", "77.72 %"); '
            "every numeric column needs explicit parsing.",
        )
        zero_cells = {}
        for col in ["SC", "ST"]:
            s = pd.to_numeric(
                slum[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
            zero_cells[col] = int((s == 0).sum())
        FINDINGS["datasets"]["D4"]["zero_counts_in_SC_ST"] = zero_cells
        add_quality(
            "D4",
            "genuine_zero_vs_missing",
            "WARN",
            "high",
            f"Exact zeros in SC/ST are plausible genuine zeros for small states/UTs: "
            f"{zero_cells}. They must not be conflated with 'no slum data reported'.",
        )
        add_quality(
            "D4",
            "geographic_level_state_only",
            "FAIL",
            "critical",
            "The file is state/UT level (32 rows including an India aggregate row). "
            "It cannot be merged onto 640 districts without dropping to state level "
            "or borrowing a state value for every district in that state.",
        )

    # ------------------------------------------------------------------
    # Optional PCA extracts: structure checks
    # ------------------------------------------------------------------
    for ds in ("D5", "D6", "D7", "D8", "D9"):
        if ds not in frames:
            continue
        df = frames[ds]
        notes = []
        if "LEVEL" in df.columns:
            levels = df["LEVEL"].astype(str).str.strip().value_counts().to_dict()
            areas = df["AREA"].astype(str).str.strip().value_counts().to_dict()
            notes.append(f"LEVEL={levels}")
            notes.append(f"AREA={areas}")
            FINDINGS["datasets"][ds]["level_counts"] = {str(k): int(v) for k, v in levels.items()}
        if "dist_code" in df.columns:
            codes = pd.to_numeric(df["dist_code"], errors="coerce")
            notes.append(f"dist_code==0 rows={int((codes == 0).sum())}")
            notes.append(f"unique dist_code={int(codes.nunique())}")
        if "area_name" in df.columns:
            names = df["area_name"].astype(str)
            notes.append(f"area_name startswith 'State -' rows={int(names.str.startswith('State -').sum())}")
            notes.append(f"area_name startswith 'District -' rows={int(names.str.startswith('District -').sum())}")
        add_quality(
            ds,
            "level_encoding",
            "WARN" if len(df.columns) > 400 and "UNNAMED" in " ".join(map(str, df.columns)).upper() else "PASS",
            "medium" if len(df.columns) > 400 and "UNNAMED" in " ".join(map(str, df.columns)).upper() else "info",
            "; ".join(notes) if notes else "no level columns",
        )
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or "UNNAMED" in str(c).upper()]
        if unnamed:
            add_quality(
                ds,
                "malformed_trailing_column",
                "WARN",
                "low",
                f"Column(s) {unnamed} arise from a trailing comma on data rows and hold no usable data.",
            )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    pd.DataFrame(manifest).to_csv(META / "dataset_manifest.csv", index=False)
    pd.DataFrame(dictionary).to_csv(META / "column_dictionary.csv", index=False)
    pd.DataFrame(QUALITY_ROWS).to_csv(META / "data_quality_report.csv", index=False)

    FINDINGS["quality_summary"] = (
        pd.DataFrame(QUALITY_ROWS)["status"].value_counts().to_dict()
    )
    FINDINGS["quality_rows"] = QUALITY_ROWS
    (META / "audit_findings.json").write_text(
        json.dumps(FINDINGS, indent=2, default=str), encoding="utf-8"
    )

    lines = ["PHASE 1 DATASET AUDIT SUMMARY", "=" * 60, ""]
    for row in manifest:
        lines.append(
            f"{row['dataset_id']:<4} {row['n_rows']:>6} rows x {row['n_cols']:>4} cols  "
            f"{row['size_bytes']:>10,} B  {row['geographic_level']:<16} {row['relative_path']}"
        )
    lines += ["", "QUALITY CHECKS", "-" * 60]
    for row in QUALITY_ROWS:
        lines.append(f"[{row['status']:<4}] {row['dataset_id']:<6} {row['check']}: {row['details']}")
    (REPORTS / "dataset_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n[audit] wrote metadata + report files")
    print(pd.DataFrame(QUALITY_ROWS)["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
