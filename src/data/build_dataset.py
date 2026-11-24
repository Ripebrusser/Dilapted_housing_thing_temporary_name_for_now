"""Build the Phase 2 dataset and run baseline evaluation.

Produces:
    datasets/processed/district_analysis.csv
    datasets/processed/feature_manifest.csv
    evaluation/metrics/phase2_baselines.csv
    evaluation/metrics/phase2_baselines_wide.csv
    evaluation/metrics/phase2_folds.csv
    evaluation/metrics/phase2_run_summary.json

Run: py src/data/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "data"))

from src.data.loaders import (  # noqa: E402
    N_DISTRICTS,
    N_DISTRICTS_WITH_D5,
    PROCESSED,
    assert_merge,
    load_d1,
    load_d2_total,
    load_d5_households,
)
from src.evaluation.baselines import run_baselines, summarise  # noqa: E402
from src.features.predictors import (  # noqa: E402
    FORBIDDEN_SOURCES,
    KNN_FEATURES,
    CONFIG_A,
    build_features,
    screen_features,
)
from src.features.target import (  # noqa: E402
    TARGET_NAME,
    build_target,
    describe_target,
    target_metadata,
)
from src.training.splits import (  # noqa: E402
    assert_folds_valid,
    fold_table,
    grouped_folds,
    random_folds,
)

METRICS_DIR = ROOT / "evaluation" / "metrics"


def check(name: str, status: str, details: str) -> dict:
    return {"check": name, "status": status, "details": details}


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Load (validating loaders)
    # ------------------------------------------------------------------
    d1 = load_d1()
    d2_total = load_d2_total()
    d5_households = load_d5_households(d1)
    checks.append(check("load_d1", "PASS", f"{len(d1)} districts, {d1.shape[1]} columns"))
    checks.append(check("load_d2_total", "PASS", f"{len(d2_total)} district Total rows"))
    checks.append(
        check(
            "load_d5_households",
            "PASS",
            f"{int(d5_households.notna().sum())} districts with a reliable D5 household count "
            f"({int(d5_households.isna().sum())} left missing: district name occurs in two states)",
        )
    )

    # ------------------------------------------------------------------
    # 2. Target (D-001, approved: T2 only)
    # ------------------------------------------------------------------
    target = build_target(d2_total)
    target_stats = describe_target(target[TARGET_NAME])
    checks.append(
        check(
            "target_construction",
            "PASS",
            f"{target_stats['n']} districts from D2 Total rows; "
            f"range {target_stats['min']}-{target_stats['max']}%, "
            f"median {target_stats['median']}%, {target_stats['n_unique']} distinct values",
        )
    )

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    features, manifest, feature_checks = build_features(d1, d5_households)
    checks.extend(feature_checks)

    # ------------------------------------------------------------------
    # 4. Merge the modelling table with explicit row-count guards
    # ------------------------------------------------------------------
    data = d1[["key", "State name", "District name", "District code"]].rename(
        columns={
            "State name": "state",
            "District name": "district",
            "District code": "district_code",
        }
    )
    data = assert_merge(
        data.merge(target, on="key", how="left", validate="one_to_one"),
        N_DISTRICTS,
        "target merge",
    )
    data = assert_merge(
        data.merge(features, on="key", how="left", validate="one_to_one"),
        N_DISTRICTS,
        "feature merge",
    )
    data["has_d5_households"] = data["key"].map(d5_households).notna()

    if set(data["key"]) != set(d1["key"]):
        raise AssertionError("target district alignment: key sets differ from D1")
    if data[TARGET_NAME].isna().any():
        raise AssertionError("district_analysis: unexpected missing target values")
    checks.append(check("district_alignment", "PASS", "640 keys identical to D1; no duplicates"))

    subset_mask = data["has_d5_households"].to_numpy()
    checks.append(
        check(
            "config_b_coverage",
            "PASS",
            f"Configuration B available for {int(subset_mask.sum())} districts "
            f"(expected {N_DISTRICTS_WITH_D5}); {int((~subset_mask).sum())} preserved as missing",
        )
    )

    # ------------------------------------------------------------------
    # 5. Screens: collinearity + indirect target leakage
    # ------------------------------------------------------------------
    config_a = manifest.loc[manifest["configuration"] == CONFIG_A, "feature_name"].tolist()
    config_b_only = manifest.loc[manifest["configuration"] != CONFIG_A, "feature_name"].tolist()

    kept_a, kept_b, manifest, screen_checks = screen_features(
        features,
        data[TARGET_NAME],
        manifest,
        checks,
        config_a,
        config_b_only,
        subset_mask,
    )
    checks.extend(screen_checks)

    missing_a = {c: int(data[c].isna().sum()) for c in kept_a if data[c].isna().any()}
    checks.append(
        check(
            "config_a_missingness",
            "PASS" if not missing_a else "WARN",
            "no missing values in Configuration A" if not missing_a else f"missing: {missing_a}",
        )
    )
    missing_b = {
        c: int(data[c].isna().sum()) for c in kept_b if int(data[c].isna().sum()) != 12
    }
    checks.append(
        check(
            "config_b_missingness",
            "PASS" if not missing_b else "WARN",
            "every Configuration-B feature is missing for exactly the 12 unresolvable districts"
            if not missing_b
            else f"unexpected missingness pattern: {missing_b}",
        )
    )

    used_sources = set()
    for row in manifest.loc[manifest["retained"], "source_columns"]:
        used_sources.update(str(row).split("|"))
    leak_hits = FORBIDDEN_SOURCES.intersection(used_sources)
    checks.append(
        check(
            "leakage_denylist",
            "PASS" if not leak_hits else "FAIL",
            "no forbidden (target-derived) column used as a predictor"
            if not leak_hits
            else f"forbidden sources present: {sorted(leak_hits)}",
        )
    )

    # ------------------------------------------------------------------
    # 6. Persist
    # ------------------------------------------------------------------
    columns = ["key", "state", "district", "district_code", TARGET_NAME, "has_d5_households"] + kept_a + kept_b
    analysis = data[columns].copy()
    analysis.to_csv(PROCESSED / "district_analysis.csv", index=False)
    manifest.to_csv(PROCESSED / "feature_manifest.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Baselines (identical folds for the A/B comparison)
    # ------------------------------------------------------------------
    y_all = data[TARGET_NAME].to_numpy(dtype=float)
    state_all = data["state"].to_numpy()

    y_sub = y_all[subset_mask]
    state_sub = state_all[subset_mask]

    folds_640 = {"grouped_by_state": grouped_folds(state_all)}
    folds_640.update(random_folds(len(y_all)))
    folds_628 = {"grouped_by_state": grouped_folds(state_sub)}
    folds_628.update(random_folds(len(y_sub), seed=42))

    assert_folds_valid(folds_640["grouped_by_state"], len(y_all), state_all)
    assert_folds_valid(folds_628["grouped_by_state"], len(y_sub), state_sub)

    knn_features = [c for c in KNN_FEATURES if c in kept_a]
    if len(knn_features) < 3:
        raise AssertionError(f"k-NN headline features missing: {KNN_FEATURES}")

    rows: list[dict] = []
    rows += run_baselines(
        "A_primary_640", y_all, state_all, data[kept_a],
        folds_640, len(y_all), data.loc[:, knn_features],
    )
    rows += run_baselines(
        "A_restricted_628", y_sub, state_sub, data.loc[subset_mask, kept_a],
        folds_628, len(y_sub), data.loc[subset_mask, knn_features],
    )
    rows += run_baselines(
        "B_augmented_628", y_sub, state_sub,
        data.loc[subset_mask, kept_a + kept_b],
        folds_628, len(y_sub), data.loc[subset_mask, knn_features],
    )

    baseline_frame = pd.DataFrame(rows)
    baseline_frame.to_csv(METRICS_DIR / "phase2_baselines.csv", index=False)
    wide = summarise(rows)
    wide.to_csv(METRICS_DIR / "phase2_baselines_wide.csv", index=False)

    fold_frames = [
        fold_table(data["key"], folds_640["grouped_by_state"], "grouped_by_state_640"),
        fold_table(data.loc[subset_mask, "key"], folds_628["grouped_by_state"], "grouped_by_state_628"),
        fold_table(data["key"], folds_640["random_rep1_seed42"], "random_rep1_seed42_640"),
    ]
    pd.concat(fold_frames, ignore_index=True).to_csv(METRICS_DIR / "phase2_folds.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    summary = {
        "phase": 2,
        "target": target_metadata(),
        "target_stats": target_stats,
        "rows": {
            "district_analysis": int(len(analysis)),
            "districts_with_config_b": int(subset_mask.sum()),
            "districts_without_config_b": int((~subset_mask).sum()),
        },
        "features": {
            "config_a": len(kept_a),
            "config_b_additional": len(kept_b),
            "config_b_total": len(kept_a) + len(kept_b),
            "built": int(len(manifest)),
            "dropped_by_screen": int((~manifest["retained"]).sum()),
            "config_a_names": kept_a,
            "config_b_additional_names": kept_b,
        },
        "screening": {
            "drops": [
                {"feature": r["feature_name"], "reason": r["drop_reason"]}
                for _, r in manifest.loc[~manifest["retained"]].iterrows()
            ],
        },
        "checks": checks,
        "baselines_wide": wide.to_dict(orient="records"),
    }
    (METRICS_DIR / "phase2_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 78)
    print("PHASE 2 BUILD SUMMARY")
    print("=" * 78)
    print(f"district_analysis rows .......... {len(analysis)}")
    print(f"target ......................... {TARGET_NAME}")
    print(f"  {target_stats}")
    print(f"Configuration A features ....... {len(kept_a)}")
    print(f"Configuration B additions ...... {len(kept_b)}")
    print(f"features built / dropped ....... {len(manifest)} / {int((~manifest['retained']).sum())}")
    print()
    for row in checks:
        print(f"[{row['status']:4}] {row['check']}: {row['details']}")
    print()
    print("BASELINES (mean over folds)")
    print(wide.to_string(index=False))
    print()
    print(f"written: {PROCESSED / 'district_analysis.csv'}")
    print(f"written: {PROCESSED / 'feature_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
