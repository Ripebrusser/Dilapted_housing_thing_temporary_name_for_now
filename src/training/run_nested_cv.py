"""CLI runner for Phase 3 nested cross-validation.

Usage
-----
    py src/training/run_nested_cv.py --scheme grouped
    py src/training/run_nested_cv.py --scheme random
    py src/training/run_nested_cv.py --scheme grouped --configs a640 --models ridge,knn

Runs are **resumable**: results are merged on (configuration, model) so a later
invocation adds or replaces models without discarding earlier ones. Pass
``--no-append`` to start clean for a scheme.

Writes intermediate artefacts prefixed with ``_phase3_``; ``finalize_phase3.py``
assembles the deliverable filenames. Nothing here touches Phase 2 outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.registry import build_models, describe_models, SEED  # noqa: E402
from src.training.nested_cv import OUTER_SPLITS, run_configuration  # noqa: E402

PROCESSED = ROOT / "datasets" / "processed"
METRICS = ROOT / "evaluation" / "metrics"
PREDICTIONS = ROOT / "evaluation" / "predictions"
TARGET = "target_deprivation_rate_pct"

CONFIG_LABELS = {
    "a640": "A_primary_640",
    "a628": "A_restricted_628",
    "b628": "B_augmented_628",
}

# Constant predictions make the rank correlation undefined; NaN is the correct
# result and is reported as such, so the warning itself is just noise.
warnings.filterwarnings("ignore", message="An input array is constant")


def build_configurations(config_ids: list[str]) -> dict[str, dict]:
    """Return X/y/groups/keys for each requested configuration.

    The 628-district subsets are built from one filter operation so A and B share
    both the districts and their row order; the fold builder then sees identical
    state vectors for both.
    """
    frame = pd.read_csv(PROCESSED / "district_analysis.csv")
    manifest = pd.read_csv(PROCESSED / "feature_manifest.csv")
    retained = manifest[manifest["retained"]]
    a_cols = retained.loc[retained["configuration"] == "A", "feature_name"].tolist()
    b_cols = retained.loc[retained["configuration"] != "A", "feature_name"].tolist()
    subset = frame[frame["has_d5_households"].astype(bool)].copy()

    configurations: dict[str, dict] = {}
    for config_id in config_ids:
        label = CONFIG_LABELS[config_id]
        if config_id == "a640":
            data, columns = frame, a_cols
        elif config_id == "a628":
            data, columns = subset, a_cols
        elif config_id == "b628":
            data, columns = subset, a_cols + b_cols
        else:
            raise SystemExit(f"unknown configuration: {config_id}")

        X = data[columns]
        if X.isna().any().any():
            raise AssertionError(f"{label}: feature matrix has missing values")
        configurations[config_id] = {
            "label": label,
            "X": X,
            "y": data[TARGET].to_numpy(dtype=float),
            "groups": data["state"].to_numpy(),
            "keys": data["key"].reset_index(drop=True),
            "n_features": len(columns),
        }
    return configurations


def _merge_csv(path: Path, new: pd.DataFrame, keys: list[str], append: bool) -> pd.DataFrame:
    if append and path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=keys, keep="last")
        return combined
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 nested CV runner")
    parser.add_argument("--scheme", choices=["grouped", "random"], required=True)
    parser.add_argument("--configs", default="a640,a628,b628")
    parser.add_argument("--models", default="")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args(argv)

    METRICS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    append = not args.no_append

    config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    models = build_models(seed=args.seed)
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = set(wanted) - set(models)
        if missing:
            raise SystemExit(f"unknown models: {sorted(missing)}")
        models = {name: models[name] for name in wanted}

    print(f"scheme={args.scheme} configs={config_ids} models={list(models)}")
    started = time.time()
    summaries: list[dict] = []
    folds: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    params: dict = {}
    fold_signature: dict = {}

    for config_id in config_ids:
        config = build_configurations([config_id])[config_id]
        print(
            f"[{config['label']}] n={len(config['y'])} features={config['n_features']} "
            f"states={len(np.unique(config['groups']))}"
        )
        summary, fold_rows, frames, best = run_configuration(
            config["label"],
            config["X"],
            config["y"],
            config["groups"],
            config["keys"],
            models,
            scheme=args.scheme,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        summaries.extend(summary)
        folds.extend(fold_rows)
        oof_frames.extend(frames)
        params[config["label"]] = best
        first_model = next(iter(models))
        fold_signature[config["label"]] = {
            str(row["fold"]): {
                "n_train": row["n_train"],
                "n_test": row["n_test"],
                "n_states_test": row["n_states_test"],
            }
            for row in fold_rows
            if row["model"] == first_model
        }

    summary_path = METRICS / f"_phase3_{args.scheme}_summary.csv"
    folds_path = METRICS / f"_phase3_{args.scheme}_folds.csv"
    oof_path = PREDICTIONS / f"_phase3_{args.scheme}_oof.csv"
    params_path = METRICS / f"_phase3_{args.scheme}_params.json"

    summary_frame = _merge_csv(
        summary_path, pd.DataFrame(summaries), ["configuration", "model"], append
    )
    folds_frame = _merge_csv(
        folds_path, pd.DataFrame(folds), ["configuration", "model", "fold"], append
    )
    oof_frame = _merge_csv(
        oof_path,
        pd.concat(oof_frames, ignore_index=True),
        ["configuration", "model", "key"],
        append,
    )
    summary_frame.to_csv(summary_path, index=False)
    folds_frame.to_csv(folds_path, index=False)
    oof_frame.to_csv(oof_path, index=False)

    previous: dict = {}
    if append and params_path.exists():
        previous = json.loads(params_path.read_text(encoding="utf-8"))
    merged_params = dict(previous.get("best_params", {}))
    merged_params.update(params)
    previous_meta = previous.get("meta", {})
    merged_signature = dict(previous_meta.get("fold_signature", {}))
    merged_signature.update(fold_signature)
    by_name = {entry["model"]: entry for entry in previous_meta.get("models", [])}
    by_name.update({entry["model"]: entry for entry in describe_models(models)})

    meta = {
        "scheme": args.scheme,
        "seed": args.seed,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": 4,
        "config_ids": sorted({c for c in summary_frame["configuration"].unique()}),
        "models": [by_name[name] for name in summary_frame["model"].drop_duplicates()],
        "fold_signature": merged_signature,
        "n_rows": int(len(oof_frame)),
        "elapsed_seconds": round(
            float(previous_meta.get("elapsed_seconds", 0.0)) + time.time() - started, 1
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "scipy"]
        },
    }
    params_path.write_text(
        json.dumps({"meta": meta, "best_params": merged_params}, indent=2, default=str),
        encoding="utf-8",
    )
    (METRICS / f"_phase3_{args.scheme}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    elapsed = round(time.time() - started, 1)
    print(f"\nThis run: {elapsed}s. Accumulated rows: {len(summary_frame)}")
    print(
        summary_frame.sort_values(["configuration", "mae_mean"])[
            ["configuration", "model", "mae_mean", "r2_mean", "spearman_mean"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
