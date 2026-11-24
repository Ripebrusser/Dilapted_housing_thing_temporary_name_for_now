"""Runner for the authorised Phase 4 tail-aware investigation.

Usage
-----
    py src/training/run_tail_experiments.py --smoke
    py src/training/run_tail_experiments.py                    # E1, E2, E3, E4a, E4b
    py src/training/run_tail_experiments.py --arms E1,E2

Only the five registered arms and their fixed grids are available; there is no
flag that widens the search. Runs are resumable and merge on ``arm``.

Stop conditions (the runner exits before fitting anything if one is hit):
* the approved target is missing or renamed;
* Configuration B is not 628 districts by 77 predictors;
* the district keys do not match the Phase 3 out-of-fold predictions;
* the frozen target thresholds no longer match the Phase 3 values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.tail_aware import (  # noqa: E402
    INNER_SPLITS,
    build_arms,
    run_arm,
)
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

PROCESSED = ROOT / "datasets" / "processed"
METRICS = ROOT / "evaluation" / "metrics"
PREDICTIONS = ROOT / "evaluation" / "predictions"

TARGET = "target_deprivation_rate_pct"
CONFIG_ID = "b628"
EXPECTED_DISTRICTS = 628
EXPECTED_FEATURES = 77
EXPECTED_P90 = 9.200
EXPECTED_P95 = 11.200

PHASE3_OOF = PREDICTIONS / "phase3_oof_predictions.csv"

REGISTERED = ("E1", "E2", "E3", "E4a", "E4b")


def preflight(config: dict) -> dict:
    """Verify every boundary before a single model is fitted."""
    X, y, keys = config["X"], config["y"], config["keys"]

    frame = pd.read_csv(PROCESSED / "district_analysis.csv")
    if TARGET not in frame.columns:
        raise SystemExit(f"STOP: approved target {TARGET!r} not found")

    if len(y) != EXPECTED_DISTRICTS:
        raise SystemExit(f"STOP: Configuration B has {len(y)} districts, expected {EXPECTED_DISTRICTS}")
    if X.shape[1] != EXPECTED_FEATURES:
        raise SystemExit(f"STOP: Configuration B has {X.shape[1]} predictors, expected {EXPECTED_FEATURES}")
    if X.isna().any().any():
        raise SystemExit("STOP: feature matrix contains missing values")

    if not PHASE3_OOF.exists():
        raise SystemExit(f"STOP: missing frozen Phase 3 predictions at {PHASE3_OOF}")
    frozen = pd.read_csv(PHASE3_OOF)
    frozen = frozen[
        (frozen["configuration"] == "B_augmented_628") & (frozen["scheme"] == "grouped")
    ]
    if frozen.empty:
        raise SystemExit("STOP: Phase 3 B/grouped OOF rows not found")
    frozen_keys = frozen[frozen["model"] == "ridge"].sort_values("key")["key"].tolist()
    if sorted(keys.tolist()) != frozen_keys:
        raise SystemExit("STOP: district keys do not match the Phase 3 reference predictions")
    if sorted(keys.tolist()) != sorted(frozen["key"].unique().tolist()):
        raise SystemExit("STOP: district keys differ from the Phase 3 configuration")

    p90, p95 = float(np.quantile(y, 0.90)), float(np.quantile(y, 0.95))
    if not (np.isclose(p90, EXPECTED_P90, atol=1e-9) and np.isclose(p95, EXPECTED_P95, atol=1e-9)):
        raise SystemExit(
            f"STOP: frozen thresholds changed (p90={p90:.3f}, p95={p95:.3f}); "
            "expected 9.200 / 11.200"
        )

    return {
        "n_districts": int(len(y)),
        "n_features": int(X.shape[1]),
        "p90": p90,
        "p95": p95,
        "target_mean": float(np.mean(y)),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 tail-aware experiment runner")
    parser.add_argument("--arms", default=",".join(REGISTERED))
    parser.add_argument("--smoke", action="store_true", help="fold 1 only, first alpha, no deliverables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    registry = build_arms()
    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(requested) - set(registry)
    if unknown:
        raise SystemExit(f"unregistered arm(s): {sorted(unknown)}; allowed: {sorted(registry)}")

    config = build_configurations([CONFIG_ID])[CONFIG_ID]
    boundary = preflight(config)
    print(f"preflight OK: {boundary['n_districts']} districts, {boundary['n_features']} predictors, "
          f"p90={boundary['p90']:.3f}, p95={boundary['p95']:.3f}")

    X, y, groups, keys = config["X"], config["y"], config["groups"], config["keys"]
    folds = build_outer_folds(groups, "grouped")
    fold_of = np.empty(len(y), dtype=int)
    for fold_id, (_, test_idx) in enumerate(folds, start=1):
        fold_of[test_idx] = fold_id

    suffix = "_smoke" if args.smoke else ""
    out_metrics = METRICS / f"tail{suffix}_selected_params.json"
    out_diag = METRICS / f"tail{suffix}_fit_diagnostics.csv"
    out_oof = PREDICTIONS / f"tail{suffix}_oof_predictions.csv"
    out_runtime = METRICS / f"tail{suffix}_runtime.csv"

    METRICS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    params: dict = {}
    runtime_rows: list[dict] = []

    for name in requested:
        arm = registry[name]
        if args.smoke:
            arm = type(arm)(
                name=arm.name,
                kind=arm.kind,
                alpha_grid=arm.alpha_grid[:1],
                quantile=arm.quantile,
                lam=arm.lam,
                tau=arm.tau,
                description=arm.description,
            )
            run_folds = folds[:1]
        else:
            run_folds = folds

        print(f"[{name}] {arm.description}")
        started = time.perf_counter()
        oof, records = run_arm(
            arm, X, y, groups, run_folds, INNER_SPLITS, diagnostics,
            require_full_coverage=not args.smoke,
        )
        elapsed = time.perf_counter() - started

        selected = [
            {"fold": r["fold"], "best_alpha": r["best_alpha"], "threshold": r["threshold"],
             "n_weighted": r["n_weighted"], "inner_table": r["inner_table"]}
            for r in records
        ]
        params[name] = {**arm.as_dict(), "per_fold": selected}
        runtime_rows.append({"arm": name, "seconds": round(elapsed, 3), "n_folds": len(run_folds)})

        frame = pd.DataFrame(
            {
                "arm": name,
                "kind": arm.kind,
                "key": keys.to_numpy(),
                "fold": fold_of,
                "actual": y,
                "predicted": oof,
            }
        )
        if args.smoke:
            frame = frame[frame["fold"] == 1].copy()
        oof_frames.append(frame)

        print(f"  done in {elapsed:.1f}s "
              f"(alpha per fold: {[r['best_alpha'] for r in records]})")

    diagnostics_frame = pd.DataFrame(diagnostics)
    if not args.smoke and out_diag.exists():
        old = pd.read_csv(out_diag)
        keep = old[~old["arm"].isin(requested)]
        diagnostics_frame = pd.concat([keep, diagnostics_frame], ignore_index=True)
    diagnostics_frame.to_csv(out_diag, index=False)

    combined = pd.concat(oof_frames, ignore_index=True)
    if not args.smoke and out_oof.exists():
        old = pd.read_csv(out_oof)
        keep = old[~old["arm"].isin(requested)]
        combined = pd.concat([keep, combined], ignore_index=True)
    combined = combined.sort_values(["arm", "key"]).reset_index(drop=True)
    combined.to_csv(out_oof, index=False)

    existing_params = {}
    if not args.smoke and out_metrics.exists():
        existing_params = json.loads(out_metrics.read_text(encoding="utf-8")).get("arms", {})
    existing_params.update(params)
    out_metrics.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed,
                "inner_splits": INNER_SPLITS,
                "configuration": config["label"],
                "outer_scheme": "grouped",
                "boundary_checks": boundary,
                "arms": existing_params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if out_runtime.exists() and not args.smoke:
        old_runtime = pd.read_csv(out_runtime)
        old_runtime = old_runtime[~old_runtime["arm"].isin(requested)]
        runtime_rows = pd.concat(
            [old_runtime, pd.DataFrame(runtime_rows)], ignore_index=True
        ).to_dict("records")
    pd.DataFrame(runtime_rows).to_csv(out_runtime, index=False)

    print(f"wrote {out_diag.relative_to(ROOT)} ({len(diagnostics_frame)} fit rows)")
    print(f"wrote {out_oof.relative_to(ROOT)} ({len(combined)} prediction rows)")
    print(f"wrote {out_metrics.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
