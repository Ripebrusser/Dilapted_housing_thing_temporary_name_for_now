"""Evaluation of the Phase 4 tail-aware arms.

Reads only frozen or Phase 4 artefacts; fits nothing. Applies the registered
criteria (C1-C6) exactly as written in
``reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md``:

* E0 is the ratified Phase 3 Ridge out-of-fold prediction, reused as-is;
* tail groups use the frozen Phase 3 target thresholds, p90 = 9.200, p95 = 11.200;
* the conditional-quantile arms are reported with pinball loss, coverage and
  calibration error, and their MAE-based gates are marked NOT APPLICABLE rather
  than forced into a like-for-like comparison.

Uncertainty is described with a **state-clustered** bootstrap, because the tail
districts are concentrated in a few states and a district-level bootstrap would
understate that dependence.

Usage: py src/evaluation/tail_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import regression_metrics  # noqa: E402
from src.evaluation.metrics import summarise_folds  # noqa: E402
from src.models.tail_aware import pinball_loss  # noqa: E402
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

METRICS = ROOT / "evaluation" / "metrics"
PREDICTIONS = ROOT / "evaluation" / "predictions"

P90 = 9.200
P95 = 11.200
TAIL_STATES = ("ASSAM", "WEST BENGAL", "UTTAR PRADESH", "ORISSA")
BOOTSTRAP_REPS = 2000
SEED = 42
REFERENCE = "E0"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load() -> dict:
    config = build_configurations(["b628"])["b628"]
    frame = pd.DataFrame(
        {
            "key": config["keys"].to_numpy(),
            "state": config["groups"],
            "actual": config["y"],
        }
    )
    folds = build_outer_folds(config["groups"], "grouped")
    fold_of = np.empty(len(frame), dtype=int)
    for fold_id, (_, test_idx) in enumerate(folds, start=1):
        fold_of[test_idx] = fold_id
    frame["fold"] = fold_of

    if not (
        np.isclose(np.quantile(frame["actual"], 0.90), P90, atol=1e-9)
        and np.isclose(np.quantile(frame["actual"], 0.95), P95, atol=1e-9)
    ):
        raise SystemExit("STOP: frozen evaluation thresholds no longer match 9.200 / 11.200")

    phase3 = pd.read_csv(PREDICTIONS / "phase3_oof_predictions.csv")
    phase3 = phase3[
        (phase3["configuration"] == "B_augmented_628")
        & (phase3["scheme"] == "grouped")
        & (phase3["model"] == "ridge")
    ]
    reference = phase3.set_index("key")["predicted"]
    frame["E0"] = frame["key"].map(reference)
    if frame["E0"].isna().any():
        raise SystemExit("STOP: the Phase 3 reference does not cover every district")

    tail = pd.read_csv(PREDICTIONS / "tail_oof_predictions.csv")
    arms = [arm for arm in tail["arm"].unique() if arm != REFERENCE]
    for arm in arms:
        block = tail[tail["arm"] == arm].set_index("key")["predicted"]
        mapped = frame["key"].map(block)
        if mapped.isna().any():
            raise SystemExit(f"STOP: {arm} does not cover every district")
        frame[arm] = mapped

    kinds = json.loads((METRICS / "tail_selected_params.json").read_text())["arms"]
    return {"frame": frame, "arms": [REFERENCE, *sorted(arms)], "kinds": kinds}


# --------------------------------------------------------------------------- #
# metric blocks
# --------------------------------------------------------------------------- #


def metrics_block(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Metrics that stay defined for tiny groups (single-district states)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    result = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": np.nan,
        "spearman": np.nan,
    }
    if len(actual) >= 2 and np.std(actual) > 0 and np.std(predicted) > 0:
        result.update(regression_metrics(actual, predicted))
    return result


def tail_block(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "share_under": float(np.mean(residual > 0)),
        "mean_actual": float(np.mean(actual)),
        "mean_predicted": float(np.mean(predicted)),
    }


def flag_block(actual: np.ndarray, predicted: np.ndarray) -> dict:
    flagged = predicted >= P90
    extreme = actual >= P95
    top = actual >= P90
    return {
        "n_flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "sensitivity_p95": float((flagged & extreme).sum() / extreme.sum()),
        "sensitivity_p90": float((flagged & top).sum() / top.sum()),
        "precision_p95": float((flagged & extreme).sum() / flagged.sum()) if flagged.sum() else float("nan"),
    }


def per_fold(frame: pd.DataFrame, arm: str) -> list[dict]:
    rows = []
    for fold_id in sorted(frame["fold"].unique()):
        block = frame[frame["fold"] == fold_id]
        actual, predicted = block["actual"].to_numpy(), block[arm].to_numpy()
        record = {"arm": arm, "fold": fold_id, **metrics_block(actual, predicted)}
        record["n_p90"] = int((actual >= P90).sum())
        record["n_p95"] = int((actual >= P95).sum())
        if record["n_p90"]:
            record.update({f"p90_{k}": v for k, v in tail_block(actual[actual >= P90], predicted[actual >= P90]).items()})
        if record["n_p95"]:
            record.update({f"p95_{k}": v for k, v in tail_block(actual[actual >= P95], predicted[actual >= P95]).items()})
        rows.append(record)
    return rows


def per_state(frame: pd.DataFrame, arm: str) -> list[dict]:
    rows = []
    for state, block in frame.groupby("state"):
        actual, predicted = block["actual"].to_numpy(), block[arm].to_numpy()
        row = {"arm": arm, "state": state, "n": int(len(block)), **metrics_block(actual, predicted)}
        row["bias"] = float(np.mean(actual - predicted))
        row["share_under"] = float(np.mean((actual - predicted) > 0))
        mask = actual >= P90
        row["n_p90"] = int(mask.sum())
        if mask.sum() >= 3:
            row.update({f"p90_{k}": v for k, v in tail_block(actual[mask], predicted[mask]).items()})
        rows.append(row)
    return rows


def error_by_predicted_decile(frame: pd.DataFrame, arm: str) -> list[dict]:
    predicted = frame[arm]
    bins = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    rows = []
    for decile in sorted(pd.Series(bins).dropna().unique()):
        block = frame[bins == decile]
        stats = tail_block(block["actual"].to_numpy(), block[arm].to_numpy())
        rows.append({"arm": arm, "predicted_decile": int(decile) + 1, **stats})
    return rows


# --------------------------------------------------------------------------- #
# quantile diagnostics
# --------------------------------------------------------------------------- #


def quantile_block(frame: pd.DataFrame, arm: str, tau: float) -> tuple[dict, list[dict]]:
    actual = frame["actual"].to_numpy()
    predicted = frame[arm].to_numpy()
    coverage = float(np.mean(actual <= predicted))
    record = {
        "arm": arm,
        "tau": tau,
        "pinball_loss": pinball_loss(actual, predicted, tau),
        "coverage": coverage,
        "calibration_error_pp": (tau - coverage) * 100.0,
        "mean_predicted": float(predicted.mean()),
        "mean_actual": float(actual.mean()),
    }

    # subgroup coverage is DESCRIPTIVE ONLY: a conditional-quantile model is not
    # required to hit nominal coverage inside a group selected on realised target.
    for label, mask in (("p90", actual >= P90), ("p95", actual >= P95)):
        record[f"coverage_{label}_descriptive"] = float(np.mean(actual[mask] <= predicted[mask]))

    bins = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    reliability, weighted_gap, total = [], 0.0, 0
    for decile in sorted(pd.Series(bins).dropna().unique()):
        block = frame[bins == decile]
        if block.empty:
            continue
        realised = float(np.mean(block["actual"].to_numpy() <= block[arm].to_numpy()))
        weight = len(block)
        reliability.append(
            {
                "arm": arm,
                "tau": tau,
                "bin": int(decile) + 1,
                "n": weight,
                "predicted_min": float(block[arm].min()),
                "predicted_max": float(block[arm].max()),
                "realised_exceedance": realised,
                "deviation_pp": (realised - tau) * 100.0,
            }
        )
        weighted_gap += abs(realised - tau) * weight
        total += weight
    record["reliability_mad_pp"] = (weighted_gap / total) * 100.0
    return record, reliability


# --------------------------------------------------------------------------- #
# uncertainty: state-clustered bootstrap
# --------------------------------------------------------------------------- #


def clustered_bootstrap(frame: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    states = frame["state"].to_numpy()
    unique_states = np.unique(states)
    index_by_state = {state: np.flatnonzero(states == state) for state in unique_states}
    actual = frame["actual"].to_numpy()
    reference = frame[REFERENCE].to_numpy()

    quantities = ("mae_all", "bias_p90", "bias_p95", "mae_p95")
    draws = {arm: {q: [] for q in quantities} for arm in arms if arm != REFERENCE}

    for _ in range(BOOTSTRAP_REPS):
        picked = rng.choice(unique_states, size=len(unique_states), replace=True)
        idx = np.concatenate([index_by_state[state] for state in picked])
        y, ref = actual[idx], reference[idx]
        base = {
            "mae_all": float(np.mean(np.abs(y - ref))),
            "bias_p90": float(np.mean((y - ref)[y >= P90])) if (y >= P90).sum() >= 5 else np.nan,
            "bias_p95": float(np.mean((y - ref)[y >= P95])) if (y >= P95).sum() >= 5 else np.nan,
            "mae_p95": float(np.mean(np.abs(y - ref)[y >= P95])) if (y >= P95).sum() >= 5 else np.nan,
        }
        for arm in draws:
            pred = frame[arm].to_numpy()[idx]
            value = {
                "mae_all": float(np.mean(np.abs(y - pred))),
                "bias_p90": float(np.mean((y - pred)[y >= P90])) if (y >= P90).sum() >= 5 else np.nan,
                "bias_p95": float(np.mean((y - pred)[y >= P95])) if (y >= P95).sum() >= 5 else np.nan,
                "mae_p95": float(np.mean(np.abs(y - pred)[y >= P95])) if (y >= P95).sum() >= 5 else np.nan,
            }
            for quantity in quantities:
                draws[arm][quantity].append(value[quantity] - base[quantity])

    rows = []
    for arm, by_quantity in draws.items():
        for quantity, values in by_quantity.items():
            array = np.asarray(values, dtype=float)
            defined = array[np.isfinite(array)]
            point_pred = frame[arm].to_numpy()
            point = {
                "mae_all": float(np.mean(np.abs(actual - point_pred)) - np.mean(np.abs(actual - reference))),
                "bias_p90": float(np.mean((actual - point_pred)[actual >= P90]) - np.mean((actual - reference)[actual >= P90])),
                "bias_p95": float(np.mean((actual - point_pred)[actual >= P95]) - np.mean((actual - reference)[actual >= P95])),
                "mae_p95": float(np.mean(np.abs(actual - point_pred)[actual >= P95]) - np.mean(np.abs(actual - reference)[actual >= P95])),
            }[quantity]
            rows.append(
                {
                    "arm": arm,
                    "quantity": quantity,
                    "delta_vs_E0": point,
                    "ci_low": float(np.percentile(defined, 2.5)) if len(defined) else np.nan,
                    "ci_high": float(np.percentile(defined, 97.5)) if len(defined) else np.nan,
                    "n_effective_reps": int(len(defined)),
                    "cluster_unit": "state",
                    "reps": BOOTSTRAP_REPS,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def evaluate_gates(frame: pd.DataFrame, summary: dict, fold_table: pd.DataFrame) -> pd.DataFrame:
    reference = summary[REFERENCE]
    rows = []

    def add(arm, gate, status, observed, threshold, note):
        rows.append(
            {
                "arm": arm,
                "gate": gate,
                "status": status,
                "observed": observed,
                "threshold": threshold,
                "note": note,
            }
        )

    reference_p90_mae = reference["p90_mae"]
    reference_p95_bias = reference["p95_bias"]
    reference_mae = reference["mae"]
    reference_rmse = reference["rmse"]
    reference_sens = reference["sensitivity_p95"]

    c1_mae_cut = reference_p90_mae * 0.85
    c1_bias_cut = reference_p95_bias * 0.70
    c2_mae_cut = reference_mae * 1.05
    c2_rmse_cut = reference_rmse * 1.05
    c6_sens_cut = reference_sens + 0.10

    tail_states = [s for s in TAIL_STATES if s in set(frame["state"])]
    actual = frame["actual"].to_numpy()

    for arm in summary:
        if arm == REFERENCE:
            continue
        is_quantile = summary[arm]["kind"] == "quantile"
        s = summary[arm]

        if is_quantile:
            add(arm, "C1", "NOT APPLICABLE",
                f"p90 MAE {s['p90_mae']:.3f}; p95 bias {s['p95_bias']:+.3f}",
                f"<= {c1_mae_cut:.3f} / <= {c1_bias_cut:+.3f}",
                "MAE/bias for a conditional mean; this arm estimates a conditional quantile.")
            add(arm, "C2", "NOT APPLICABLE",
                f"MAE {s['mae']:.3f}; RMSE {s['rmse']:.3f}",
                f"<= {c2_mae_cut:.3f} / <= {c2_rmse_cut:.3f}",
                "Structural penalty: a tau-quantile estimator is not a mean predictor.")
        else:
            tail_ok = s["p90_mae"] <= c1_mae_cut and s["p95_bias"] <= c1_bias_cut
            add(arm, "C1", "PASS" if tail_ok else "FAIL",
                f"p90 MAE {s['p90_mae']:.3f}; p95 bias {s['p95_bias']:+.3f}",
                f"<= {c1_mae_cut:.3f} / <= {c1_bias_cut:+.3f}",
                "Both conditions must hold.")
            cost_ok = s["mae"] <= c2_mae_cut and s["rmse"] <= c2_rmse_cut
            add(arm, "C2", "PASS" if cost_ok else "FAIL",
                f"MAE {s['mae']:.3f}; RMSE {s['rmse']:.3f}",
                f"<= {c2_mae_cut:.3f} / <= {c2_rmse_cut:.3f}",
                "At most 5% worse than the reference on both.")

        # C3: same-direction improvement in >=3 of the 4 folds that contain tail data
        fold_rows = fold_table[fold_table["arm"] == arm]
        ref_fold = fold_table[fold_table["arm"] == REFERENCE]
        merged = fold_rows.merge(ref_fold, on="fold", suffixes=("", "_ref"))
        merged = merged[merged["n_p95"] > 0]
        better_mae = int((merged["p90_mae"] < merged["p90_mae_ref"]).sum())
        better_bias = int((merged["p95_bias"] < merged["p95_bias_ref"]).sum())
        c3_ok = better_mae >= 3 and better_bias >= 3 and len(merged) >= 4
        add(arm, "C3", "PASS" if c3_ok else "FAIL",
            f"p90 MAE better in {better_mae}/{len(merged)}; p95 bias better in {better_bias}/{len(merged)}",
            ">= 3 of 4 folds with tail districts",
            "Fold 5 has no district at or above p95 and is excluded.")

        # C4: not one state
        reduction = {}
        for state in tail_states:
            mask = (frame["state"] == state).to_numpy() & (actual >= P90)
            if mask.sum() == 0:
                continue
            reduction[state] = float(
                np.sum(np.abs(frame.loc[mask, "actual"] - frame.loc[mask, REFERENCE]))
                - np.sum(np.abs(frame.loc[mask, "actual"] - frame.loc[mask, arm]))
            )
        positive = {k: v for k, v in reduction.items() if v > 0}
        total_positive = sum(positive.values())
        max_state = max(positive, key=positive.get) if positive else None
        max_share = (positive[max_state] / total_positive) if total_positive > 0 else 1.0
        c4_ok = len(positive) >= 2 and total_positive > 0 and max_share <= 0.60
        detail = "; ".join(f"{k} {v:+.2f}" for k, v in sorted(reduction.items()))
        add(arm, "C4", "PASS" if c4_ok else "FAIL",
            f"{len(positive)} states improved; largest share {max_share:.0%} ({max_state})",
            ">= 2 states and no state > 60% of tail error reduction",
            detail)

        # C5: no over-correction
        c5_ok = s["p95_bias"] >= -1.0
        add(arm, "C5", "PASS" if c5_ok else "FAIL",
            f"p95 bias {s['p95_bias']:+.3f}", ">= -1.000 pp", "Guards against shifting everything upward.")

        # C6: practical read-out
        c6_ok = s["sensitivity_p95"] >= c6_sens_cut and s["precision_p95"] >= 0.40
        add(arm, "C6", "PASS" if c6_ok else "FAIL",
            f"sensitivity {s['sensitivity_p95']:.2f}; precision {s['precision_p95']:.2f}",
            f">= {c6_sens_cut:.2f} and precision >= 0.40",
            "Read-out, not a gate.")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    loaded = load()
    frame, arms = loaded["frame"], loaded["arms"]
    kinds = loaded["kinds"]
    actual = frame["actual"].to_numpy()

    summary: dict[str, dict] = {}
    tail_rows: list[dict] = []
    fold_rows: list[dict] = []
    state_rows: list[dict] = []
    error_rows: list[dict] = []

    for arm in arms:
        predicted = frame[arm].to_numpy()
        pooled = regression_metrics(actual, predicted)
        folds = per_fold(frame, arm)
        fold_rows.extend(folds)
        fold_summary = summarise_folds(
            [{"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"], "spearman": r["spearman"]} for r in folds]
        )
        record = {
            "arm": arm,
            "kind": kinds.get(arm, {}).get("kind", "reference"),
            **pooled,
            **fold_summary,
            **flag_block(actual, predicted),
        }
        for label, mask in (("all", np.ones(len(actual), dtype=bool)), ("p90", actual >= P90), ("p95", actual >= P95)):
            stats = tail_block(actual[mask], predicted[mask])
            record.update({f"{label}_{k}": v for k, v in stats.items()})
            tail_rows.append({"arm": arm, "group": label, **stats})
        record["tradeoff_mae_per_pp_bias_p95"] = (
            (record["mae"] - summary[REFERENCE]["mae"]) / (summary[REFERENCE]["p95_bias"] - record["p95_bias"])
            if arm != REFERENCE and (summary[REFERENCE]["p95_bias"] - record["p95_bias"]) > 1e-9
            else np.nan
        )
        summary[arm] = record
        state_rows.extend(per_state(frame, arm))
        error_rows.extend(error_by_predicted_decile(frame, arm))

    fold_table = pd.DataFrame(fold_rows)
    gate_table = evaluate_gates(frame, summary, fold_table)

    gate_status = {
        arm: {row["gate"]: row["status"] for _, row in gate_table[gate_table["arm"] == arm].iterrows()}
        for arm in arms
    }

    def verdict(arm: str) -> str:
        if arm == REFERENCE:
            return "reference (ratified Phase 3 Ridge, not refitted)"
        status = gate_status[arm]
        if summary[arm]["kind"] == "quantile":
            if status.get("C5") == "PASS" and status.get("C6") == "PASS":
                return "specialised targeting tool (not a mean predictor; no candidate status)"
            return "not accepted"
        if all(status.get(g) == "PASS" for g in ("C1", "C2", "C3", "C4", "C5")):
            return "candidate for separate ratification"
        if status.get("C1") == "PASS" and status.get("C2") == "FAIL":
            return "specialised tail model (overall cost gate failed)"
        if status.get("C1") == "FAIL":
            return "not accepted - tail criterion failed"
        return "not accepted - failed consistency/state/over-correction gate"

    for record in summary.values():
        record["verdict"] = verdict(record["arm"])

    experiments = pd.DataFrame(list(summary.values()))
    experiments.to_csv(METRICS / "tail_experiments.csv", index=False)
    fold_table.to_csv(METRICS / "tail_fold_results.csv", index=False)
    pd.DataFrame(state_rows).to_csv(METRICS / "tail_state_results.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(METRICS / "tail_tail_metrics.csv", index=False)
    pd.DataFrame(error_rows).to_csv(METRICS / "tail_error_by_predicted_decile.csv", index=False)
    gate_table.to_csv(METRICS / "tail_gate_results.csv", index=False)

    quantile_rows, reliability_rows = [], []
    for arm in arms:
        for tau in (0.75, 0.90):
            record, reliability = quantile_block(frame, arm, tau)
            quantile_rows.append(record)
            reliability_rows.extend(reliability)
    pd.DataFrame(quantile_rows).to_csv(METRICS / "tail_quantile_metrics.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(METRICS / "tail_quantile_reliability.csv", index=False)

    bootstrap = clustered_bootstrap(frame, arms)
    bootstrap.to_csv(METRICS / "tail_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    print(experiments[[
        "arm", "mae", "rmse", "r2", "spearman", "p90_mae", "p90_bias",
        "p95_mae", "p95_bias", "sensitivity_p95", "precision_p95", "verdict",
    ]].round(3).to_string(index=False))
    print()
    print(gate_table.pivot(index="arm", columns="gate", values="status").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
