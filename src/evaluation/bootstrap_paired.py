"""Paired bootstrap for Phase 4 uncertainty, with an explicit cluster unit.

Why this module exists
----------------------
``reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md`` registers a
**district-level** bootstrap: resample the 628 districts with replacement,
recompute each arm's metric on the resample, and compare it with the reference
arm on the *same* resample (paired). The executed Phase 4 pipeline
(``src/evaluation/tail_evaluation.py::clustered_bootstrap``) instead resampled
**states**, on the argument that tail districts cluster within a few states and
a district-level bootstrap would understate that dependence.

That is a real methodological deviation, and it is recorded as such (D-01 /
audit defect D-05). This module supplies the registered procedure *separately*
so both can be reported, without touching the executed results.

Contract
--------
* ``cluster_unit="district"`` — the registered procedure. Resamples districts.
* ``cluster_unit="state"`` — the executed procedure. Called with seed 42 it
  reproduces ``evaluation/metrics/tail_bootstrap.csv`` exactly; the regression
  test asserts that, so the executed numbers stay verifiable.
* Arms are always compared with the reference on identical resampled rows
  (paired), never on independent draws.
* Quantities whose subgroup is too small (fewer than ``min_tail_group`` tail
  districts in the resample) are dropped for that replicate and counted in
  ``n_effective_reps``, exactly as the executed code does.

Nothing here fits a model or changes a gate decision. It is a description of
uncertainty around frozen out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

P90 = 9.200
P95 = 11.200
REFERENCE = "E0"
QUANTITIES = ("mae_all", "bias_p90", "bias_p95", "mae_p95")
CLUSTER_UNITS = ("district", "state")


def _quantity_block(actual: np.ndarray, predicted: np.ndarray, min_tail_group: int) -> dict:
    """The four compared quantities, NaN where the tail subgroup is too small."""
    residual = actual - predicted
    top90 = actual >= P90
    top95 = actual >= P95
    return {
        "mae_all": float(np.mean(np.abs(residual))),
        "bias_p90": float(np.mean(residual[top90])) if top90.sum() >= min_tail_group else np.nan,
        "bias_p95": float(np.mean(residual[top95])) if top95.sum() >= min_tail_group else np.nan,
        "mae_p95": float(np.mean(np.abs(residual[top95]))) if top95.sum() >= min_tail_group else np.nan,
    }


def _resample_index(frame: pd.DataFrame, cluster_unit: str, rng: np.random.Generator) -> np.ndarray:
    """Draw one replicate's row indices for the requested cluster unit.

    Both branches draw from the same generator, so a single seed fixes the whole
    procedure. The state branch mirrors the executed code's draw exactly
    (choose states, then concatenate their member rows), which is what makes the
    executed results reproducible here.
    """
    if cluster_unit == "district":
        n_rows = len(frame)
        return rng.choice(np.arange(n_rows), size=n_rows, replace=True)
    if cluster_unit == "state":
        states = frame["state"].to_numpy()
        unique = np.unique(states)
        index_by_state = {state: np.flatnonzero(states == state) for state in unique}
        picked = rng.choice(unique, size=len(unique), replace=True)
        return np.concatenate([index_by_state[state] for state in picked])
    raise ValueError(f"cluster_unit must be one of {CLUSTER_UNITS}, got {cluster_unit!r}")


def paired_bootstrap(
    frame: pd.DataFrame,
    arms: list[str],
    reference: str = REFERENCE,
    cluster_unit: str = "district",
    reps: int = 2000,
    seed: int = 42,
    min_tail_group: int = 5,
) -> pd.DataFrame:
    """Percentile bootstrap of each arm's difference from ``reference``.

    Parameters
    ----------
    frame
        One row per district, with ``actual``, ``state`` and one column per arm.
    arms
        Arm columns to compare; ``reference`` must be among them.
    cluster_unit
        ``"district"`` (registered) or ``"state"`` (executed).
    reps, seed
        Registered values are 2000 and 42.
    min_tail_group
        Minimum tail-district count for a tail quantity to be defined in a
        replicate; 5, as executed.

    Returns
    -------
    DataFrame with ``arm, quantity, delta_vs_E0, ci_low, ci_high,
    n_effective_reps, cluster_unit, reps`` — the schema of the frozen
    ``tail_bootstrap.csv`` so the two can be stacked directly.
    """
    if cluster_unit not in CLUSTER_UNITS:
        raise ValueError(f"cluster_unit must be one of {CLUSTER_UNITS}, got {cluster_unit!r}")
    if reference not in arms:
        raise ValueError(f"reference arm {reference!r} is not in arms {arms!r}")
    if reps < 1:
        raise ValueError("reps must be at least 1")

    missing = {"actual", "state", *arms} - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    if frame[["actual", *arms]].isna().any().any():
        raise ValueError("frame contains missing predictions; the bootstrap needs complete rows")

    compared = [arm for arm in arms if arm != reference]
    actual = frame["actual"].to_numpy(dtype=float)
    predictions = {arm: frame[arm].to_numpy(dtype=float) for arm in arms}

    point_reference = _quantity_block(actual, predictions[reference], min_tail_group)
    point = {arm: _quantity_block(actual, predictions[arm], min_tail_group) for arm in compared}

    rng = np.random.default_rng(seed)
    draws = {arm: {quantity: [] for quantity in QUANTITIES} for arm in compared}

    for _ in range(reps):
        idx = _resample_index(frame, cluster_unit, rng)
        y = actual[idx]
        base = _quantity_block(y, predictions[reference][idx], min_tail_group)
        for arm in compared:
            value = _quantity_block(y, predictions[arm][idx], min_tail_group)
            for quantity in QUANTITIES:
                draws[arm][quantity].append(value[quantity] - base[quantity])

    rows = []
    for arm in compared:
        for quantity in QUANTITIES:
            array = np.asarray(draws[arm][quantity], dtype=float)
            defined = array[np.isfinite(array)]
            rows.append(
                {
                    "arm": arm,
                    "quantity": quantity,
                    "delta_vs_E0": point[arm][quantity] - point_reference[quantity],
                    "ci_low": float(np.percentile(defined, 2.5)) if len(defined) else np.nan,
                    "ci_high": float(np.percentile(defined, 97.5)) if len(defined) else np.nan,
                    "n_effective_reps": int(len(defined)),
                    "cluster_unit": cluster_unit,
                    "reps": int(reps),
                }
            )
    return pd.DataFrame(rows)
