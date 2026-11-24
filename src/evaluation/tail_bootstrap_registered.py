"""Compute the *registered* district-level bootstrap and compare it with the
executed state-clustered one.

Background
----------
The Phase 4 proposal registered a district-level paired bootstrap. The executed
pipeline resampled states instead. Both are legitimate descriptions of
uncertainty and they answer slightly different questions:

* **district-level** answers "how precise is this difference if these 628
  districts are a sample from a wider population of districts?";
* **state-clustered** answers "how precise is it if the unit of independent
  replication is the state?", and is the more conservative choice when tail
  districts are concentrated in a handful of states.

This script adds the registered result; it does not replace the executed one.
``evaluation/metrics/tail_bootstrap.csv`` is **read-only** here and its hashes
are verified before and after the run.

Outputs (new files only)
------------------------
``evaluation/metrics/tail_bootstrap_district.csv``   registered procedure
``evaluation/metrics/tail_bootstrap_comparison.csv`` both units, stacked

Usage: py -3.14 src/evaluation/tail_bootstrap_registered.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap_paired import paired_bootstrap  # noqa: E402
from src.evaluation.tail_evaluation import METRICS, load  # noqa: E402

FROZEN = METRICS / "tail_bootstrap.csv"
TOLERANCE = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not FROZEN.exists():
        raise SystemExit(f"STOP: the frozen bootstrap file is missing: {FROZEN}")

    digest_before = sha256(FROZEN)
    frozen = pd.read_csv(FROZEN)

    loaded = load()
    frame, arms = loaded["frame"], loaded["arms"]

    registered = paired_bootstrap(frame, arms, cluster_unit="district", reps=2000, seed=42)
    executed = paired_bootstrap(frame, arms, cluster_unit="state", reps=2000, seed=42)

    # The refactored state path must reproduce the executed artefact exactly,
    # otherwise this script is describing a different procedure.
    checks = []
    for column in ("delta_vs_E0", "ci_low", "ci_high"):
        left = frozen.sort_values(["arm", "quantity"])[column].to_numpy(dtype=float)
        right = executed.sort_values(["arm", "quantity"])[column].to_numpy(dtype=float)
        checks.append(bool(left.shape == right.shape and max(abs(left - right), default=0.0) <= TOLERANCE))
    if not all(checks):
        diff = (frozen.sort_values(["arm", "quantity"])["ci_low"].to_numpy()
                - executed.sort_values(["arm", "quantity"])["ci_low"].to_numpy())
        raise SystemExit(
            "STOP: the state-clustered path no longer reproduces tail_bootstrap.csv "
            f"(max |delta ci_low| = {max(abs(diff), default=0.0):.3e}). "
            "The comparison would be invalid; investigate before publishing."
        )

    registered.to_csv(METRICS / "tail_bootstrap_district.csv", index=False)
    comparison = pd.concat([executed, registered], ignore_index=True)
    comparison = comparison[["arm", "quantity", "delta_vs_E0", "ci_low", "ci_high", "n_effective_reps", "cluster_unit", "reps"]]
    comparison.to_csv(METRICS / "tail_bootstrap_comparison.csv", index=False)

    digest_after = sha256(FROZEN)
    if digest_before != digest_after:
        raise SystemExit("STOP: tail_bootstrap.csv changed during this script")

    wide = comparison.pivot_table(
        index=["arm", "quantity"], columns="cluster_unit", values=["ci_low", "ci_high"]
    )
    pd.set_option("display.width", 200)
    print("Frozen state-clustered results reproduced exactly:", all(checks))
    print(f"tail_bootstrap.csv sha256 unchanged: {digest_after}")
    print()
    print("Registered district-level intervals (2000 reps, seed 42):")
    print(registered.pivot(index="arm", columns="quantity", values=["ci_low", "ci_high"]).round(3).to_string())
    print()
    print("Width of the 95% interval, district vs state (pp):")
    width = pd.DataFrame(
        {
            "district": registered["ci_high"] - registered["ci_low"],
            "state": executed["ci_high"] - executed["ci_low"],
        }
    )
    width["arm"] = registered["arm"].to_numpy()
    width["quantity"] = registered["quantity"].to_numpy()
    print(width.pivot(index="arm", columns="quantity", values=["district", "state"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
