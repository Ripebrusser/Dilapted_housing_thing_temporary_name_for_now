"""Phase 4 regression tests added in response to audit defects D-07, D-08, D-01.

Two blocks, both required by the independent Phase 4 audit:

**A. Quantile loss and coverage**
    Pinball loss is checked against hand-computed values at several quantiles,
    including over-prediction, exact hits and ties. The direction of the loss is
    pinned down by an asymmetry assertion, so an implementation that swapped
    ``y`` and ``prediction`` (or flipped the inequality) fails rather than passes.
    Stored coverage is recomputed from the frozen predictions.

**B. Bootstrap reproducibility**
    A fixed seed must reproduce the intervals exactly; resampling must be paired
    across arms; the executed state-clustered path must still reproduce
    ``tail_bootstrap.csv``; and the registered district-level path must not be
    silently interchangeable with it. Interval constructions are verified
    against their defining mathematical properties rather than by restating the
    formula that produced them.

Run: py -3.14 tests/test_phase4_quantile_and_bootstrap.py
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap_paired import CLUSTER_UNITS, paired_bootstrap  # noqa: E402
from src.evaluation.tail_coverage_intervals import jeffreys_interval, wilson_interval  # noqa: E402
from src.evaluation.tail_evaluation import METRICS, load  # noqa: E402
from src.models.tail_aware import pinball_loss  # noqa: E402

FROZEN = {
    ("E4a", 0.75): (0.7464610718963143, 0.6703821656050956),
    ("E4a", 0.90): (0.6200913467317561, 0.6703821656050956),
    ("E4b", 0.75): (0.8841730081330276, 0.8248407643312102),
    ("E4b", 0.90): (0.5411374198472226, 0.8248407643312102),
}
TOL = 1e-12


@lru_cache(maxsize=1)
def frame_and_arms():
    loaded = load()
    return loaded["frame"], loaded["arms"]


# --------------------------------------------------------------------------- #
# A. quantile loss
# --------------------------------------------------------------------------- #


def test_pinball_loss_matches_hand_computed_values():
    # residual = y - p; loss = max(tau * r, (tau - 1) * r)
    cases = [
        # y,   p,   tau,  expected
        (10.0, 8.0, 0.90, 1.80),   # under-prediction: tau * 2
        (10.0, 8.0, 0.75, 1.50),
        (10.0, 8.0, 0.50, 1.00),
        (10.0, 8.0, 0.10, 0.20),   # under-prediction costs little at tau = 0.1
        (10.0, 12.0, 0.90, 0.20),  # over-prediction: (1 - tau) * 2
        (10.0, 12.0, 0.10, 1.80),
        (10.0, 10.0, 0.90, 0.00),  # exact hit is free at every tau
        (10.0, 10.0, 0.10, 0.00),
    ]
    for y, p, tau, expected in cases:
        observed = pinball_loss(np.array([y]), np.array([p]), tau)
        assert abs(observed - expected) < TOL, (y, p, tau, observed, expected)


def test_pinball_loss_is_a_mean_over_districts():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([8.0, 22.0, 30.0])  # residuals +2, -2, 0
    expected = (1.0 + 1.0 + 0.0) / 3.0  # at tau = 0.5 both directions cost |r|/2
    assert abs(pinball_loss(y, p, 0.5) - expected) < TOL
    # at tau = 0.9 the under-prediction dominates: 0.9*2, 0.1*2, 0
    assert abs(pinball_loss(y, p, 0.9) - (1.8 + 0.2 + 0.0) / 3.0) < TOL


def test_pinball_loss_direction_would_fail_if_reversed():
    """Ties the loss to the quantile it claims to estimate.

    ``pinball(y, p, tau) == pinball(p, y, 1 - tau)`` must hold; a reversal of the
    inequality or of the arguments breaks it. The two asymmetric values below
    are different, so an implementation that ignored direction cannot pass.
    """
    y, p, tau = 10.0, 8.0, 0.9
    forward = pinball_loss(np.array([y]), np.array([p]), tau)
    swapped_args = pinball_loss(np.array([p]), np.array([y]), tau)
    swapped_tau = pinball_loss(np.array([p]), np.array([y]), 1.0 - tau)
    assert abs(forward - 1.8) < TOL
    assert abs(swapped_args - 0.2) < TOL, "direction is not being respected"
    assert forward != swapped_args
    assert abs(forward - swapped_tau) < TOL, "loss must be symmetric under (y<->p, tau->1-tau)"


def test_coverage_is_recomputed_from_frozen_predictions():
    frame, _ = frame_and_arms()
    actual = frame["actual"].to_numpy(dtype=float)
    for (arm, tau), (loss, coverage) in FROZEN.items():
        predicted = frame[arm].to_numpy(dtype=float)
        recomputed = float(np.mean(actual <= predicted))
        assert abs(recomputed - coverage) < TOL, (arm, tau, recomputed, coverage)
        assert abs(pinball_loss(actual, predicted, tau) - loss) < TOL, (arm, tau)
        # calibration error is (tau - coverage) in percentage points, sign included
        stored = pd.read_csv(METRICS / "tail_quantile_metrics.csv")
        row = stored[(stored["arm"] == arm) & (stored["tau"] == tau)].iloc[0]
        assert abs(row["calibration_error_pp"] - (tau - coverage) * 100.0) < 1e-9
        assert (row["calibration_error_pp"] > 0) == (coverage < tau)


def test_coverage_counts_ties_as_covered_and_is_bounded():
    # a tie is covered: coverage uses y <= prediction
    assert float(np.mean(np.array([1.0, 2.0]) <= np.array([1.0, 2.0]))) == 1.0
    # boundary behaviour of the exceedance fraction
    assert float(np.mean(np.array([5.0]) <= np.array([4.0]))) == 0.0
    assert float(np.mean(np.array([5.0]) <= np.array([6.0]))) == 1.0
    frame, _ = frame_and_arms()
    actual = frame["actual"].to_numpy(dtype=float)
    for arm in ("E0", "E1", "E2", "E3", "E4a", "E4b"):
        coverage = float(np.mean(actual <= frame[arm].to_numpy(dtype=float)))
        assert 0.0 <= coverage <= 1.0, (arm, coverage)


# --------------------------------------------------------------------------- #
# B. bootstrap reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_reproduces_the_registered_intervals_exactly():
    frame, arms = frame_and_arms()
    first = paired_bootstrap(frame, arms, cluster_unit="district", reps=200, seed=7)
    second = paired_bootstrap(frame, arms, cluster_unit="district", reps=200, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert (first["cluster_unit"] == "district").all()


def test_district_bootstrap_is_reproducible_at_the_registered_settings():
    """The registered procedure itself: 2000 reps, seed 42, twice."""
    frame, arms = frame_and_arms()
    first = paired_bootstrap(frame, arms, cluster_unit="district", reps=2000, seed=42)
    second = paired_bootstrap(frame, arms, cluster_unit="district", reps=2000, seed=42)
    assert len(first) == len(second) == len(arms[:-1] if arms[0] == "E0" else arms) * 4
    assert np.array_equal(first["ci_low"].to_numpy(), second["ci_low"].to_numpy())
    assert np.array_equal(first["ci_high"].to_numpy(), second["ci_high"].to_numpy())
    assert (first["reps"] == 2000).all()
    # a different seed must move the interval, otherwise nothing is being resampled
    other = paired_bootstrap(frame, arms, cluster_unit="district", reps=2000, seed=1234)
    assert not np.allclose(first["ci_low"].to_numpy(), other["ci_low"].to_numpy())


def test_resampling_is_paired_across_arms():
    """Two identical arms must differ by exactly zero in every replicate.

    Under paired resampling the same rows are used for both arms, so an
    identical arm has delta 0 by construction and the interval collapses to a
    point. Under unpaired resampling the two arms would be drawn independently
    and the deltas would scatter around zero. Point-collapse is therefore a
    direct, asymmetric test of pairing — it cannot pass by accident.
    """
    rng = np.random.default_rng(0)
    n = 40
    actual = rng.normal(5.0, 2.0, n)
    frame = pd.DataFrame(
        {
            "state": [f"S{i % 8}" for i in range(n)],
            "actual": actual,
            "E0": actual - rng.normal(0.0, 0.5, n),
        }
    )
    frame["E1"] = frame["E0"]  # identical arm

    for unit in CLUSTER_UNITS:
        result = paired_bootstrap(frame, ["E0", "E1"], cluster_unit=unit, reps=300, seed=11)
        for quantity in ("mae_all", "bias_p90", "bias_p95", "mae_p95"):
            row = result[result["quantity"] == quantity].iloc[0]
            if not np.isfinite(row["ci_low"]):
                continue
            assert abs(row["delta_vs_E0"]) < 1e-12, (unit, quantity, row["delta_vs_E0"])
            assert row["ci_low"] == 0.0 and row["ci_high"] == 0.0, (
                unit, quantity, row["ci_low"], row["ci_high"])

    # An independently constructed example with a knowable answer: if arm 1 is
    # uniformly 1.0 worse than the reference, its delta MAE is exactly +1.0 in
    # every replicate, so the interval must be the single point [1, 1].
    tiny = pd.DataFrame(
        {
            "state": ["A", "A", "B", "B"],
            "actual": [1.0, 2.0, 3.0, 4.0],
            "E0": [1.0, 2.0, 3.0, 4.0],
            "E1": [2.0, 3.0, 4.0, 5.0],
        }
    )
    one = paired_bootstrap(tiny, ["E0", "E1"], cluster_unit="district", reps=25, seed=0)
    mae = one[one["quantity"] == "mae_all"].iloc[0]
    assert abs(mae["delta_vs_E0"] - 1.0) < 1e-12, mae["delta_vs_E0"]
    assert abs(mae["ci_low"] - 1.0) < 1e-12 and abs(mae["ci_high"] - 1.0) < 1e-12


def test_state_path_still_reproduces_the_executed_artefact():
    """The refactored state path must equal the frozen Phase 4 results."""
    frozen = pd.read_csv(METRICS / "tail_bootstrap.csv").sort_values(["arm", "quantity"])
    frame, arms = frame_and_arms()
    executed = paired_bootstrap(frame, arms, cluster_unit="state", reps=2000, seed=42)
    executed = executed.sort_values(["arm", "quantity"])
    assert len(frozen) == len(executed) == 20
    for column in ("delta_vs_E0", "ci_low", "ci_high"):
        difference = np.abs(frozen[column].to_numpy() - executed[column].to_numpy())
        assert difference.max() < 1e-9, (column, difference.max())
    assert (executed["cluster_unit"] == "state").all()


def test_cluster_units_are_not_accidentally_mixed():
    frame, arms = frame_and_arms()
    district = paired_bootstrap(frame, arms, cluster_unit="district", reps=300, seed=5)
    state = paired_bootstrap(frame, arms, cluster_unit="state", reps=300, seed=5)
    assert set(district["cluster_unit"]) == {"district"}
    assert set(state["cluster_unit"]) == {"state"}
    # the two procedures give different intervals; they are not interchangeable
    assert not np.allclose(district["ci_low"].to_numpy(), state["ci_low"].to_numpy())
    for bad in ("states", "", "district-level"):
        try:
            paired_bootstrap(frame, arms, cluster_unit=bad, reps=10, seed=5)
        except ValueError:
            continue
        raise AssertionError(f"cluster_unit={bad!r} was silently accepted")


def test_wilson_interval_satisfies_its_defining_score_equation():
    """Independently of the algebra: the endpoints are where the score equals +/- z."""
    z = 1.959963984540054
    for covered, n in ((4, 35), (11, 64), (516, 628), (0, 35), (35, 35), (7, 64)):
        low, high = wilson_interval(covered, n, z)
        p = covered / n
        assert 0.0 <= low <= p + 1e-12 and p - 1e-12 <= high <= 1.0, (covered, n, low, p, high)
        if 0 < covered < n:
            score_low = (p - low) / np.sqrt(low * (1 - low) / n)
            score_high = (p - high) / np.sqrt(high * (1 - high) / n)
            assert abs(score_low - z) < 1e-9, (covered, n, score_low)
            assert abs(score_high + z) < 1e-9, (covered, n, score_high)


def test_jeffreys_interval_matches_its_beta_quantiles():
    for covered, n in ((4, 35), (11, 64), (516, 628)):
        low, high = jeffreys_interval(covered, n)
        assert abs(beta.cdf(low, covered + 0.5, n - covered + 0.5) - 0.025) < 1e-9
        assert abs(beta.cdf(high, covered + 0.5, n - covered + 0.5) - 0.975) < 1e-9
        assert low < covered / n < high


def test_small_tail_subgroup_interval_is_reported_and_wide():
    table = pd.read_csv(METRICS / "tail_quantile_coverage_ci.csv")
    p95 = table[table["scope"] == "p95_descriptive"]
    assert len(p95) == 10, len(p95)          # 5 arms x 2 quantiles
    assert (p95["n"] == 35).all(), p95["n"].unique()
    assert p95["small_sample"].all()
    # smallest observed p95 width is ~0.14 (1 of 35 covered); assert the real
    # property rather than a convenient round number
    assert (p95["wilson_width"] > 0.13).all(), p95["wilson_width"].min()
    overall = table[table["scope"] == "all"]
    assert (overall["n"] == 628).all()
    assert (overall["wilson_width"] < 0.10).all()
    # the same arm's 35-district interval is materially wider than its 628-district one
    # (observed ratio 1.8-3.6 across arms, so 1.7 is the honest lower bound)
    ratios = []
    for arm in p95["arm"].unique():
        for tau in (0.75, 0.90):
            tail_width = p95[(p95["arm"] == arm) & (p95["tau"] == tau)]["wilson_width"].iloc[0]
            overall_width = overall[(overall["arm"] == arm) & (overall["tau"] == tau)]["wilson_width"].iloc[0]
            assert tail_width > 1.7 * overall_width, (arm, tau, tail_width, overall_width)
            ratios.append(tail_width / overall_width)
    assert float(np.median(ratios)) > 2.5, float(np.median(ratios))
    # every interval must contain its own point estimate
    assert ((table["wilson_low"] <= table["coverage"]) & (table["coverage"] <= table["wilson_high"])).all()
    assert ((table["jeffreys_low"] <= table["coverage"]) & (table["coverage"] <= table["jeffreys_high"])).all()


def test_coverage_interval_table_matches_recomputation():
    frame, _ = frame_and_arms()
    actual = frame["actual"].to_numpy(dtype=float)
    table = pd.read_csv(METRICS / "tail_quantile_coverage_ci.csv")
    for _, row in table.iterrows():
        if row["scope"] == "all":
            mask = np.ones(len(actual), dtype=bool)
        elif row["scope"] == "p90_descriptive":
            mask = actual >= 9.200
        else:
            mask = actual >= 11.200
        predicted = frame[row["arm"]].to_numpy(dtype=float)
        covered = int(np.sum(actual[mask] <= predicted[mask]))
        assert row["n"] == int(mask.sum()), (row["arm"], row["scope"], row["n"])
        assert row["n_covered"] == covered, (row["arm"], row["scope"], row["n_covered"], covered)
        assert abs(row["coverage"] - covered / int(mask.sum())) < TOL


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")

    passed = len(tests) - failures
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
