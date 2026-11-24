"""Binomial uncertainty intervals for the Phase 4 quantile-coverage results.

Coverage of a nominal ``tau`` quantile is a proportion estimated from
``n`` districts, so it carries sampling uncertainty that a bare percentage
hides. With ``n = 628`` the overall interval is narrow; inside the tail
subgroups it is wide — the p95 subgroup has only 35 districts, so a coverage
of 11% has an interval spanning roughly 5% to 24%. Reporting the point estimate
alone would invite over-reading.

Two intervals are reported for every cell:

* **Wilson score** — the standard interval for a binomial proportion; behaves
  well near 0 and 1, unlike the Wald interval.
* **Jeffreys** — a Beta(k + 1/2, n - k + 1/2) equal-tailed credible interval
  with a uniform prior; included as a sensitivity check on the method choice.

Neither interval is a calibration guarantee. Subgroup coverage is computed on
districts selected by their *realised* target value, which is descriptive: a
conditional-quantile model is not required to hit nominal coverage inside a
group chosen that way.

Reads the frozen predictions and the frozen quantile table; writes one new CSV.
Does not modify any Phase 4 artefact.

Usage: py -3.14 src/evaluation/tail_coverage_intervals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.tail_evaluation import METRICS, P90, P95, load  # noqa: E402

Z_95 = 1.959963984540054
ALPHA = 0.05
TOLERANCE = 1e-9


def wilson_interval(covered: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity correction)."""
    if n <= 0:
        return float("nan"), float("nan")
    if not 0 <= covered <= n:
        raise ValueError(f"covered must lie in [0, {n}], got {covered}")
    # exact boundaries: the score interval degenerates to the edge, and the
    # floating-point form can return +/-1e-17 instead of a clean 0 or 1
    if covered == 0:
        return 0.0, float(min(1.0, z * z / (n + z * z)))
    if covered == n:
        return float(max(0.0, n / (n + z * z))), 1.0
    p = covered / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def jeffreys_interval(covered: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Equal-tailed Jeffreys interval, Beta(k + 1/2, n - k + 1/2)."""
    if n <= 0:
        return float("nan"), float("nan")
    lower = beta.ppf(alpha / 2.0, covered + 0.5, n - covered + 0.5)
    upper = beta.ppf(1.0 - alpha / 2.0, covered + 0.5, n - covered + 0.5)
    return float(lower), float(upper)


def main() -> int:
    loaded = load()
    frame = loaded["frame"]
    arms = [arm for arm in loaded["arms"] if arm != "E0"]
    frozen = pd.read_csv(METRICS / "tail_quantile_metrics.csv")

    actual = frame["actual"].to_numpy(dtype=float)
    scopes = {
        "all": np.ones(len(actual), dtype=bool),
        "p90_descriptive": actual >= P90,
        "p95_descriptive": actual >= P95,
    }

    rows = []
    for arm in arms:
        predicted = frame[arm].to_numpy(dtype=float)
        for tau in (0.75, 0.90):
            for scope, mask in scopes.items():
                n = int(mask.sum())
                covered = int(np.sum(actual[mask] <= predicted[mask]))
                coverage = covered / n
                record = frozen[(frozen["arm"] == arm) & (frozen["tau"] == tau)].iloc[0]
                if scope == "all":
                    stored = float(record["coverage"])
                else:
                    stored = float(record[f"coverage_{scope.split('_')[0]}_descriptive"])
                if abs(stored - coverage) > TOLERANCE:
                    raise SystemExit(
                        f"STOP: recomputed coverage for {arm} tau={tau} scope={scope} "
                        f"({coverage:.12f}) disagrees with the frozen table ({stored:.12f})"
                    )
                low_w, high_w = wilson_interval(covered, n)
                low_j, high_j = jeffreys_interval(covered, n)
                rows.append(
                    {
                        "arm": arm,
                        "tau": tau,
                        "scope": scope,
                        "n": n,
                        "n_covered": covered,
                        "coverage": coverage,
                        "wilson_low": low_w,
                        "wilson_high": high_w,
                        "jeffreys_low": low_j,
                        "jeffreys_high": high_j,
                        "wilson_width": high_w - low_w,
                        "small_sample": n < 100,
                        "note": (
                            "descriptive subgroup: districts selected by realised target"
                            if scope != "all"
                            else "overall coverage across all 628 districts"
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "tail_quantile_coverage_ci.csv", index=False)

    pd.set_option("display.width", 200)
    shown = table.copy()
    for column in ("coverage", "wilson_low", "wilson_high", "jeffreys_low", "jeffreys_high"):
        shown[column] = (shown[column] * 100).round(1)
    print("Coverage with 95% binomial intervals (percent):")
    print(shown[["arm", "tau", "scope", "n", "n_covered", "coverage",
                 "wilson_low", "wilson_high", "jeffreys_low", "jeffreys_high"]].to_string(index=False))
    print()
    print("Cells with n < 100 (interval must be read as wide):",
          f"{int(table['small_sample'].sum())} of {len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
