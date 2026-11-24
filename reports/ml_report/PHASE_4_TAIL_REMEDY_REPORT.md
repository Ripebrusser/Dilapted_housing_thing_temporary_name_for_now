# Phase 4 — Extreme-District Tail Investigation: Experimental Report

**Status:** executed to completion under the authorised scope.
**Scope:** exactly the five registered arms in `reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md`.
**Ratified Phase 3 model roles:** unchanged. Ridge on Configuration B remains the primary interpretable model; Elastic Net secondary; SVR-RBF benchmark. **No challenger is promoted.**
**Phase 5:** not started.

---

## 1. What was implemented

| File | Purpose |
| --- | --- |
| `src/models/tail_aware.py` | The five registered arms, the weight construction, a hand-written inner search (needed because `GridSearchCV` cannot recompute weights per inner split), per-fold fitting, and `pinball_loss` |
| `src/training/run_tail_experiments.py` | Preflight boundary checks, resumable runner, fit-level audit trail, OOF predictions, selected parameters, runtime |
| `src/evaluation/tail_evaluation.py` | Overall, tail, fold-level and state-level metrics; frozen-threshold flag metrics; quantile pinball/coverage/calibration; C1–C6 gates; state-clustered bootstrap |
| `tests/test_phase4_isolation.py` | The three mandated leakage checks plus registration, routing, determinism, fold-identity and frozen-definition tests |

Nothing in `src/models/registry.py`, the Phase 3 trainer, the Phase 2 feature pipeline or any frozen Phase 3 output was modified.

**Registered arms (unchanged from the proposal):**

| Arm | Definition | Selection grid |
| --- | --- | --- |
| E0 | Ratified Phase 3 Ridge OOF predictions, reused verbatim | — (not refitted) |
| E1 | `w = 1 + 2·1[y ≥ q75(train)]`, normalised to mean 1 | α ∈ {30, 100, 300} |
| E2 | `w = 1 + 4·1[y ≥ q75(train)]` | α ∈ {30, 100, 300} |
| E3 | `w = 1 + 4·1[y ≥ q90(train)]` | α ∈ {30, 100, 300} |
| E4a | Linear quantile regression, τ = 0.75 | L1 α ∈ {0.001, 0.01, 0.1} |
| E4b | Linear quantile regression, τ = 0.90 | L1 α ∈ {0.001, 0.01, 0.1} |

## 2. What was actually executed

- **Data:** Configuration B, 628 districts × 77 predictors, approved target `target_deprivation_rate_pct`.
- **Folds:** the exact Phase 3 grouped folds (5 outer, 4 inner `GroupKFold` on held-out-state-free training rows). Verified by reproducing the frozen Phase 3 Ridge out-of-fold predictions to < 1e-8.
- **Frozen evaluation thresholds:** p90 = 9.200 (64 districts), p95 = 11.200 (35 districts) — asserted at runtime in both the runner and the evaluation.
- **Seed:** 42. No randomness is used by the weighted arms; the bootstrap is seeded.
- **Fits:** 325 outer-and-inner fits.

| Arm | Selected α per fold | Outer-refit threshold / weighted rows | Runtime |
| --- | --- | --- | --- |
| E1 | 100, 30, 100, 300, 300 | q75 = 6.250–7.100; 126–129 of ~502 training rows | 0.37 s |
| E2 | 100, 30, 100, 300, 100 | q75 same as E1 (identical thresholds; only λ differs) | 0.34 s |
| E3 | 300, 300, 100, 300, 300 | q90 = 8.70–9.68; 51–54 training rows | 0.43 s |
| E4a | 0.01 in all five folds | — | 4.33 s |
| E4b | 0.01, 0.01, 0.1, 0.01, 0.01 | — | 3.56 s |

Total wall time **11.7 s**; evaluation including the 2,000-repetition bootstrap ≈ 3 s.

## 3. Verification

| Check | Result |
| --- | --- |
| Registered arms/grids exactly the five proposed | PASS |
| Weight values match registration (0.667/2.0, 0.5/2.5, 0.714/3.571) | PASS |
| Weights normalised to mean 1, positive, monotone at the threshold | PASS |
| Weight routing: weighted pipeline equals a direct weighted `Ridge` on the same design | PASS |
| Inner-isolation: corrupting an inner split's validation targets leaves that split's threshold, weight count and weight mean unchanged | PASS |
| Outer-isolation: corrupting the outer test targets leaves α, threshold, weights, coefficients and predictions bit-identical | PASS |
| Outer folds reproduce the frozen Phase 3 Ridge (< 1e-8) | PASS |
| Frozen thresholds and per-fold extreme counts (11/20/21/10/2 and 4/12/14/5/0) | PASS |
| Determinism of the inner search | PASS |
| **Total** | **9/9** |

Reproducibility: the entire pipeline was re-run from scratch. All nine artefacts — OOF predictions, diagnostics, experiments, folds, states, gates, quantile metrics, bootstrap, and the parameter block — were **byte-identical** (sha256).

## 4. What the results demonstrate

### 4.1 Reference and overall performance

| Arm | MAE | Fold MAE (mean ± SD) | RMSE | R² | Spearman | Δ MAE vs E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **E0** (ratified Ridge) | **1.750** | 1.749 ± 0.177 | **2.284** | **0.468** | **0.667** | — |
| E1 | 1.915 | 1.915 ± 0.181 | 2.431 | 0.398 | 0.615 | +9.5% |
| E2 | 1.975 | 1.975 ± 0.236 | 2.525 | 0.350 | 0.602 | +12.9% |
| E3 | 1.985 | 1.984 ± 0.271 | 2.531 | 0.347 | 0.631 | +13.5% |
| E4a (τ = 0.75) | 1.914 | 1.914 ± 0.226 | 2.476 | 0.375 | 0.672 | +9.4% |
| E4b (τ = 0.90) | 2.912 | 2.912 ± 0.556 | 3.574 | −0.302 | 0.545 | +66.4% |

Every arm is worse than the ratified Ridge on overall accuracy, and **not in one or two folds**: overall MAE is worse in all five folds for E1, E2, E3 and E4b, and in four of five for E4a (fold 2 is the single exception, 1.8976 vs 1.9111 for E0). Fold-2 values for the weighted arms are 1.9396 (E1), 1.9254 (E2) and 1.9113 (E3) against 1.9111 for the reference — E3 is worse by 0.0002 pp, i.e. effectively a tie.

### 4.2 The tail did move — consistently, but not far enough

| Arm | p90 MAE (n=64) | p90 bias | p95 MAE (n=35) | p95 bias | p95 bias reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 | 3.648 | +3.593 | 4.527 | **+4.510** | — |
| E1 | 3.032 | +2.935 | 3.786 | +3.693 | −18.1% |
| E2 | 2.831 | +2.724 | 3.545 | +3.435 | −23.8% |
| E3 | 2.807 | +2.501 | 3.530 | +3.271 | −27.5% |
| E4a | 2.624 | +2.303 | 3.431 | +3.109 | −31.1% |
| E4b | 2.331 | +1.827 | 3.102 | +2.994 | −33.6% |

The direction is uniform and the fold-level evidence is strong: for E1–E4a the p90 MAE **and** the p95 bias improve in **4 of 4** folds that contain tail districts; E4b improves in 3 of 4 (fold 3 p95 bias 4.467 vs 3.028 is worse).

The magnitude is the problem: the registered C1 bar was a **30%** reduction in p95 bias (≤ +3.157). The weighted arms reached 18–27% and fell short; the quantile arms crossed it, but their MAE-based gates are not applicable to a conditional-quantile estimator.

### 4.3 The cost is systematic, and the bootstrap supports it

State-clustered bootstrap (2,000 resamples of the 35 states), Δ versus E0:

| Arm | Δ overall MAE [95% CI] | Δ p95 bias [95% CI] | MAE cost per 1 pp of tail-bias reduction |
| --- | --- | --- | ---: |
| E1 | +0.165 [+0.022, +0.327] | −0.817 [−1.077, −0.487] | 0.20 |
| E2 | +0.225 [+0.041, +0.428] | −1.074 [−1.455, −0.722] | 0.21 |
| E3 | +0.235 [+0.055, +0.457] | −1.239 [−1.508, −0.704] | 0.19 |
| E4a | +0.165 [−0.023, +0.374] | −1.401 [−1.695, −1.068] | 0.12 |
| E4b | +1.162 [+0.718, +1.653] | −1.516 [−3.782, +1.251] | 0.77 |

Both effects are real for E1–E3: the tail improvement and the overall cost both exclude zero. E4a is the only arm whose overall cost interval includes zero — the most efficient trade, though still a ~9% point estimate. E4b's tail interval spans zero: its apparent gain is not stable.

So this is not a case of "the tail was untouched". It is a **trade**: roughly 0.12–0.21 pp of overall MAE per 1 pp of tail bias recovered.

### 4.4 State dependence

Absolute-error reduction in the tail (≥ p90), by state:

| Arm | Assam | West Bengal | Uttar Pradesh | Orissa | Largest share |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | +19.89 | +7.04 | +4.91 | +3.48 | 56% (Assam) |
| E2 | +26.41 | +8.30 | +6.10 | +5.56 | 57% (Assam) |
| E3 | +23.12 | +9.83 | +6.63 | +7.47 | 49% (Assam) |
| E4a | +21.16 | +9.22 | +19.36 | +8.07 | 37% (Assam) |
| E4b | +59.97 | **−23.40** | +25.27 | +9.63 | 63% (Assam) |

E1–E4a spread the gain across all four tail-heavy states. E4b makes West Bengal **worse** while over-correcting Assam (−0.31 pp, i.e. now over-predicting it), which is exactly the failure mode C4 exists to catch.

Overall signed bias in the worst reference states (positive = under-prediction):

| State | E0 | E1 | E2 | E3 | E4a | E4b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assam | +3.26 | +2.42 | +2.13 | +2.12 | +2.43 | −0.31 |
| West Bengal | +1.71 | +1.09 | +0.99 | +0.34 | +0.20 | +2.79 |
| Punjab | +2.13 | +2.53 | +2.62 | +2.15 | +1.41 | +1.22 |
| Sikkim | +2.75 | +2.09 | +1.84 | +2.30 | +1.65 | +1.32 |

Punjab gets **worse** under E1 and E2. The reweighting is not uniformly beneficial across high-bias states; it moves mass toward whatever the weighted loss finds, which happens to favour Assam more than Punjab.

### 4.5 Detection at the frozen p90 threshold

| Arm | Flagged (of 628) | Sensitivity at ≥ p95 | Precision for ≥ p95 |
| --- | ---: | ---: | ---: |
| E0 | 25 | 0.371 | 0.520 |
| E1 | 49 | 0.543 | 0.388 |
| E2 | 62 | 0.686 | 0.387 |
| E3 | 70 | 0.600 | 0.300 |
| E4a | 68 | 0.629 | 0.324 |
| E4b | 112 | 0.543 | 0.170 |

Every arm finds more of the extreme districts, and every arm also flags far more districts overall, so precision falls below the registered 0.40 floor. E2 is the best detector (24 of 35 extremes found, 0.686) but flags 62 districts to do it.

### 4.6 Conditional-quantile calibration (E4a / E4b)

| Arm | τ | Pinball loss | E0 pinball at same τ | Coverage | Calibration error | Reliability MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4a | 0.75 | **0.746** | 0.898 | 0.670 | −7.96 pp | 11.78 pp |
| E4b | 0.90 | **0.541** | 0.912 | 0.825 | −7.52 pp | 8.63 pp |

Both quantile arms beat the reference on the loss they were fitted to, and both still **under-cover by about 8 points**: E4b promises the 90th percentile and delivers empirical coverage of 82.5%. The reliability check shows the gap is not uniform across the prediction range.

Subgroup coverage (descriptive only, **not** a calibration requirement): E4b covers 11.4% of the ≥ p95 group at τ = 0.90, E4a 11.4% at τ = 0.75. This is reported as description, not as a failure — a conditional-quantile model is not required to hit nominal coverage inside a group selected on realised outcomes.

E4b's overall MAE of 2.912 and pooled R² of −0.302 (fold-mean R² −0.531) confirm that a τ = 0.90 estimator is not a mean predictor and must not be read as one.

## 5. Gate-by-gate results

C1–C6 applied exactly as registered. Thresholds derive from the E0 reference: C1 ≤ 3.101 (p90 MAE) and ≤ +3.157 (p95 bias); C2 ≤ 1.837 (MAE) and ≤ 2.398 (RMSE); C6 ≥ 0.471 sensitivity and ≥ 0.40 precision.

| Arm | C1 | C2 | C3 | C4 | C5 | C6 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | FAIL | FAIL | PASS 4/4 | PASS (56%) | PASS | FAIL (0.39) | not accepted — tail criterion failed |
| E2 | FAIL | FAIL | PASS 4/4 | PASS (57%) | PASS | FAIL (0.39) | not accepted — tail criterion failed |
| E3 | FAIL | FAIL | PASS 4/4 | PASS (49%) | PASS | FAIL (0.30) | not accepted — tail criterion failed |
| E4a | N/A | N/A | PASS 4/4 | PASS (37%) | PASS | FAIL (0.32) | not accepted |
| E4b | N/A | N/A | PASS 3/4 | FAIL (WB worse) | PASS | FAIL (0.17) | not accepted |

Numerical detail per gate is in `evaluation/metrics/tail_gate_results.csv`.

- **C1** failed for all three weighted arms: the p90 MAE target was met (3.032 / 2.831 / 2.807, all ≤ 3.101) but the p95 bias target was not (3.693 / 3.435 / 3.271, all > 3.157).
- **C2** failed for every arm: the smallest overall cost was +9.4% (E4a), against a 5% allowance.
- **C5** passed everywhere — no arm systematically over-predicts the tail (the closest is E4b at +2.994, still well above −1.0).
- **C6** failed everywhere, entirely on precision. Sensitivity passed for all five arms.

**No challenger meets the registered gates.** The registered acceptance rule requires C1–C5 for a candidate and the C1-and-C2 pair even for "specialised tail model" status; neither condition is satisfied by any arm.

## 6. Registered null-result statement

Because C1 failed for every arm to which it applies, the proposal's outcome rule applies, and the report states the required wording:

> No improvement was demonstrated using the tested tail-aware approaches. Limited predictor information is one possible explanation, but cannot be established from these experiments alone. Further evidence, potentially including additional independent predictors, would be required to establish an information ceiling.

**How that statement should be read alongside §4.** "No improvement" here means *no improvement sufficient to meet the pre-registered acceptance criteria*, not *no change in the tail*. The tail bias did fall by 0.8–1.4 pp, consistently across folds, with state-clustered intervals excluding zero — but that gain was below the registered 30% bar for the weighted arms, and it was purchased with an overall MAE increase of 9–14% that breaches the 5% allowance. Both facts belong in the record; collapsing either one would misstate what happened.

**No information ceiling is claimed.** These five remedies are a narrow class. Their failure is consistent with limited predictor information (H3) but does not establish it, and nothing here shows the tail gap is intrinsic at n = 628.

## 7. What remains uncertain

1. **Whether a better-specified weighting could meet both gates.** E4a's cost interval includes zero and its trade ratio (0.12) is the most favourable observed. A continuous rather than step weighting, or a weight tied to the p95 group rather than p75/p90, might shift the trade — but that is a new experiment requiring new authorisation, and adding arms after seeing these results would be exactly the post-hoc search the pre-registration exists to prevent.
2. **Whether the tail gain generalises to other states.** The extreme group is 35 districts, 24 in Assam and West Bengal. Punjab worsened under two arms. With this concentration, "improves the tail" is partly "improves Assam".
3. **Whether the quantile arms are miscalibrated by construction or by sample size.** An 8-point under-coverage at n = 628 with 77 collinear predictors is consistent with either; this design cannot separate them.
4. **Whether the detection gain is useful.** Sensitivity rose to 0.54–0.69, but precision fell to 0.17–0.39. Whether that trade is acceptable depends on the cost of a false flag, which is a policy question, not a statistical one.
5. **Any present-day or causal reading.** None is supported. The data are 2011, the design is ecological, and no coefficient in any of these arms has a causal interpretation.

## 8. Decision and next steps

- The ratified Phase 3 models and roles **stand unchanged**. Ridge on Configuration B remains the primary interpretable model.
- No challenger is promoted, and no model decision is revised by this report.
- The investigation is closed as executed. Further tail work would require a new proposal and separate authorisation; nothing here authorises Phase 5.
- The substantive finding for the record: **tail compression in this dataset can be partially reduced, at a measurable cost to overall accuracy, and not far enough to satisfy either the tail or the cost criterion at the same time.**

## 9. Reproduction

```
py tests/test_phase4_isolation.py            # 9/9, must pass before results
py src/training/run_tail_experiments.py      # ~12 s, writes OOF + diagnostics
py src/evaluation/tail_evaluation.py         # metrics, gates, bootstrap
```

## 10. Outputs

`evaluation/predictions/tail_oof_predictions.csv` · `evaluation/metrics/`: `tail_experiments.csv`, `tail_fold_results.csv`, `tail_state_results.csv`, `tail_tail_metrics.csv`, `tail_gate_results.csv`, `tail_quantile_metrics.csv`, `tail_quantile_reliability.csv`, `tail_bootstrap.csv`, `tail_error_by_predicted_decile.csv`, `tail_fit_diagnostics.csv`, `tail_selected_params.json`, `tail_runtime.csv`.
Full file-level verification is in `reports/ml_report/PHASE_4_VERIFICATION.md`.

---

## 11. Post-audit corrections (added 2026-11-12)

An independent audit (`PHASE_4_INDEPENDENT_AUDIT.md`, `FINAL_MASTER_AUDIT.md`)
confirmed every reported number but found five documentary defects and one
methodological deviation. This section records the corrections. **No experiment
was re-run, no gate changed status, no model decision was revisited, and no
Phase 4 artefact was overwritten.** The new material is additive and derived
from the stored predictions.

### 11.1 The outcome, named explicitly

The quantity predicted throughout this investigation is the **district-level
prevalence of dilapidated census houses** in India (Census 2011) — the share of
a district's census houses classified as *Dilapidated* by enumerators. It is a
structural-condition measure of the housing stock.

**This project does not predict individual homelessness**, and nothing in these
experiments is a homelessness measure. Earlier versions of this report referred
to "extreme districts" without naming the outcome; that shorthand is now
replaced wherever it could be read as a different construct.

### 11.2 Threshold range (defect D-06 / audit D-02)

The report and experiment log stated the q75 weighting threshold as
**6.58–7.10**. The recorded per-fold thresholds are

| Fold | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| q75 (E1, E2) | 6.575 | 6.700 | **6.250** | 6.700 | 7.100 |
| q90 (E3) | 9.200 | 8.790 | 8.700 | 9.200 | 9.680 |

so the correct range is **6.250–7.100** (q90: 8.700–9.680). The error came from
transcribing fold 1's value as if it were the minimum. Source:
`evaluation/metrics/tail_selected_params.json`. Corrected in this report
(§2), in `PHASE_4_VERIFICATION.md` and in `Admin/EXPERIMENT_LOG.md`.

### 11.3 Overall bias and the p95 calibration gap (defects D-05 / D-09)

Neither quantity appeared in the original report. Both are computed from
`tail_experiments.csv` and add the missing half of the tail story:

| Arm | Mean actual | Mean predicted | Overall bias (actual − predicted) | p95 mean actual | p95 mean predicted | p95 gap (predicted − actual) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 5.058 | 4.966 | **+0.092** | 13.231 | 8.722 | **−4.510** |
| E1 | 5.058 | 5.422 | −0.364 | 13.231 | 9.539 | −3.693 |
| E2 | 5.058 | 5.586 | −0.528 | 13.231 | 9.796 | −3.435 |
| E3 | 5.058 | 5.615 | −0.557 | 13.231 | 9.961 | −3.271 |
| E4a | 5.058 | 5.900 | −0.842 | 13.231 | 10.123 | −3.109 |
| E4b | 5.058 | 7.345 | **−2.287** | 13.231 | 10.238 | **−2.994** |

Sign convention: overall bias is mean(actual − predicted), so **positive means
the arm under-predicts on average**. The p95 gap is mean(predicted − actual)
within the 35 districts at or above 11.200, so **negative means the arm
under-predicts the extreme group** — the deficiency every arm was built to
reduce.

**The bias reversal.** E0 is almost unbiased overall (+0.092) while
under-predicting the tail by 4.5 pp. Every tail-aware arm reverses the sign of
the overall bias: E1 −0.36, E2 −0.53, E3 −0.56, E4a −0.84, E4b −2.29. That is
the arithmetic signature of shifting the prediction distribution upward rather
than identifying the tail: the mean prediction now sits above the mean
outcome, while the p95 group is still under-predicted by 3–3.7 pp.

E4b shows the mechanism at its extreme — over-predicting by 2.287 pp overall
while still under-predicting the tail — which is why its pooled R² is −0.302
and why it must not be read as a mean predictor. A tail gain bought by lifting
every district is not a tail gain in the sense the proposal intended; the
registered C4 gate (state concentration) and C2 (overall cost) exist precisely
to catch this, and both fired for E4b.

### 11.4 Registered district-level bootstrap (defect D-01 / audit D-05)

**The deviation.** `EXTREME_DISTRICT_REMEDY_PROPOSAL.md` registers a
**district-level** bootstrap: resample the 628 districts with replacement,
recompute each arm's metric on the resample, and compare arms on the *same*
resample. The executed evaluation resampled **states** (35 clusters), on the
argument that tail districts concentrate in four states, so districts are not
independent. That reasoning was not in the pre-registration, and the executed
method must not be described as preregistered.

Both are now computed from the stored predictions and reported side by side
(`evaluation/metrics/tail_bootstrap_comparison.csv`; the registered result alone
is in `tail_bootstrap_district.csv`). The state path was re-implemented in
`src/evaluation/bootstrap_paired.py` and reproduces the frozen
`tail_bootstrap.csv` to < 1e-9, so the executed numbers remain verifiable.

| Arm | Δ MAE: registered district [95% CI] | Δ MAE: executed state [95% CI] | Δ p95 bias: registered district [95% CI] | Δ p95 bias: executed state [95% CI] |
| --- | --- | --- | --- | --- |
| E1 | +0.165 [+0.115, +0.215] | +0.165 [+0.022, +0.327] | −0.817 [−0.961, −0.682] | −0.817 [−1.077, −0.487] |
| E2 | +0.225 [+0.159, +0.293] | +0.225 [+0.041, +0.428] | −1.074 [−1.256, −0.901] | −1.074 [−1.455, −0.722] |
| E3 | +0.235 [+0.165, +0.304] | +0.235 [+0.055, +0.457] | −1.239 [−1.400, −1.053] | −1.239 [−1.508, −0.704] |
| E4a | +0.165 [+0.078, +0.250] | +0.165 [−0.023, +0.374] | −1.401 [−1.654, −1.144] | −1.401 [−1.695, −1.068] |
| E4b | +1.162 [+0.995, +1.328] | +1.162 [+0.718, +1.653] | −1.516 [−2.404, −0.553] | −1.516 [−3.782, +1.251] |

The point estimates are identical by construction; only the interval width
depends on the cluster unit. District-level intervals are **1.07–3.55×**
narrower across the 20 cells (E4b's p95 interval is 1.852 pp wide against
5.033 pp state-clustered, a 2.7× difference; the smallest difference is E4a's
p95 interval, 0.608 vs 0.651 pp, where only four states carry extremes).

**Two interpretive consequences, both in favour of the conservative reading:**

1. §4.3's statement that E4a is "the only arm whose overall cost interval
   includes zero" holds **only under state clustering**. Under the registered
   district-level procedure E4a's cost is +0.165 [+0.078, +0.250] — clearly
   positive. The correct statement is: *every* arm's overall cost excludes zero
   under district resampling; E4a's is merely the smallest.
2. E4b's tail interval spans zero under state clustering but **excludes zero**
   under district resampling ([−2.404, −0.553]). A reader told only the
executed result would conclude E4b's tail gain is unstable; that conclusion is
method-dependent. It changes nothing about E4b's fate — C2, C4 and C6 fail
regardless — but integrity requires reporting both.

**No gate is recomputed from either interval.** C1–C6 are point-estimate rules
as registered; uncertainty is descriptive. Re-running gates against a different
bootstrap would be changing the acceptance rule after seeing results.

### 11.5 Binomial uncertainty for quantile coverage (defect D-08)

Coverage is a proportion of 628 districts, so it needs an interval. Wilson
score and Jeffreys intervals are reported in
`evaluation/metrics/tail_quantile_coverage_ci.csv`.

| Arm | τ | Coverage | Wilson 95% | Jeffreys 95% | Nominal |
| --- | ---: | ---: | --- | --- | ---: |
| E4a | 0.75 | 67.0% | 63.3 – 70.6% | 63.3 – 70.6% | 75% |
| E4b | 0.90 | 82.5% | 79.3 – 85.3% | 79.4 – 85.3% | 90% |

Both intervals exclude the nominal level, so the ~8-point under-coverage is not
sampling noise at n = 628: it is a property of these fits. (E1–E3 report
identical coverage at both τ — 57.5%, 58.9%, 61.0% — because their predictions
are mean predictions and do not depend on τ; their "coverage" is not a
calibration statement about a fitted quantile.)

Subgroup coverage at the p95 threshold rests on **n = 35 districts** and must
be read with that in mind: E4b covers 11.4%, Wilson 95% [4.5%, 26.0%]; E4a
11.4%, [4.5%, 26.0%]. The interval is 21.5 pp wide. **This descriptive figure
is not evidence of calibration failure or of calibration success** — the group
is selected on realised outcomes, which a conditional-quantile model is not
required to track.

### 11.6 E4b predictions outside the observed range (defect D-13)

E4b's out-of-fold predictions run from **1.911 to 29.437**, while the observed
target maximum is **17.700**. By contrast E1's range is −1.475 to 13.727, E3's
−2.953 to 14.657, E4a's −1.558 to 15.202.

**Why this happens, and why it is not a bug.**

- A τ = 0.90 linear quantile model is an unconstrained linear function of 77
  features. Fitting a hyperplane that leaves 90% of a right-skewed target
  below it necessarily positions the plane high, and the ceiling is set by the
  features, not by the observed target.
- The target's upper tail is concentrated in West Bengal and Assam. Districts
  whose feature vectors are extreme in that direction are scored by
  extrapolation along the fitted direction; 29.437 is what the linear form
evaluates to for the most extreme such vector, roughly 1.66× the largest
  observed value.
- The optimiser is exact (`QuantileRegressor(solver="highs")`), all predictions
  are finite, and the arm reproduces byte-identically on re-run. No numerical
  defect was found.
- Synthetic check: Monte-Carlo and hand examples in
  `tests/test_phase4_quantile_and_bootstrap.py` confirm the loss and coverage
  arithmetic at τ = 0.75 and 0.90, including ties and both directions.

**Practical reading.** A prediction of 29.4 means "this model's estimated 90th
percentile of the conditional outcome distribution is 29.4 percentage points",
not "29.4% of this district's houses are dilapidated". The quantity is a
screening score for a quantile, and E4b is not a mean predictor — its overall
MAE (+66%) and negative R² are the price of that.

**Why the values were not clipped.** Clipping at the observed maximum would
(a) alter a registered experiment's outputs after seeing them, (b) conceal the
extrapolation that is itself the finding, and (c) break the consistency between
the stored predictions and the pinball/coverage diagnostics computed from them.
Values are reported as computed. The same principle is applied to the two
slightly negative fitted values in the Phase 6 Ridge deliverable.

### 11.7 Added regression tests (defect D-07)

`tests/test_phase4_quantile_and_bootstrap.py` — 14 tests, all passing:

- **quantile loss and coverage:** pinball loss against eight hand-computed
  values at τ ∈ {0.1, 0.5, 0.75, 0.9} including over-prediction and exact hits;
  an asymmetry assertion that fails if the argument order or inequality is
  reversed; loss symmetry under (y ↔ p, τ ↔ 1−τ); coverage recomputed from the
  frozen predictions for all four E4a/E4b cells; tie and boundary behaviour.
- **bootstrap:** fixed-seed reproducibility (including the registered 2,000-rep
  procedure); paired resampling proved by two constructions (an identical arm
  must collapse to a zero-width interval; a uniformly-1.0-worse arm must give
  exactly [1, 1]); the state path reproducing `tail_bootstrap.csv` to < 1e-9;
  district and state units not interchangeable and invalid units rejected;
  Wilson verified against its defining score equation and Jeffreys against its
  Beta quantiles.

### 11.8 What was deliberately not done

- No arm was added, removed or re-tuned; no gate threshold was edited.
- E4b was not clipped, smoothed or refitted.
- No Phase 3 artefact, target, fold or feature definition was touched.
- No promotion decision was revisited; the registered null-result statement in
  §6 stands, and its reading alongside §4 and §11.3 remains the same: the tail
  moved, the movement was below the registered bar, and it cost overall
  accuracy.

### 11.9 Remaining low and informational items — status

These were raised by the audit and are recorded here so nothing is silently dropped.

| Ref | Item | Status |
| --- | --- | --- |
| D-12 | Report runtime (11.7 s) is not recoverable from `tail_runtime.csv` (sums to 8.652 s) | **Acknowledged.** Wall-clock time is machine-dependent and is not a reproducible quantity; the data artefacts are byte-identical on re-run, which is what matters. Timing is not used in any comparison. |
| D-16 | `n_configs` docstring implied the grid was already crossed with the inner CV | **Fixed** in `src/models/registry.py`: it now states that it returns the grid size and that the inner CV multiplies it by the fold count. No behaviour change. |
| D-17 | Grid-free specification fitted as a shared object; resumable merge keeps the last row on duplicate keys | **Justified, not changed.** Either would require editing a frozen code path; neither affects any stored result (the grid-free family has a single fit, and no sweep was interrupted). Documented here rather than silently fixed. |
| D-18 | Collinearity screen is order-dependent and computed on the full sample | **Disclosed** in Phase 2 and Phase 3 documentation; deterministic under the declared feature order. |
| D-19 | 17 PNGs and one output-bearing notebook are large binaries | **Accepted**; they are the figures used by the presentation. The cleanup manifest notes they should be excluded from a text-only submission. |
| D-20 | Empty directories and `_audit/` scratch | **Partially resolved**: `reports/final_report/` is now populated. The remaining empty directories and the audit scratch are retained deliberately and listed in `REPOSITORY_CLEANUP_MANIFEST.md`. |
| D-21 | Primary model had no permutation importance | **Resolved**: `src/evaluation/ridge_importance.py` → `phase3_ridge_permutation_importance.csv`; 26/77 features stable, 7/10 overlap with the largest coefficients. Reported in `PHASE_3_MODELLING_REPORT.md` Appendix A. |

No item in this table changes a result, a gate, a threshold or a model decision.
