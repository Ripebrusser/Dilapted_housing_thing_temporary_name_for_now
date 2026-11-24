# Proposal — Tail-Aware Modelling for Extreme-District Underprediction

**Status:** **proposal only — not authorised, not implemented.**
**Phase:** post-Phase-3 investigation proposal. Phase 4 has **not** started.
**Effect on ratified work:** none. No model was fitted, no dataset modified, no Phase 3 metric, prediction or plot changed.
**Ratified roles (unchanged):** Configuration B (628 districts, 77 predictors) primary; Ridge primary interpretable; Elastic Net secondary interpretable; SVR-RBF accuracy benchmark; Configuration A (640 districts, 43 predictors) secondary full-coverage.

---

## 1. Problem statement

Phase 3 established a systematic failure at the top of the target distribution. The ratified models do not merely have noisy tail errors — they **compress the distribution**: they predict extreme districts close to the middle and leave the largest values unreached.

Evidence from the frozen Phase 3 artefacts (out-of-fold predictions, Configuration B, grouped scheme):

| Model | Overall MAE | ≥ p90 (n=64) MAE | ≥ p90 bias | ≥ p95 (n=35) MAE | ≥ p95 bias | Mean prediction in ≥ p95 group (actual 13.23) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Ridge (primary)** | 1.750 | 3.648 | +3.593 | 4.527 | +4.510 | 8.72 |
| Elastic Net | 1.748 | 3.587 | +3.180 | 4.292 | +3.874 | 9.36 |
| SVR-RBF (benchmark) | 1.725 | 3.572 | +3.461 | 4.378 | +4.298 | 8.93 |

Bias is `mean(actual − predicted)`: **+4.51 pp** for Ridge means the model under-predicts the most deprived 5% of districts by more than four percentage points on average. 34 of 35 extreme districts are under-predicted.

The practical consequence is worse than the MAE suggests. Using the frozen p90 threshold (9.20 pp) as a flag:

| Model | Districts flagged | Extreme districts flagged (sensitivity) | Top-decile flagged | Precision for ≥ p95 |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 25 / 628 | 13 / 35 (37%) | 16 / 64 (25%) | 0.52 |
| Elastic Net | 34 / 628 | 14 / 35 (40%) | 20 / 64 (31%) | 0.41 |
| SVR-RBF | 35 / 628 | 15 / 35 (43%) | 19 / 64 (30%) | 0.43 |

Ridge misses 22 of the 35 most deprived districts outright. The successes are concentrated: 11 of 13 are West Bengal districts in fold 3; in fold 2 only **1 of 12** extreme districts is flagged (Assam). The failure is therefore not uniform noise — it is a specific inability to reach values that are far from the national socioeconomic relationship.

## 2. Frozen evaluation definitions

These are **evaluation constants**, fixed from the Phase 3 target distribution and never passed to any learner.

| Constant | Value | Source |
| --- | ---: | --- |
| Evaluation population | 628 districts (Configuration B) | ratified Phase 3 |
| Target mean | 5.058 | frozen `district_analysis.csv` |
| p75 | 6.700 | frozen target |
| **p90 (top-decile threshold)** | **9.200** | frozen target |
| **p95 (extreme threshold)** | **11.200** | frozen target |
| Top-decile group size | 64 | — |
| Extreme group size | 35 | — |

**Extreme districts per outer test fold** (the requested count):

| Fold | n test | n ≥ p90 | n ≥ p95 | States held out |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 126 | 11 | 4 | 6 |
| 2 | 126 | 20 | 12 | 7 |
| 3 | 125 | 21 | 14 | 7 |
| 4 | 126 | 10 | 5 | 8 |
| 5 | 125 | **2** | **0** | 7 |

**This is the single most important design constraint.** Fold 5 contains no district above p95 and only two above p90. Top-5% performance is therefore estimable on **four folds and 35 districts**, and fold 5 cannot contribute to it at all. Any claim about tail generalisation rests on four folds.

**Where the extremes are:** Assam 12, West Bengal 12, Uttar Pradesh 4, Orissa 4, Bihar 2, Arunachal Pradesh 1. Two states hold 24 of 35 (69%). With this concentration, a "one-state effect" cannot be statistically separated from a "deprivation effect" — the design must instead show that improvement is not *confined* to one state.

## 3. Hypotheses to test

The proposal is structured around three competing explanations, because they imply different remedies and different expected outcomes.

- **H1 — Loss specification.** The squared-error objective is dominated by the 593 non-extreme districts; the conditional mean is pulled toward the bulk. *Remedy:* reweight the loss (E1–E3).
- **H2 — Functional form / estimand.** Even with correct weighting, a linear model in 77 mostly compositional features cannot represent the non-linear threshold behaviour at the top. *Remedy:* a quantile estimand as a challenger (E4), which changes what is being estimated rather than how errors are weighted.
- **H3 — Limited predictor information.** One possible explanation is that the predictors do not contain enough signal to identify Assam-type extremes. **This cannot be established from these experiments alone.** *Prediction, if H3 were true:* all arms fail the tail gate, or succeed only by inflating predictions broadly (bias flips, precision collapses).

H1 and H2 are directly testable by these arms; H3 is not. The evidence is asymmetric: consistent failure across E1–E4 would be **consistent with** H3 but would not confirm it, because loss weighting and quantile estimands are a narrow class of remedies. Establishing an information ceiling would require further evidence, potentially including additional independent predictors.

A well-designed experiment must be able to separate H1 from H2, and to detect over-correction. That is why the evaluation includes **tail bias, overall cost and flag precision together** — an arm that removes bias by shifting every prediction upward is distinguishable from one that actually identifies extremes.

## 4. Proposed experiments

All arms use Configuration B, the frozen five outer state-grouped folds, and identical districts. Arm E0 is **not refitted** — it is the completed, ratified Ridge.

| Arm | Definition | What it tests | Rationale |
| --- | --- | --- | --- |
| **E0** Reference | Ratified Ridge, α = 100 per fold, uniform weights (Phase 3 result) | Baseline for every comparison | Reusing the frozen OOF predictions avoids re-fitting and keeps the reference exactly as ratified |
| **E1** Mild weighting | w = 1 + 2·𝕀[y ≥ q75(train)], normalised to mean 1 | Does a modest overweight of the top quartile move the tail? | Top quartile ≈ 157 districts: enough signal for stable estimation without starving the bulk |
| **E2** Strong weighting | w = 1 + 4·𝕀[y ≥ q75(train)] | Is the tail gap a weighting-strength problem? | 5× weight on the top quartile is a large but still estimable shift (effective n ≈ 0.57n) |
| **E3** Threshold weighting | w = 1 + 4·𝕀[y ≥ q90(train)] | Is the failure specific to the top decile rather than the upper quartile? | Sharp focus; distinguishes "upper-tail compression" from "general high-end compression" |
| **E4a / E4b** Quantile regression | Linear quantile regression at τ = 0.75 and τ = 0.90 (L1 penalty) | Can a different estimand reach the tail at all? | Predicts a conditional quantile, not a mean. Reported as a **separate challenger**, expected to worsen MAE by construction |

**Why only these five.** They span the two plausible mechanisms (loss weighting, estimand) with one strength dimension each, and they are cheap. A broader architecture search (two-stage/hurdle, target transformation, mixture models) is deliberately **not** proposed: those change the estimand structurally, would require their own leakage and interpretation review, and are premature until we know whether a simple reweighting moves the tail at all.

**Explicitly rejected for this investigation:**
- deleting, capping or winsorising extreme observations (forbidden, and it would corrupt the target);
- log or power transformation of the target (changes the estimand and biases MAE on the original scale);
- post-hoc state-level additive corrections (forbidden; and would be test-residual-based);
- isotonic recalibration of predictions (cheap, but it cannot create information the predictors lack; may be a later, separately justified step).

### Weight construction (training-fold only)

For a training fold with labels `y_tr`:

```
t        = quantile(y_tr, q)          # q ∈ {0.75, 0.90}
w_i      = 1 + λ · 1[y_i ≥ t]         # λ ∈ {2, 4}
w_i      = w_i / mean(w)              # normalise so the loss scale is unchanged
```

Normalising to mean 1 keeps the Ridge penalty comparable to α = 100, so α remains interpretable. Resulting weights: E1 → 0.667 / 2.0; E2 → 0.5 / 2.5; E3 → 0.714 / 3.57.

Thresholds are **quantiles of the training fold**, recomputed in every fold and every inner split. The evaluation constants (§2) are never used to build weights. The target is legitimately used here — it is the label, and cost-sensitive weighting is not leakage — but its *test-fold* values never enter.

### Sample-weight isolation (inner CV versus outer refit)

Weights are computed **inside whichever fit consumes them**, never once on the complete outer training set. The procedure is two-staged:

1. **Inner loop.** For each inner split, the threshold `t` is the `q`-quantile of that split's **inner-training target values only**; the weights are computed from those inner-training targets only; and both are applied only to that inner model fit. Inner-validation targets never enter `t` or `w`.
2. **Hyperparameter selection.** α (or the quantile L1 penalty) is chosen from inner-fold scores. No weight or threshold from the inner stage is carried forward.
3. **Outer refit.** Once α is fixed, the threshold and weights are **recomputed from the complete outer training set** and applied to that single final outer-fold refit.
4. **Isolation.** Outer test targets are excluded from every weight calculation at both stages: they never set `t`, `w`, α, or any preprocessing statistic.

This ordering matters because computing weights once on the whole outer training set — including the rows that later serve as inner-validation — would let the weighting threshold see data it is being scored against, and would bias hyperparameter selection toward whichever α best exploits that leak.

### Hyperparameter selection

| Arm | Grid | Inner selection criterion |
| --- | --- | --- |
| E1–E3 | α ∈ {30, 100, 300} | mean absolute error, 4-fold inner `GroupKFold` |
| E4a / E4b | L1 penalty ∈ {0.001, 0.01, 0.1} | mean pinball loss at the same τ |

α must be re-selected because weighting reduces the effective sample size (Kish effective n ≈ 0.57n for E2). The grid is deliberately three points wide: the purpose is to test a mechanism, not to squeeze out metric gains.

Quantile regression: `QuantileRegressor(quantile=τ, alpha=α_q, solver="highs")` inside the same pipeline (median imputer + `StandardScaler`).

## 5. Nested validation design

- **Outer folds:** the exact Phase 3 grouped folds — 5-fold `GroupKFold` on the 35 states, byte-identical district assignments (verified by the fold-signature check already in the test suite).
- **Inner folds:** 4-fold `GroupKFold` on the outer-training states only.
- **No test outcomes** may set weights, thresholds, α, or any preprocessing statistic. Weights and thresholds are functions of training labels only; scaling and imputation are fitted per fold inside the pipeline.
- **Single pass.** One pre-registered configuration per arm. No iterative tuning after seeing results, no arm added after the fact.
- **Reference reuse.** E0 is read from `phase3_oof_predictions.csv`; it is not refitted, so the comparison is against exactly the ratified model.

Implementation note: weighted fitting requires passing per-sample weights through the pipeline (`model__sample_weight`). `GridSearchCV` must slice the weight vector consistently with each split. A silently misaligned or leaky weight vector would invalidate every number produced, so the following checks are **prerequisites for reporting any result**:

- **Weight routing test.** The vector reaching the estimator must equal the expected weights for exactly the rows in that split's train index. A shift by one row, or a full-length vector applied to a subset, fails the test.
- **Inner-isolation test.** For one outer fold and one inner split, overwrite the inner-validation targets with arbitrary values (e.g. a shuffled copy) and re-run the inner fit. The threshold, the normalised weight vector and the fitted coefficients must be **bit-identical** to the unmodified run. Any change indicates that inner-validation targets reached the weighting step.
- **Outer-isolation test.** Overwrite the outer-test targets with arbitrary values and re-run the outer refit. Threshold, weights and coefficients must again be bit-identical.
- **Audit trail.** Every fit writes its fold, stage (`inner` / `outer_refit`), threshold value, weighted-row count, weight mean and selected α to `tail_fit_diagnostics.csv`, so isolation can be verified after the fact without re-running the experiment.

## 6. Evaluation

All metrics are computed on out-of-fold predictions only.

**Primary (overall):** MAE, RMSE, R², Spearman — pooled and mean ± SD across the five folds.

**Tail (frozen groups):**
- MAE and signed bias in ≥ p90 (n = 64) and ≥ p95 (n = 35);
- share of extreme districts under-predicted;
- mean predicted vs mean actual in each group (calibration gap).

**Stability:**
- fold-level tail MAE and bias, with the per-fold extreme counts from §2 (folds 1–4 only for p95);
- per-state tail MAE and bias for states with ≥ 3 extreme districts (Assam, West Bengal, Uttar Pradesh, Orissa; Bihar has 2 and is reported descriptively).

**Diagnostics:**
- distribution of predictions in the extreme group (does the arm actually reach 9.2+?);
- flag sensitivity/precision at the frozen p90 threshold, so over-correction is visible;
- paired district-level bootstrap (2,000 resamples) on ΔMAE and Δbias versus E0 — reported with the caveat that it ignores fold and spatial dependence;
- an explicit trade-off ratio: overall MAE change (pp) per 1 pp of tail bias reduction.

### E4a / E4b — evaluation of the conditional-quantile challengers

A quantile model does not estimate the conditional mean, so it is **not evaluated as if it did**. Its output is the estimated τ-quantile of the conditional target distribution. MAE, RMSE and R² against the realised target are therefore reported for transparency but are structurally penalised and say nothing on their own about whether the estimand is useful.

Required additional metrics for E4a (τ = 0.75) and E4b (τ = 0.90), on out-of-fold predictions:

- **Pinball (quantile) loss at the model's own τ** — the loss the model is actually fitted to minimise, reported alongside the same functional evaluated on Ridge's point predictions as a non-optimised reference;
- **empirical quantile coverage** — the share of districts with `y ≤ ŷ`, which should be ≈ τ under correct conditional calibration; reported overall and separately for the ≥ p90 and ≥ p95 groups, with a binomial interval that acknowledges n = 35 in the extreme group;
- **calibration error** — nominal minus empirical coverage in percentage points, plus a coarse reliability check: bin districts by predicted value and compare realised exceedance rates with τ;
- **extreme-district detection at the frozen p90 threshold** — flag sensitivity and precision defined exactly as for the point-prediction arms, so E4 can be compared on the practical targeting task despite the different estimand.

**Gate application for E4.** The six gates in §7 are **unchanged — not relaxed, not waived** — and every gate result for E4a/E4b is reported. The reason for differential interpretation is documented here rather than by editing a gate: C1 and C2 are expressed in MAE/RMSE for a conditional mean, which a τ = 0.90 quantile estimator does not target, so an E4 failure on C1/C2 is recorded as *"not like-for-like — different estimand"* and must not be presented as a refutation of quantile regression. C3–C6 (fold consistency, state dependence, over-correction, detection) remain directly meaningful and are applied as written. E4 can be recommended only as a specialised targeting tool, never as a mean predictor and never as a replacement for the ratified primary model.

**Reporting:** every arm appears in every table. No selective reporting, and no arm removed for performing badly.

## 7. Pre-registered decision criteria

A challenger is a **candidate for recommendation** only if all of C1–C5 hold. C6 is a practical read-out, not a gate.

| # | Criterion | Threshold |
| --- | --- | --- |
| **C1** Tail effect | ≥ p90 MAE reduced ≥ 15% relative (≤ 3.101 vs 3.648) **and** ≥ p95 bias reduced ≥ 30% (≤ +3.16 vs +4.510) |
| **C2** Overall cost | Overall MAE ≤ 1.837 (+5% on 1.7496) **and** RMSE ≤ 2.398 (+5% on 2.2842) |
| **C3** Consistency | Improvement in the same direction in ≥ 3 of the 4 folds that contain extreme districts (folds 1–4); fold 5 excluded because it has none |
| **C4** Not one state | Absolute-error reduction appears in ≥ 2 of {Assam, West Bengal, Uttar Pradesh, Orissa}; no single state contributes > 60% of the total tail error reduction |
| **C5** No over-correction | ≥ p95 bias does not fall below −1.0 pp |
| **C6** Practical | Flag sensitivity for ≥ p95 rises by ≥ 10 points (37% → ≥ 47%) with precision ≥ 0.40 |

**Outcome rules, fixed in advance:**

- **C1 + C2 pass** → recommend the arm as a tail-aware alternative, still subject to a separate ratification decision. The ratified primary remains Ridge unless that decision says otherwise.
- **C1 passes, C2 fails** → document as a **specialised tail model** for targeting use, explicitly not a replacement; the ratified primary is unchanged.
- **C1 fails** → report a **null result** and recommend stopping. The report must state, verbatim: *"No improvement was demonstrated using the tested tail-aware approaches. Limited predictor information is one possible explanation, but cannot be established from these experiments alone. Further evidence, potentially including additional independent predictors, would be required to establish an information ceiling."* The null result is a legitimate finding about these remedies; it is **not** evidence that the tail gap is intrinsic to n = 628.
- Five arms × two tail metrics create multiple-comparison risk; because the criteria are pre-registered and one pass only is allowed, the correct interpretation of a single marginal pass is "not convincing".

## 8. Risks and limitations

1. **The tail sample is 35 districts, 24 of them in two states.** Tail estimates have wide uncertainty and cannot separate "this is about deprivation" from "this is about Assam and West Bengal". C4 mitigates but cannot eliminate this.
2. **Fold 5 contributes nothing to the tail.** Tail validation is effectively 4-fold, so C3 is weaker than it looks.
3. **Weighting changes the estimand.** A reweighted fit is no longer estimating the conditional mean; its overall MAE is an empirical outcome, not a guarantee, and its coefficients are not comparable with the ratified Ridge coefficients.
4. **Hard thresholds are discontinuous.** A district just above q75 receives 5× the weight of its neighbour just below. This can destabilise coefficients near the cut and is a further reason not to read weighted coefficients individually.
5. **Quantile regression is not a mean predictor.** At τ = 0.90 it estimates the conditional 90th percentile; MAE and R² will worsen by construction, and the arm must be judged on tail coverage, not on overall accuracy.
6. **Bias may flip.** Shifting the top upward can create systematic over-prediction; C5 exists for that reason, and the overall bias will be reported for every arm.
7. **Bootstrap CIs are optimistic** because they ignore fold assignment and spatial dependence.
8. **Multiple comparisons.** Five arms, two tail metrics, one pass; a marginal single-criterion pass should not be treated as evidence.
9. **2011 data, ecological design.** Nothing here supports present-day inference or household-level claims, and nothing here is causal.
10. **Coverage.** Configuration B excludes 12 districts with unresolved D5 matching; any tail conclusion inherits that caveat.

## 9. What this investigation cannot establish

- Whether the *unobserved* determinants of Assam-type extremes are socio-cultural, historical or administrative — the predictors contain no such variables.
- Whether a different model class would solve the problem; only whether loss weighting and quantile estimands help.
- Whether any remedy generalises to later censuses.
- Whether limited predictor information explains the tail gap. A null result from E1–E4 would not establish an information ceiling; that would need further evidence, potentially including additional independent predictors.

## 10. Cost

| Component | Estimated fits | Estimated time |
| --- | ---: | --- |
| E1–E3 weighted Ridge (3 arms × 5 folds × 3 α × 4 inner) | 180 | < 1 min |
| E4a/E4b quantile regression (2 τ × 5 folds × 3 α × 4 inner) | 120 | 1–6 min (L1 solver dominates) |
| Evaluation, bootstrap, tables | — | < 2 min |
| **Total** | **300** | **≈ 5–10 min** |

Estimate, not a measurement. No new dependencies and no data changes are required.

## 11. Deliverables if authorised

- `src/models/tail_aware.py` — weight constructors and the quantile wrapper, with the weight-slicing contract.
- `src/training/run_tail_experiments.py` — resumable runner over the frozen folds.
- `evaluation/metrics/tail_experiments.csv`, `tail_fold_results.csv`, `tail_selected_params.json`, `tail_fit_diagnostics.csv` (weight audit trail), `tail_quantile_metrics.csv` (pinball loss, coverage, calibration error for E4).
- `evaluation/predictions/tail_oof_predictions.csv`.
- `reports/ml_report/EXTREME_DISTRICT_REMEDY_REPORT.md` — all arms, all criteria, explicit accept/reject per criterion, and the verbatim null-result wording if C1 fails.
- Tests: training-only weight thresholds; the three isolation checks in §5 (inner-split, outer-test, weight-to-estimator routing); fold identity with Phase 3; pinball loss and coverage verified against a synthetic case with known calibration; bootstrap reproducibility.

**Explicitly out of scope:** modifying `phase3_*` outputs, the ratified model roles, the target, or any Phase 2 artefact.

## 12. Recommendation

**Authorise the investigation as specified — a single bounded pass, with the pre-registered criteria above.**

Reasons:
1. The defect is large and practically decisive: a +4.51 pp bias and 37% flag sensitivity make the current model unusable for identifying the most deprived districts, which is the project's stated interest.
2. The cost is ~5–10 minutes of compute, no new dependencies, no data changes and no risk to ratified results.
3. The pre-registered gates make a **negative** result informative rather than a failure: if reweighting and quantile estimands both fail, we would know that these specific approaches do not help, which is worth reporting and narrows the options. It would **not** establish an information ceiling — limited predictor information is one possible explanation among several, and further evidence, potentially including additional independent predictors, would be required to establish one.

**Set expectations honestly.** Reweighting usually buys a partial improvement in tail bias at a cost in overall error. A realistic best case is halving the bias (+4.51 → ~+2.5) with a 2–5% overall MAE cost, not closing the gap. A null result is a likely outcome and should be pre-declared as acceptable.

**Not recommended at this stage:** two-stage or hurdle models, target transformations, architecture search, or any post-hoc state correction. Those are bigger commitments that should wait for the evidence from this pass.

**Governance.** This proposal does not authorise implementation, does not change the ratified primary/secondary/benchmark roles, and does not start Phase 4. Implementation requires separate approval.

---

## Execution note (added 2026-11-12 — record, not amendment)

The registration above is unchanged. Two facts must travel with it:

1. **The executed evaluation resampled states, not districts.** The registered
   district-level paired bootstrap (line 163) was not the procedure used in the
   first execution: `src/evaluation/tail_evaluation.py` clustered by state, on the
   argument that tail districts concentrate in four states. That reasoning is
   defensible but it was not pre-registered, and the executed result must not be
   described as preregistered.
2. **The registered procedure has now also been run.** `src/evaluation/bootstrap_paired.py`
   implements both units; the district-level result is in
   `evaluation/metrics/tail_bootstrap_district.csv` and both are reported side by
   side in `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md` §11.4. The state path
   reproduces the original `tail_bootstrap.csv` to < 1e-9.

No gate, threshold or model decision was changed by either interval set; C1–C6 are
point-estimate rules as registered.
