# Phase 4 — Independent Verification and Technical Audit

**Auditor role:** independent ML research auditor.
**Object:** Phase 4, the authorised tail-aware investigation (arms E0–E4b) for *Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning*.
**Method:** every number below was recomputed from raw artefacts with formulas written by the auditor; the project's evaluation module was not trusted as an oracle. Nothing was fitted into a project file, no artefact was overwritten, no gate was changed, and Phase 5 was not started.

---

## 1. Executive verdict

**VERIFIED WITH LIMITATIONS.**

The Phase 4 implementation, the stored results and the written conclusions are independently supported. Specifically:

- every reported overall, tail, fold-level, state-level and flag metric reproduces to **max |difference| = 0.00e+00** across all 30 reported metric cells × 6 arms;
- the registered gate matrix reproduces **exactly**, arm by arm and gate by gate, from independently recomputed quantities;
- weights and thresholds are provably leakage-free: 180/180 inner thresholds equal the quantile of exactly that split's inner-training rows, and corruption tests return **bit-identical** coefficients and predictions;
- the 325-fit figure reconciles arithmetically (180 + 120 + 25) and matches the audit trail exactly, with no duplicates or omissions;
- E0 is the frozen Phase 3 Ridge prediction vector, bit-for-bit, never refitted;
- no Phase 1, 2 or 3 artefact carries a post-Phase-3 timestamp.

The limitations are real but do not overturn any conclusion: one analysis (the bootstrap) was executed on a **different cluster unit than the pre-registration specifies**; two tests the proposal required were never written; one required quantile diagnostic (a binomial interval) is missing; one range in the prose is wrong (6.58 vs the actual 6.25); and the reports omit two required tail-calibration columns. Each is itemised in §13 with its consequence.

**No challenger satisfies the registered promotion criteria.** That conclusion is robust to every discrepancy found, because the failing gates (C1, C2) fail by wide margins that no reconstruction choice could reverse.

---

## 2. Scope and inspected artefacts

**Source audited:** `src/models/tail_aware.py` (380 lines), `src/training/run_tail_experiments.py` (241), `src/evaluation/tail_evaluation.py` (516), `tests/test_phase4_isolation.py` (283).

**Documents audited:** `reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md`, `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md`, `reports/ml_report/PHASE_4_VERIFICATION.md`, `Admin/EXPERIMENT_LOG.md` (entry E-22).

**Outputs audited (13):** `tail_oof_predictions.csv` (3,140 rows), and `tail_experiments`, `tail_fold_results`, `tail_state_results`, `tail_tail_metrics`, `tail_gate_results`, `tail_quantile_metrics`, `tail_quantile_reliability`, `tail_bootstrap`, `tail_error_by_predicted_decile`, `tail_fit_diagnostics` (325 rows), `tail_selected_params.json`, `tail_runtime.csv`.

**Cross-referenced:** `datasets/processed/district_analysis.csv`, `evaluation/predictions/phase3_oof_predictions.csv`, `Admin/MODEL_DECISIONS.md`, `Admin/METHODOLOGY.md`, `Admin/PROJECT_SCOPE.md`, `README.md`.

**Auditor's scratch code** (isolated directory, no project artefact touched): `_audit/audit_phase4.py`, `_audit/corrupt_test.py`, `_audit/audit_log.txt`.

**Environment note.** `py` on this machine now resolves to a Python **3.15** installation that has no numpy/pandas/scikit-learn. Phase 4 was executed on Python **3.14.3** (numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1), which is still installed but must be requested explicitly with `py -3.14`. Every command in the Phase 4 documents (`py src/...`) therefore **fails as written** on this machine today. This is environment drift, not a project defect, but it is a reproducibility hazard for anyone replaying the work.

---

## 3. Implementation audit (AUDIT A)

| Claim | Independent finding | Status |
| --- | --- | --- |
| 628 districts, 77 predictors | Reconstructed configuration gives exactly 628 × 77 | VERIFIED |
| Approved target | `target_deprivation_rate_pct`; equals the value in `district_analysis.csv` bit-for-bit | VERIFIED |
| Outer folds are the frozen Phase 3 folds | A fresh `GroupKFold(5)` on the 35 state labels reproduces the stored `fold` column with **0 mismatches**; all five arms use that same column | VERIFIED |
| Whole states held out | No state appears in both the train and test side of any fold (6/7/7/8/7 states held out) | VERIFIED |
| E0 is the frozen Phase 3 Ridge, not refitted | `E0` in the evaluation is `np.array_equal` to the Phase 3 `ridge`/`B_augmented_628`/`grouped` vector; `tail_oof_predictions.csv` contains **no E0 rows** (5 arms × 628) | VERIFIED |
| Arms match the proposal | E1 (q75, λ=2), E2 (q75, λ=4), E3 (q90, λ=4), E4a (τ=0.75), E4b (τ=0.90); grids exactly {30,100,300} and {0.001,0.01,0.1}; no code path widens them; the runner rejects unregistered names | VERIFIED |
| Sample weights actually reach fitting | Weights are passed as `model__sample_weight`; the fitted pipeline's coefficients equal a direct weighted `Ridge` on the same design to < 1e-10 | VERIFIED |
| Hyperparameter selection correct | In all 25 fold×arm cells the recorded α is the argmin of the recorded mean inner score (recomputed independently) | VERIFIED |
| Preprocessing isolated per fit | `SimpleImputer` + `StandardScaler` live inside the pipeline, so they are fitted on each fit's training rows only; no global scaling exists anywhere | VERIFIED |
| No district leakage | 628 unique keys per arm, no duplicates, full OOF coverage | VERIFIED |

**One structural deviation from the proposal's letter, correctly justified.** The proposal anticipated `GridSearchCV` with careful weight slicing; the implementation hand-rolls the inner search because `GridSearchCV` accepts one weight vector per search and therefore cannot recompute a threshold per inner split. The hand-rolled version is the safer design and is what the proposal actually required functionally. Verified: it uses `GroupKFold(4)` on outer-training states only.

---

## 4. Weight-isolation audit (AUDIT B)

This was tested structurally (does the code path admit the value at all?) and numerically (do the recorded values equal the value they should?).

**Structural.** `sample_weights()` takes a target vector as its only data input; both call sites pass a slice of training rows:
- inner loop: `y_fit = y_train[inner_train]` → threshold from inner-training rows only;
- outer refit: `weights, threshold = sample_weights(y_train, ...)` **after** `inner_search` returns, i.e. after α is fixed.

No threshold, weight vector, α or preprocessing statistic is computed from the full label vector, and no weight vector is cached across splits or folds.

**Numerical (independent).** For every one of the **180 weighted inner fit records** the auditor recomputed the threshold as `np.quantile(y[inner_train_rows], q)` using its own fold reconstruction:

| Check | Result |
| --- | --- |
| inner threshold == quantile of that split's inner-TRAIN targets | **180/180**, 0 failures |
| inner `n_train` == that split's training size | 180/180 |
| inner `n_weighted` == count at/above threshold | 180/180 |
| inner `weight_mean` == 1.0 | 180/180 |
| inner thresholds coinciding by chance with the *outer*-train quantile | 9/180 (5%) — consistent with a discrete target, not with a shared threshold |
| outer-refit threshold, `n_train`, `n_valid`, α vs fold definition | 25/25 records consistent |

**Corruption tests, reproduced from scratch** (`_audit/corrupt_test.py`, fold 1, arms E2 and E3):

| Corruption | Observed |
| --- | --- |
| Outer test targets → 999 (E2) | α 100 → 100; threshold 6.575 → 6.575; n_weighted 126 → 126; **coef max\|Δ\| = 0.000e+00**; intercept Δ = 0.000e+00; **prediction max\|Δ\| = 0.000e+00** |
| Outer test targets → 999 (E3) | α 300 → 300; threshold 9.200 → 9.200; n_weighted 53 → 53; coef Δ = 0.000e+00; prediction Δ = 0.000e+00 |
| Inner-validation targets → 999 (E2, split 1) | threshold 6.525 → 6.525 at all three α; n_weighted 94 → 94; weights identical; **inner-split coefficients at fixed α: Δ = 0.000e+00** |

The last row closes a gap in the shipped test, which checks recorded diagnostics but not the inner-split coefficients (see D11).

**Thresholds genuinely track the training fold.** Distinct outer-refit thresholds: E1/E2 {6.250, 6.575, 6.700, 7.100}; E3 {8.700, 8.790, 9.200, 9.680}. A global constant would produce a single value.

**Verdict: no leakage found at either stage.** The specific mechanism the proposal feared — weights computed once on the complete outer training set, letting the threshold see the rows it scores — is absent.

---

## 5. Fit-count reconciliation (AUDIT C)

Reconstructed from the implementation, not from the report:

| Component | Arithmetic | Fits |
| --- | --- | ---: |
| Weighted Ridge inner search | 3 arms × 5 folds × 3 α × 4 inner splits | 180 |
| Quantile inner search | 2 arms × 5 folds × 3 α × 4 inner splits | 120 |
| Outer refits | 5 arms × 5 folds | 25 |
| **Total** | | **325** |

The stored audit trail contains exactly 325 rows, 65 per arm (60 `inner` + 5 `outer_refit`), no duplicate `(arm, fold, stage, inner_split, alpha)` key, `inner_split` covering 1–4, and only the registered α values. Exactly 5 rows per arm carry `selected=True`.

**One accounting discrepancy, in the proposal rather than the work:** the proposal's cost table estimated **300** fits and "≈5–10 min", because it counted only the 300 tuning fits and omitted the 25 outer refits. The report's **325** is correct; the proposal's estimate was the low one. Measured wall time is ~8.7 s for the sweep, well inside the estimate.

**Verdict: 325 is arithmetically entailed and fully evidenced in the trail.**

---

## 6. Independent metric reproduction (AUDIT D)

Formulas used by the auditor: MAE = mean|y−ŷ|; RMSE = √mean(y−ŷ)²; R² = 1 − SS_res/SS_tot; bias = mean(y−ŷ); tail groups by the frozen thresholds y ≥ 9.200 (n=64) and y ≥ 11.200 (n=35); flag = ŷ ≥ 9.200; sensitivity = flagged∧(y≥11.2)/(y≥11.2); precision = flagged∧(y≥11.2)/flagged.

| Arm | MAE | RMSE | R² | ρ | p90 MAE | p90 bias | p95 MAE | p95 bias | Sens | Prec | Flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 1.7496 | 2.2842 | 0.4680 | 0.6672 | 3.6481 | 3.5935 | 4.5273 | 4.5096 | 0.3714 | 0.5200 | 25 |
| E1 | 1.9150 | 2.4308 | 0.3976 | 0.6152 | 3.0324 | 2.9346 | 3.7855 | 3.6926 | 0.5429 | 0.3878 | 49 |
| E2 | 1.9750 | 2.5253 | 0.3498 | 0.6017 | 2.8315 | 2.7241 | 3.5451 | 3.4353 | 0.6857 | 0.3871 | 62 |
| E3 | 1.9848 | 2.5314 | 0.3467 | 0.6312 | 2.8066 | 2.5012 | 3.5298 | 3.2707 | 0.6000 | 0.3000 | 70 |
| E4a | 1.9142 | 2.4758 | 0.3751 | 0.6721 | 2.6236 | 2.3033 | 3.4308 | 3.1088 | 0.6286 | 0.3235 | 68 |
| E4b | 2.9118 | 3.5736 | −0.3020 | 0.5452 | 2.3305 | 1.8267 | 3.1018 | 2.9937 | 0.5429 | 0.1696 | 112 |

**Discrepancy table:**

| Metric | Reported value | Recomputed value | Difference | Status |
| --- | --- | --- | --- | --- |
| All 30 headline cells × 6 arms | as in report §4.1/§4.2/§4.5 | as above | 0.00e+00 | MATCH |
| Spearman (6 arms) | report §4.1 | recomputed | 0.00e+00 | MATCH |
| 30 fold-level MAE values | `tail_fold_results.csv` | recomputed | 0.00e+00 | MATCH |
| 210 state-level biases | `tail_state_results.csv` | recomputed | 0.00e+00 | MATCH |
| Trade-off ratios 0.202/0.210/0.190/0.118/0.767 | report §4.3 | 0.2024/0.2099/0.1899/0.1175/0.7667 | < 5e-4 (rounding) | MATCH |
| E3 fold-2 MAE "1.9113 vs 1.9111, worse by 0.0002" | report §4.1 | 1.911322 vs 1.911138 | 0.000184 | MATCH |
| E4b fold-3 p95 bias "4.467 vs 3.028" | report §4.2 | 4.467111 vs 3.028018 | 0.00e+00 | MATCH |
| Runtime "11.7 s total" | report §2 | 8.652 s in the stored CSV | −3.05 s | **NOT REPRODUCIBLE** (D7) |
| q75 threshold range "6.58–7.10" | report §2, verification §4, E-22 | **6.250–7.100** | lower bound wrong | **MISREPORTED** (D2) |

All arms are evaluated on the identical 628 districts under the identical fold assignment — confirmed, not assumed.

---

## 7. Quantile regression audit (AUDIT E)

**Objective.** `QuantileRegressor(quantile=τ, alpha=α, solver="highs")` on L1 loss; τ is read from the registered arm and equals 0.75 (E4a) / 0.90 (E4b) in both code and the parameters file.

**Pinball loss verified against an independent definition.** Brute-force piecewise `mean(max(τr, (τ−1)r))` agrees with the shipped `pinball_loss` to < 1e-12 at τ ∈ {0.1, 0.5, 0.75, 0.9}, and `pinball(τ=0.5) == 0.5·MAE` exactly. The implementation is correct.

| Metric | E4a (τ=0.75) | E4b (τ=0.90) | E0 reference at same τ |
| --- | ---: | ---: | ---: |
| Pinball (recomputed) | 0.746461 | 0.541137 | 0.8978 / 0.9116 |
| Coverage = share(y ≤ ŷ) | 0.670382 | 0.824841 | 0.5000 (n/a as a quantile) |
| Calibration error (τ − coverage) | +7.96 pp | +7.52 pp | +25.0 / +40.0 pp |
| Reliability MAD (decile-weighted) | 11.78 pp | 8.63 pp | — |
| Mean predicted vs actual (5.058) | 5.900 | 7.345 | 4.966 |

Both coverage figures reproduce to 1e-9. The reported numbers 0.670 and 0.825 are correct.

**Interpretation — the crucial distinction.** These are conditional-quantile estimators, not conditional-mean estimators:
- lower pinball loss shows the model beats the reference *at the loss it minimises*; it says nothing by itself about calibration;
- an empirical coverage of 0.670 at τ = 0.75 and 0.825 at τ = 0.90 means both **under-cover by ~8 points**: they are *not* calibrated conditional quantiles. E4b promises the 90th percentile and delivers the 82.5th;
- E4b's pooled R² of −0.302 (fold-mean −0.531) and its mean prediction of 7.35 against an actual mean of 5.06 confirm it is a shifted upper-quantile surface, not a mean predictor. It also **extrapolates to 29.44** while the target maximum is 17.7 — an extrapolation the report does not mention (D13).

The report's own framing is correct: it labels E4a/E4b as structurally non-comparable on MAE/R², reports pinball loss, coverage, calibration error and reliability separately, and does not claim calibration. **The description "calibrated conditional quantile model" would not be justified for either arm**, and the report does not use it.

---

## 8. State-clustered bootstrap audit (AUDIT F)

**Mechanics — verified.** `clustered_bootstrap` resamples the 35 state labels with replacement (`rng.choice(uniq, size=35, replace=True)`), concatenates each drawn state's district indices, and evaluates **both** the arm and the E0 reference on that same resample before differencing — so the comparison is genuinely paired. Intervals are the 2.5th/97.5th percentiles of the differenced draws. Seed 42, 2,000 repetitions, both recorded in the CSV (`cluster_unit=state`, `reps=2000`). Cells with fewer than 5 tail districts in a resample are dropped (`n_effective_reps` 1,977–2,000), which is disclosed in the file but not in the report.

**Independently replicated.** The auditor reimplemented the algorithm and reproduced every reported interval to **|Δ| = 0.0000**, including the two specifically requested:

| Quantity | Reported | Auditor | Status |
| --- | --- | --- | --- |
| E1 ΔMAE | +0.165 [+0.022, +0.327] | +0.165 [+0.022, +0.327] | MATCH |
| E1 Δp95 bias | −0.817 [−1.077, −0.487] | −0.817 [−1.077, −0.487] | MATCH |

**However: this is not the registered procedure.** The proposal (§6, line 163) specifies a *"paired **district-level** bootstrap (2,000 resamples)"*, and §8 risk 7 accordingly warns the intervals "are optimistic because they ignore fold assignment and spatial dependence". The executed analysis substitutes a state-clustered bootstrap — a defensible and arguably better choice given that 24 of 35 extreme districts sit in two states, and the Phase 4 report does disclose it — but it is a **post-registration change to a pre-registered analysis that was not recorded as an amendment**.

Consequence, quantified under the registered design (same seed, same reps):

| Arm | Quantity | District-level (registered) | State-clustered (executed) |
| --- | --- | --- | --- |
| E1 | ΔMAE | [+0.115, +0.215] | [+0.022, +0.327] |
| E1 | Δp95 bias | [−0.961, −0.682] | [−1.077, −0.487] |
| E4a | ΔMAE | [+0.078, +0.250] *(excludes 0)* | [−0.023, +0.374] *(includes 0)* |
| E4b | Δp95 bias | [−2.404, −0.553] *(excludes 0)* | [−3.782, +1.251] *(includes 0)* |

Intervals widen by roughly 2–3×. The direction is **conservative** — the executed analysis never manufactures significance — so no conclusion is inflated. But two statements in the report depend on the choice: "E4a is the only arm whose overall cost interval includes zero" and "E4b's tail interval spans zero: its apparent gain is not stable" would both read differently under the registered design. Both should carry the qualifier "under the state-clustered bootstrap".

**What the intervals establish:** for E1–E3, a real and consistent trade — tail bias falls and overall MAE rises, both excluding zero even under the conservative clustering. **What they do not establish:** anything about fold-partition uncertainty, spatial dependence *within* a state, or anything causal.

---

## 9. Extreme-district audit (AUDIT G)

| Check | Result |
| --- | --- |
| p90 = 9.200 | Recomputed exactly; n = 64 |
| p95 = 11.200 | Recomputed exactly; n = 35 |
| Per-fold p90 counts | 11 / 20 / 21 / 10 / 2 — matches the proposal table |
| Per-fold p95 counts | 4 / 12 / 14 / 5 / 0 — matches; fold 5 contributes nothing to the tail |
| Same threshold used everywhere | Verified: evaluation asserts it at load, and no per-arm re-derivation exists |
| State distribution (≥ p95) | Assam 12, West Bengal 12, UP 4, Orissa 4, Bihar 2, Arunachal Pradesh 1 |
| Two-state concentration | **68.6%** — the report's "24 of 35" and "approximately 69%" are accurate |
| Sensitivity / precision | Recomputed exactly for all arms (see §6) |

**State-level findings verified against `tail_state_results.csv`:**

| State | E0 | E1 | E2 | E3 | E4a | E4b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assam | +3.26 | +2.42 | +2.13 | +2.12 | +2.43 | **−0.31** |
| West Bengal | +1.71 | +1.09 | +0.99 | +0.34 | +0.20 | **+2.79** |
| Punjab | +2.13 | **+2.53** | **+2.62** | +2.15 | +1.41 | +1.22 |
| Sikkim | +2.75 | +2.09 | +1.84 | +2.30 | +1.65 | +1.32 |

The report's claims are accurate: Punjab deteriorates under E1 and E2; E4b over-corrects Assam into a slight over-prediction while making West Bengal worse; E3 and E4a spread the gain most widely. The report states plainly that "improves the tail" is partly "improves Assam", and that a one-state effect cannot be separated from a deprivation effect at this sample size. **Geographic concentration and the associated limitation are represented honestly, not smoothed away.**

---

## 10. C1–C6 gate verification (AUDIT H)

Thresholds reconstructed from the proposal's own multipliers applied to the E0 reference: C1 ≤ 3.1009 (p90 MAE, −15%) and ≤ +3.1567 (p95 bias, −30%); C2 ≤ 1.8370 (MAE) and ≤ 2.3984 (RMSE, +5%); C6 sensitivity ≥ 0.4714 and precision ≥ 0.40. No threshold was altered after results were seen.

| Gate | Criterion | E1 | E2 | E3 | E4a | E4b |
| --- | --- | --- | --- | --- | --- | --- |
| **C1** | p90 MAE ≤ 3.101 **and** p95 bias ≤ +3.157 | FAIL (3.032 ✓ / 3.693 ✗) | FAIL (2.831 ✓ / 3.435 ✗) | FAIL (2.807 ✓ / 3.271 ✗) | N/A | N/A |
| **C2** | MAE ≤ 1.837 **and** RMSE ≤ 2.398 | FAIL (1.915 / 2.431) | FAIL (1.975 / 2.525) | FAIL (1.985 / 2.531) | N/A | N/A |
| **C3** | improvement in ≥3 of 4 tail folds | PASS 4/4 | PASS 4/4 | PASS 4/4 | PASS 4/4 | PASS 3/4 |
| **C4** | ≥2 states improved, none >60% | PASS 56.3% | PASS 57.0% | PASS 49.1% | PASS 36.6% | FAIL 63.2% (WB worse) |
| **C5** | p95 bias ≥ −1.0 | PASS +3.693 | PASS +3.435 | PASS +3.271 | PASS +3.109 | PASS +2.994 |
| **C6** | sens ≥ 0.471 **and** prec ≥ 0.40 | FAIL 0.39 | FAIL 0.39 | FAIL 0.30 | FAIL 0.32 | FAIL 0.17 |

**The independently derived matrix is identical to the reported one, cell for cell (30/30).**

- **N/A on C1/C2 for E4a/E4b is justified.** C1/C2 are defined on MAE/RMSE of a conditional mean; a τ-quantile estimator does not target that functional, so scoring it there measures the estimand, not the model. The proposal predetermines this treatment and requires the raw values to be reported anyway — they are, in `observed`. Justification: **valid**.
- C1 fails for all three weighted arms because the p90 half passed and the p95 half did not — the report states exactly this and does not claim a partial pass.
- C6 is officially "a practical read-out, not a gate", yet the implementation's verdict function requires C5∧C6 for an E4 arm to be called a "specialised targeting tool". That rule is **invented in code** — the proposal defines candidate status only via C1–C5 and never states what E4 would need (D12). Since C6 fails for both E4 arms, the outcome ("not accepted") is the same either way, so the ambiguity is immaterial here but should be closed before reuse.

**Promotion check.** Candidate status requires C1–C5. No arm has C1 and C2 both PASS. **No challenger was eligible for promotion at any point.** The report's "no challenger is promoted" is correct and, if anything, understated — there is no near-miss.

---

## 11. Scientific-validity assessment (AUDIT I)

1. **Do the tail-aware approaches genuinely improve extreme-district errors?** Partly, and measurably. p95 bias falls from +4.510 to +3.693 (E1), +3.435 (E2), +3.271 (E3), +3.109 (E4a), +2.994 (E4b); p90 MAE falls from 3.648 to 2.331–3.032. The direction holds in 4/4 tail folds for E1–E4a. This is a real effect, not noise, and the report does not hide it.
2. **Is that at a material overall cost?** Yes. MAE rises +9.4% to +13.5% (E4b +66%); every arm is worse in essentially every fold. Both effects exclude zero under the conservative clustering. The "trade ratio" of 0.12–0.21 pp MAE per 1 pp of tail bias is a fair summary.
3. **Does sensitivity come at precision's expense?** Directly. Sensitivity rises 0.371 → 0.543–0.686, but the flagged set grows 25 → 49–112 and precision falls 0.52 → 0.17–0.39. Every arm violates the 0.40 precision floor. The report states this rather than leading with sensitivity.
4. **Does the evidence support retaining Ridge B as primary?** Yes. It has the best MAE, RMSE, R², the highest precision, and no competitor clears the cost gate. Retaining it is the evidence-led choice, not inertia.
5. **Does the report overstate the evidence?** No. It separates what was implemented, executed, verified and demonstrated, reports every arm including the failures, includes the mean cost, and never converts a gate failure into a partial success. Two omissions bias the *reader's picture*, not the verdict: overall bias (which flips negative for every arm, to −2.29 for E4b) and the p95 calibration gap (mean predicted 10.24 vs actual 13.23 for the best arm) are absent from the prose though present in the CSVs (D5).
6. **Is the "limited predictor information" conclusion sufficiently cautious?** Yes, verbatim and correctly scoped. The mandated sentence appears once, unaltered, including "cannot be established from these experiments alone" and "would be required to establish an information ceiling". The report adds a reconciliation paragraph making clear that "no improvement" means "no improvement sufficient to meet the registered criteria", not "no change in the tail" — which is the accurate reading and prevents both over- and under-claiming. No information ceiling is claimed anywhere.
7. **Is the project correctly framed?** Yes. The target is the Census-computed share of dilapidated census houses at district level. A repository-wide search confirms the reframing is maintained: `README.md`, `Admin/PROJECT_SCOPE.md`, `Admin/MODEL_DECISIONS.md`, `Admin/DATASET_AUDIT.md`, `src/features/target.py` and the Phase 3 report all state explicitly that this is **not** a homelessness measure. No document claims individual or household homelessness. The only weakness is that the two Phase 4 documents never name the outcome at all — they refer to "`target_deprivation_rate_pct`" and "extreme districts" — so a reader arriving at Phase 4 alone would not know what is being predicted (D6).

---

## 12. Phase-boundary integrity (AUDIT J)

| Area | Finding |
| --- | --- |
| Phase 1 artefacts | No file in `evaluation/metrics/phase1_*` or `datasets/processed/` postdates Phase 3 (`district_analysis.csv` 15:35, `phase2_*` 15:35) |
| Phase 2 artefacts | 15:35, untouched |
| Phase 3 metrics | All 16:23, `permutation_importance` 16:24, `coefficient_stability` 16:34 — none later |
| Phase 3 OOF predictions | 16:23, byte-unchanged; used read-only as E0 and verified `array_equal` |
| Phase 3 decisions | `MODEL_DECISIONS.md`, `PHASE_3_MODEL_DECISIONS.md`, `METHODOLOGY.md` all 16:33 |
| Phase 4 altering earlier work | No evidence. Phase 4 writes only `tail_*` files |
| Unapproved Phase 5 work | None. `reports/ml_report/` holds only the Phase 3 and Phase 4 documents; no Phase 5 source, metric or prediction exists |
| Experiment log accuracy | E-22 describes the work actually performed — configuration, folds, seed, grids, α, gate outcomes, isolation, outputs, runtime. One numeric error (D2). No existing entry was altered |

**Limitation stated plainly: full immutability cannot be proven.** There is no git repository in this workspace, no manifest of pre-Phase-4 hashes, and no external reference copy. The evidence for boundary integrity is therefore timestamps plus the fact that no Phase 4 code path writes outside `tail_*`. A sufficiently careful edit that preserved mtimes would be undetectable by this audit. Timestamps are consistent and mutually corroborating, but they are not cryptographic proof.

---

## 13. All identified discrepancies

| # | Severity | Finding | Consequence |
| --- | --- | --- | --- |
| **D1** | Medium | Bootstrap executed as **state-clustered**; the pre-registration specifies **district-level**. Documented in the Phase 4 report but not recorded as an approved amendment to the proposal, which still reads "district-level" and still carries the matching caveat. | Intervals 2–3× wider than registered. Conservative, so no conclusion is inflated, but two report statements ("E4a is the only arm whose cost interval includes zero"; "E4b's tail gain is not stable") hold **only** under the executed clustering. Under the registered design E4a's cost and E4b's tail gain are both significant. |
| **D2** | Low | q75 threshold range reported as **6.58–7.10**; actual is **6.250–7.100** (fold 3 = 6.250). Appears in Phase 4 report §2, PHASE_4_VERIFICATION §4, and E-22. | Factual error in prose. No effect on results, α, weights or gates — `tail_fit_diagnostics.csv` and `tail_selected_params.json` both carry the correct 6.250. |
| **D3** | Medium | Two tests required by the proposal (§11) were never written: **pinball/coverage against a synthetic case with known calibration**, and **bootstrap reproducibility**. Repository search finds no pinball, coverage, calibration or bootstrap test anywhere. | The behaviours are correct — the auditor verified both independently (§7, §8) — but no regression guard exists. A future edit could silently break the pinball definition or the clustering with the suite still green. |
| **D4** | Low | No **binomial interval** for quantile coverage, although proposal §6 requires one that "acknowledges n = 35". `tail_quantile_metrics.csv` has point coverage and subgroup coverage only. | Coverage is reported (and is correct) but without uncertainty, which matters most exactly where n = 35. |
| **D5** | Low | The report omits two required tail-calibration outputs present in the CSVs: overall signed bias (`all_bias`) and the mean-predicted-vs-mean-actual gap in each tail group. | Material nuance lost: **every** arm's overall bias flips negative (E0 +0.09 → E4b −2.29), i.e. the arms purchase tail-bias reduction partly by over-predicting the bulk; and even the best arm leaves the p95 group at 10.24 predicted vs 13.23 actual. |
| **D6** | Low | The two Phase 4 documents never name the outcome (prevalence of dilapidated census houses). | A Phase-4-only reader cannot tell what is predicted. No misdescription exists anywhere in the repository. |
| **D7** | Info | Report quotes "11.7 s total" and per-arm seconds (0.37/0.34/0.43/4.33/3.56); the stored `tail_runtime.csv` now sums to **8.652 s** with different per-arm values. | `tail_runtime.csv` records the most recent execution, so the quoted figures are no longer recoverable from it. Timing only; all data artefacts are byte-identical. |
| **D8** | Info | `py` now resolves to Python **3.15** without numpy/pandas/sklearn. | Every documented command (`py src/...`, `py tests/...`) fails as written; `py -3.14` is required. Environment drift on the host, not a project change. |
| **D9** | Info | The proposal estimated 300 fits; the implementation performs 325. | 325 is correct; the proposal's cost table omitted the 25 outer refits. The report's figure is right and the proposal's estimate was low. |
| **D10** | Info | The proposal names the deliverable `reports/ml_report/EXTREME_DISTRICT_REMEDY_REPORT.md`; the actual file is `PHASE_4_TAIL_REMEDY_REPORT.md`. The audit brief refers to `EXPERIMENT_LOG.md`; the file is `Admin/EXPERIMENT_LOG.md`. | Naming drift only; both files exist and are complete. |
| **D11** | Info | The inner-isolation test asserts thresholds and weights with `np.isclose` and never checks the fitted **coefficients**, although the proposal demanded bit-identity including coefficients. | Test weaker than specified. The auditor closed the gap manually: inner-split coefficients under validation corruption differ by **0.000e+00**. |
| **D12** | Info | The rule for calling an E4 arm a "specialised targeting tool" (C5 ∧ C6) exists only in code; the proposal defines candidate status via C1–C5 and never specifies the E4 path. | Underspecified in the pre-registration. Immaterial here because C6 fails for both E4 arms under either reading, but the rule should be written down before it is reused. |
| **D13** | Low | E4b's predictions range from 1.91 to **29.44** against a target range of 0.2–17.7, with mean prediction 7.35 vs actual 5.06. The extrapolation is not mentioned. | A reader could over-read E4b's tail numbers. It is an upper-quantile surface extrapolating far outside the observed range — relevant to why its MAE and R² collapse. |

**Nothing in this list changes the gate outcomes.** C1 fails by 0.11–0.54 pp of bias and C2 by 4.4–13.5 percentage points of MAE; no reconstruction, re-clustering or re-reporting could close those margins.

---

## 14. Limitations of this audit

1. **Immutability is not proven.** No git history, no pre-Phase-4 hash manifest. Boundary integrity rests on timestamps and on the absence of any Phase 4 write path outside `tail_*`.
2. **The audit reproduced the implementation, not an independent specification of it.** Where the auditor refitted arms, it did so with the project's own model code; agreement therefore proves the stored artefacts derive from the shipped code, not that the code is the only correct implementation of the proposal.
3. **Metric reproduction cannot catch a wrong-but-consistent definition.** Every gate and metric was recomputed from the frozen definitions in the proposal; if a frozen definition were itself mis-specified, this audit would not detect it. The p90/p95 thresholds, group sizes and fold counts were independently re-derived from raw targets, which limits this risk but does not eliminate it.
4. **The bootstrap replication reimplements the same algorithm.** Agreement to 0.0000 confirms the code matches its own described procedure and is seeded; it cannot confirm that clustering on states is the right dependence assumption.
5. **No data-level provenance check.** The audit verified the target values against `district_analysis.csv` but did not re-derive them from the original Census sources.
6. **One scratch-script line failed** (a cosmetic loop in `_audit/corrupt_test.py` after all substantive checks had printed). It has no bearing on any finding; the checks it would have summarised are reported from direct output above.
7. **Judgement calls.** Calling D1 "medium" and the overall result "verified with limitations" is the auditor's assessment, not a mechanical output.

---

## 15. Final recommendations

1. **Close Phase 4 as executed.** The work is sound, the results are reproducible to the digit, no leakage exists, and the conclusion is correctly conservative. The limitations are documentation and pre-registration hygiene, not evidence defects.
2. **Amend the proposal record, retroactively and explicitly,** to state that the bootstrap was state-clustered (with the reason) rather than district-level, and add the district-level intervals alongside. This is the one change that would have altered an inference if the pre-registered design had been followed.
3. **Add the two missing tests** — pinball and coverage on a synthetic case with known calibration, and bootstrap reproducibility — so the verified behaviours are guarded. The auditor's independent checks can serve as the specification.
4. **Correct D2 in place** (6.25, not 6.58) and add the missing rows for D5 (overall bias, p95 calibration gap) to the report's tables.
5. **Add one sentence naming the outcome** in the Phase 4 documents (D6), and record the state-clustered qualifier next to the two interval-based statements (D1).
6. **Do not re-run or extend the investigation.** Nothing found here justifies new arms; any further tail work needs a new proposal and separate authorisation.
7. **Do not begin Phase 5.** The audit gives no authority for it.

---

## Verification matrix

| Audit area | Result | Evidence | Explanation |
| --- | --- | --- | --- |
| Data identity (628 × 77, target) | **PASS** | Recomputed config + `district_analysis.csv` join; values bit-equal | Target equals the frozen column exactly |
| Outer fold identity | **PASS** | Fresh `GroupKFold(5)`, 0 mismatches vs stored `fold`; whole states held out | Five arms share one fold column |
| E0 is the frozen Ridge, not refitted | **PASS** | `array_equal` vs Phase 3 vector; no E0 rows in the OOF file | Reference cannot have been silently regenerated |
| Arm definitions vs proposal | **PASS** | Code inspection + params JSON; runner rejects unregistered arms | No grid widened |
| Weights reach the estimator | **PASS** | Pipeline ≡ direct weighted `Ridge`, < 1e-10 | Row-aligned and applied |
| Preprocessing isolation | **PASS** | Imputer/scaler inside the pipeline; no global fit anywhere | Per-fit only |
| Inner threshold isolation | **PASS** | 180/180 thresholds equal `quantile(inner-train)`; 9/180 coincide with outer by chance | Not a shared threshold |
| Outer test isolation (corruption) | **PASS** | α, threshold, weights, coefficients, intercept, predictions all Δ = 0.000e+00 | Bit-identical, as required |
| Inner validation isolation (corruption) | **PASS** | Threshold/weights identical; inner coefficients Δ = 0.000e+00 (auditor-supplied check) | Closes D11 |
| Fit count 325 | **PASS** | 180 + 120 + 25; trail has exactly 325 rows, 65/arm, no duplicates | Arithmetically entailed |
| α selection | **PASS** | argmin of mean inner score in 25/25 fold×arm cells | Selection is genuine |
| Overall metric reproduction | **PASS** | 30/30 cells × 6 arms, max\|Δ\| = 0.00e+00 | Independent formulas |
| Fold-level metrics | **PASS** | 30/30 MAE values, 0 mismatches | — |
| State-level metrics | **PASS** | 210 biases reproduce; Punjab/WB/Assam/Sikkim claims confirmed | — |
| Pinball loss implementation | **PASS** | Matches piecewise definition < 1e-12; `pinball(0.5) = 0.5·MAE` | Correct objective |
| Coverage / calibration error | **PASS** | 0.670382 and 0.824841 reproduce to 1e-9 | Numbers are right |
| Quantile interpretation | **PARTIAL** | Both under-cover by ~8 pp → **not** calibrated; report says so, but omits the binomial interval (D4) and E4b's 29.4 extrapolation (D13) | Reporting gap, not a result error |
| Bootstrap mechanics (clustering, pairing, seed, reps) | **PASS** | Reimplemented; all intervals reproduce to 0.0000; `cluster_unit=state`, `reps=2000` | Internally correct |
| Bootstrap conformance to pre-registration | **FAIL** | Proposal specifies district-level; execution is state-clustered (D1) | Conservative direction, but unregistered; two inferences depend on it |
| Extreme-district definitions | **PASS** | p90 9.200 / p95 11.200, n = 64/35, per-fold 11/20/21/10/2 and 4/12/14/5/0 | Frozen and applied consistently |
| Geographic concentration | **PASS** | 24/35 in Assam + West Bengal = 68.6% | Report accurate |
| Gate matrix C1–C6 | **PASS** | 30/30 cells identical to independent evaluation | See §10 |
| N/A justification (E4 C1/C2) | **PASS** | Different estimand; raw values still reported | Justified |
| Promotion eligibility | **PASS** | C1 and C2 fail for every arm that they apply to | No challenger eligible |
| Scientific conclusions | **PASS** | Tail gain and overall cost both evidenced; precision collapse disclosed | Not overstated |
| "Information ceiling" wording | **PASS** | Mandated sentence verbatim; no ceiling claimed anywhere | Correctly cautious |
| Target framing (dilapidated houses, not homelessness) | **PARTIAL** | Repository-wide correct; Phase 4 documents never name the outcome (D6) | No misdescription exists |
| Phase 1/2/3 boundary | **PASS (timestamp-based)** | No post-Phase-3 mtimes; `phase3_oof` byte-unchanged | Not cryptographic (no git) |
| Phase 5 absence | **PASS** | No Phase 5 source, metric, prediction or report | — |
| Experiment log accuracy | **PARTIAL** | E-22 matches the work; threshold range carries D2 | One numeric error |
| Required test deliverables | **FAIL** | Pinball/coverage synthetic test and bootstrap reproducibility test absent (D3) | Behaviours verified by auditor, unguarded in code |
| Documented commands runnable | **FAIL** | `py` → 3.15 without the stack; needs `py -3.14` (D8) | Environment drift |

---

## Answers to the six required questions

1. **Is Phase 4 safe to formally close?** **Yes.** Implementation, results and conclusions are independently supported; nothing found undermines them. Close it with the five documentation corrections in §15 items 2–5 recorded as errata, not by re-running anything.
2. **Is Ridge B still the primary model?** **Yes.** It holds the best MAE (1.750), RMSE (2.284), R² (0.468) and precision (0.520), and no challenger clears the cost gate. Nothing in Phase 4 displaces it.
3. **Were any challenger models eligible for promotion?** **No.** Candidate status requires C1–C5; every arm fails C1 and C2 (or has them not applicable). There is no marginal case — the closest arm, E4a, is 0.05 pp short on tail bias and 4.4 pp over on overall MAE.
4. **Are the reported results independently reproducible?** **Yes.** All 30 headline metric cells, 30 fold values, 210 state values, 12 quantile figures, 20 bootstrap intervals and 30 gate cells reproduce — the metrics to 0.00e+00, the gate matrix cell-for-cell. Four arms refit from the registered code reproduce the stored OOF predictions with max |Δ| = 0.00e+00.
5. **What issues remain unresolved?** The bootstrap cluster unit versus the pre-registration (D1, medium); two missing required tests (D3, medium); the missing binomial coverage interval (D4); two omitted required report rows (D5); the wrong q75 range (D2); the unstated outcome (D6); E4b's unmentioned extrapolation (D13); and environment drift that breaks the documented commands (D8). None changes a gate or a conclusion.
6. **Is another experiment necessary, or should the project move to final documentation and presentation?** **Move to final documentation and presentation.** The registered hypothesis set was tested once, as designed, and answered: these remedies buy 18–34% of tail bias at a 9–66% overall cost and at the price of flag precision, and none is acceptable. A further experiment would either repeat the same class of remedy or introduce a post-hoc arm, which is precisely what pre-registration exists to prevent. The honest position — partial tail improvement, material cost, no promotion, no information-ceiling claim — is already stated and is defensible as the project's finding.
