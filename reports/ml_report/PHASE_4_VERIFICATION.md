# Phase 4 — Final Verification Report

**Scope:** the authorised Phase 4 tail-aware investigation only.
**Result:** all checks pass. No boundary was breached, and no frozen Phase 3 artefact was modified.
**Companion documents:** `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md` (results), `Admin/EXPERIMENT_LOG.md` (E-22).

---

## 1. Environment

| Item | Value |
| --- | --- |
| Interpreter | `py` → Python 3.14.3 |
| numpy / pandas / scikit-learn / scipy | 2.5.4 / 3.1.1 / 1.9.1 / 1.18.1 |
| Seed | 42 |
| New dependencies | none |
| Device | local Windows host, single process, `n_jobs=1` |

## 2. Boundary checks (executed before any fit)

`preflight()` in `src/training/run_tail_experiments.py` halts with a non-zero exit rather than substituting anything. All passed:

| Check | Result |
| --- | --- |
| Approved target `target_deprivation_rate_pct` present and unchanged | PASS |
| Configuration B = 628 districts | PASS |
| Configuration B = 77 predictors | PASS |
| No missing values in the feature matrix | PASS |
| District keys identical to the Phase 3 out-of-fold predictions | PASS |
| Frozen thresholds still p90 = 9.200, p95 = 11.200 | PASS |

The runner also refuses unregistered arm names; only E1, E2, E3, E4a, E4b can be requested, with their fixed grids.

## 3. Isolation tests — 9/9

`py tests/test_phase4_isolation.py`

| Test | What it proves | Result |
| --- | --- | --- |
| `test_registered_arms_are_exactly_the_five_proposed` | No arm or grid was widened | PASS |
| `test_weight_values_match_registration` | E1 → 0.667/2.0, E2 → 0.5/2.5, E3 → 0.714/3.571 | PASS |
| `test_weights_are_normalised_and_bounded` | Mean 1, positive, monotone at the threshold | PASS |
| `test_weights_reach_the_estimator_as_a_row_aligned_vector` | Weighted pipeline ≡ direct weighted `Ridge` on the same design (coef/intercept within 1e-10) | PASS |
| `test_inner_validation_targets_cannot_reach_that_split_weights` | Overwriting an inner split's validation targets leaves that split's threshold, weighted-row count and weight mean unchanged | PASS |
| `test_inner_search_is_deterministic` | Repeated inner search selects the same α and produces identical scores | PASS |
| `test_outer_test_targets_cannot_reach_any_fold_decision` | Overwriting outer test targets leaves α, threshold, weights, **coefficients** and predictions bit-identical (E2 and E4b) | PASS |
| `test_outer_folds_reproduce_the_frozen_phase3_ridge` | Uniform `Ridge(α=100)` on these folds matches the Phase 3 OOF predictions to < 1e-8, and the metric arithmetic agrees to 1e-9 | PASS |
| `test_frozen_evaluation_definitions_are_unchanged` | p90/p95, group sizes (64/35) and per-fold counts (11/20/21/10/2 and 4/12/14/5/0) | PASS |

Note on the inner-isolation test: it compares coefficients at a **fixed α**, because α selection legitimately consumes inner-validation scores. The leakage property it targets — that weights and thresholds never see validation targets — is asserted directly on the recorded thresholds, weighted-row counts and weight means.

## 4. Nested-weight ordering as executed

Confirmed from the audit trail (`tail_fit_diagnostics.csv`, 325 rows = 60 inner + 5 outer-refit per arm):

- 60 inner fits per arm, each with its own threshold computed from **its own** inner-training rows.
- 5 outer refits per arm, with the threshold recomputed from the **complete** outer training set after α was fixed.
- Thresholds differ across folds as expected (E1/E2 q75: 6.250–7.100; E3 q90: 8.700–9.680), which would not happen if a single global threshold had been used. *(Corrected 2026-11-12: the earlier wording said 6.58–7.10; fold 3's threshold is 6.250. See `PHASE_4_TAIL_REMEDY_REPORT.md` §11.2.)*

## 5. Decision-gate integrity

- C1–C6 thresholds were computed from the E0 reference at evaluation time: 3.101 (p90 MAE), +3.157 (p95 bias), 1.837 (MAE), 2.398 (RMSE), 0.471 (sensitivity), 0.40 (precision).
- No gate was relaxed, waived or reordered after seeing results.
- E4a/E4b are marked **NOT APPLICABLE** on C1/C2 with the reason recorded per row; their raw values are still reported. C3–C6 were applied as written to every arm.
- Every arm receives a status of PASS, FAIL or NOT APPLICABLE on every gate — 30 rows in `tail_gate_results.csv`.

## 6. Reproducibility

The full pipeline was re-run from scratch and compared by sha256:

| Artefact | sha256 (first 16) | Identical |
| --- | --- | --- |
| `tail_oof_predictions.csv` | `60229303362619a4` | yes |
| `tail_fit_diagnostics.csv` | `8d4b308dc3573ae2` | yes |
| `tail_experiments.csv` | `efdce41f787c1bb9` | yes |
| `tail_fold_results.csv` | `78e83e3555007604` | yes |
| `tail_state_results.csv` | `a7260cc1cbaf32b2` | yes |
| `tail_gate_results.csv` | `71765015f3710658` | yes |
| `tail_quantile_metrics.csv` | `c5f45a37f037facc` | yes |
| `tail_quantile_reliability.csv` | `f8b7dfde9d082ae7` | yes |
| `tail_tail_metrics.csv` | `12728ab2cf3b8537` | yes |
| `tail_error_by_predicted_decile.csv` | `53e9f0279e3eea14` | yes |
| `tail_bootstrap.csv` | `9f54536d87b29c33` | yes |
| `tail_selected_params.json` (arms block) | `093910ce74c502f0` | yes |

**12/12 byte-identical.** `tail_runtime.csv` is excluded from the comparison because it records wall-clock seconds, which vary between runs by construction; the parameters JSON as a whole is compared on its `arms` block because it also carries a `generated_at` timestamp.

## 7. Frozen Phase 3 integrity

Checked after every execution step; all retain their pre-Phase-4 timestamps:

| Artefact | Timestamp |
| --- | --- |
| `datasets/processed/district_analysis.csv` | 15:35 |
| `evaluation/metrics/phase3_nested_cv_results.csv` | 16:23 |
| `evaluation/metrics/phase3_random_cv_results.csv` | 16:23 |
| `evaluation/metrics/phase3_fold_results.csv` | 16:23 |
| `evaluation/metrics/phase3_model_comparison.csv` | 16:23 |
| `evaluation/metrics/phase3_fold_stability.csv` | 16:23 |
| `evaluation/metrics/phase3_selected_models.csv` | 16:23 |
| `evaluation/metrics/phase3_ab_comparison.csv` | 16:23 |
| `evaluation/metrics/phase3_best_parameters.json` | 16:23 |
| `evaluation/metrics/phase3_permutation_importance.csv` | 16:24 |
| `evaluation/metrics/phase3_coefficient_stability.csv` | 16:34 |
| `evaluation/predictions/phase3_oof_predictions.csv` | 16:23 |

No Phase 3 plot, prediction, metric or decision file was written to. The Phase 3 OOF predictions were **read** as the E0 reference and never rewritten.

## 8. Regression: earlier phases still pass

| Suite | Result |
| --- | --- |
| `tests/test_phase1_contracts.py` | 17/17 |
| `tests/test_phase2_contracts.py` | 19/19 |
| `tests/test_phase3_contracts.py` | 21/21 |
| `tests/test_phase4_isolation.py` | 9/9 |

## 9. Complete file inventory

**Created — source (1,420 lines):**

- `src/models/tail_aware.py` (380)
- `src/training/run_tail_experiments.py` (241)
- `src/evaluation/tail_evaluation.py` (516)
- `tests/test_phase4_isolation.py` (283)

**Created — outputs (13):**

`evaluation/predictions/tail_oof_predictions.csv` (3,140 rows) · `evaluation/metrics/`: `tail_experiments.csv` (6), `tail_fold_results.csv` (30), `tail_state_results.csv` (210), `tail_tail_metrics.csv` (18), `tail_gate_results.csv` (30), `tail_quantile_metrics.csv` (12), `tail_quantile_reliability.csv` (120), `tail_bootstrap.csv` (20), `tail_error_by_predicted_decile.csv` (60), `tail_fit_diagnostics.csv` (325), `tail_selected_params.json`, `tail_runtime.csv` (5).

**Created — documents:**

- `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md`
- `reports/ml_report/PHASE_4_VERIFICATION.md` (this file)

**Modified:**

- `Admin/EXPERIMENT_LOG.md` — E-22 appended. No existing entry altered.
- `reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md` — amended before execution with the three required corrections (ceiling wording, weight isolation, quantile evaluation). Recorded in the previous step.

**Explicitly unchanged:** `src/models/registry.py`, `src/training/nested_cv.py`, `src/training/run_nested_cv.py`, `src/evaluation/{metrics,baselines,compare,error_analysis,coefficient_stability,finalize_phase3}.py`, all Phase 1–3 tests, the target column, the fold assignments, the 77-predictor feature set, `Admin/MODEL_DECISIONS.md`, `Admin/PHASE_3_MODEL_DECISIONS.md`, `Admin/METHODOLOGY.md`, and `reports/ml_report/PHASE_3_MODELLING_REPORT.md`.

## 10. Commands run

```
py tests/test_phase4_isolation.py            # 9/9 before any result was produced
py src/training/run_tail_experiments.py --smoke
py src/training/run_tail_experiments.py      # 325 fits, 11.7 s
py src/evaluation/tail_evaluation.py         # metrics, gates, 2,000-rep bootstrap
py tests/test_phase{1,2,3}_contracts.py      # 17/17, 19/19, 21/21
```

## 11. What this verification does not establish

- It does not establish that the evaluation metrics are the right ones for a policy decision — only that they are the registered ones, computed as specified.
- It does not establish Workers/deployment or UI behaviour; nothing was deployed and no interface was produced.
- No plots were requested or produced, so nothing here is a visual verification.
- The bootstrap is state-clustered but cannot represent dependence between states or the uncertainty of the fold partition itself; intervals are indicative, not exact.
- Passing isolation tests proves the *implementation* honours the weight contract; it does not prove the resulting estimates are unbiased or causal.
