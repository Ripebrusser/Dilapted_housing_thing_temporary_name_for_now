# Experiment Log

Chronological record of every executed step. No model has been trained; entries
below are audits, not experiments in the modelling sense.

Environment: Windows, Python 3.14.3 (`py`), pandas 3.1.1, numpy 2.5.4,
scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.

---

## E-01 — Workspace inventory

**Command:** directory listing and file-size scan of the project root.
**Result:** 13 data files in three owner-supplied folders (`datasets/samrus (1..3)`)
plus three archives. No Python packages, notebooks or docs existed. No Git repository
initialised at capture time.
**Finding:** all four expected Census/slum files are present; the PCA bundle contains
six additional files not anticipated in the brief (including a 2001 extract and a 2016
geography lookup).
**Artefact:** `Admin/DATASET_REGISTER.md`.

## E-02 — Raw-data staging

**Command:** create the project tree; copy (not move) every CSV into
`datasets/raw/{india_census,slum_population,india_pca}` and the archives into
`datasets/downloads/`.
**Result:** 13 files copied; originals intact and unmodified.
**Note:** `docs/` already existed as `Docs/`; the existing folder was kept rather than
creating a case-variant duplicate on a case-insensitive filesystem.

## E-03 — Automated dataset audit

**Command:** `py src/data/audit_datasets.py`
**Result:** 10 datasets loaded (640×118 … 61,857×45). 64 quality checks recorded:
**48 PASS, 15 WARN, 1 FAIL**.
**Findings:**
- D2 is percentage-normalised (1,908/1,908 rows sum to 100) — no recoverable counts.
- D3 is not valid CSV (unquoted commas) and is a code→description map for D2 only.
- D4 is state-level, 32 rows; 4 states/UTs absent.
- D1 has no missing values, no duplicates, no negatives.
**Artefacts:** `datasets/metadata/dataset_manifest.csv`, `column_dictionary.csv`,
`data_quality_report.csv`, `audit_findings.json`,
`reports/dataset_report/dataset_audit_summary.txt`.

## E-04 — Geographic reconciliation

**Command:** `py src/data/audit_geography.py`
**Result:** first pass matched only 606/640 districts. Root cause: state aliases
(ORISSA/ODISHA, PONDICHERRY/PUDUCHERRY) and one district alias
(PONDICHERRY→PUDUCHERRY). After adding `src/data/naming.py`: **640/640 matched, 0
unmatched either way**.
**Further findings:**
- 12 districts share a name with a district in another state → state-qualified keys
  are mandatory.
- D2 has 628 complete `Rural|Total|Urban` triples; 9 urban-only and 3 rural-only keys;
  every key has a `Total` row.
- D5 has 634 district names, 2 rows each, no state column and no district code;
  only **628 D1 districts** can take its household count reliably.
- D10 is 2016 vintage (Telangana present): 629/640 match, 11 Andhra/Telangana
  districts do not, and `New Delhi` is absent.
**Artefacts:** `datasets/metadata/state_alias_map.csv`, `district_crosswalk.csv`,
`geography_findings.json`, `reports/dataset_report/geographic_matching_report.txt`.

## E-05 — Target feasibility (first pass)

**Command:** `py src/data/target_feasibility.py`
**Result:** the proposed target builds cleanly (640 values, no nulls, no zero
denominators, range 0.127–13.871%), but an independent reconstruction from D2 differs
by a median of 1.116 pp with a systematic sign (T1 always lower).
**Diagnostic:** Σ D1 `Households` = 330,835,767 — India had 246,740,228 households and
330,983,625 census houses in 2011. The column is the house stock.
**Artefact:** `datasets/metadata/target_feasibility.json`, `target_candidates.csv`,
`reports/dataset_report/target_feasibility_report.txt`.

## E-06 — Denominator forensics

**Command:** `py src/data/resolve_denominator.py`
**Result:**
- `Rural_Households + Urban_Households` = 330,835,767 = `Households` → same wrong universe.
- Asset columns are true household counts: electricity 165,897,294 (67.2% of 247M),
  TV 116,493,624 (47.2%), latrine-in-premises 115,737,458 (46.9%),
  LPG/PNG 70,422,883 (28.5%) — all within 0.1 pp of published Census 2011 shares.
- D1 population sums to 1,210,854,977 exactly → geography and population are sound.
- 12 districts cannot receive a D5 household count (six names shared by two states).
- `geography.csv` confirms 2016 vintage, 36 states including Telangana.
**Artefacts:** `datasets/metadata/denominator_resolution.json`,
`reports/dataset_report/denominator_resolution_report.txt`.

## E-07 — Corrected target validation

**Command:** `py src/data/target_feasibility.py` (after the ambiguous-key fix)
**Result:**
- T2 (D2 Census share) vs T3 (D1 count ÷ D5 households), n = 628:
  **Pearson 0.9994, Spearman 0.9992, median |diff| 0.053 pp**.
- T1 vs T2: r = 0.9847, median |diff| 1.116 pp (systematic).
- Applying D2's share to D5 households reproduces D1's dilapidated count with median
  absolute relative error 0.86% → the numerator is confirmed as the c14 "Total"
  household count.
**Conclusion:** target is feasible after correction. Recorded as D-001 in
`Admin/MODEL_DECISIONS.md`, pending approval.

## E-08 — Phase 1 contract tests

**Command:** `py tests/test_phase1_contracts.py`
**Result:** see the final verification line in the project hand-off. Tests cover
alias collapse, key uniqueness, the 640/640 join, percentage normalisation of D2,
target/denominator compatibility and the presence of every metadata artefact.

---

## E-09 — District-code cross-check

**Command:** join D1 on `District code` to D2's `Total` rows on `District Code`.
**Result:** 640 rows, **640/640 canonical state+district names agree**; both files
code districts nationally as 1..640 (D2 state codes 1..35). D5 has no district code.
**Consequence:** the name-based canonical key stays the single join path, and the code
join is documented as corroboration. An earlier draft of `DATASET_AUDIT.md` described
the codes as state-local; that was incorrect and has been fixed. A contract test now
asserts both uniqueness and cross-file agreement.

## E-10 — Phase 2 build

**Command:** `py src/data/build_dataset.py`
**Result:** 640-row modelling table; 81 features built, 4 dropped, 77 retained
(43 A + 34 B); target range 0.2–17.7%, 124 distinct values, no nulls; 628 districts
have a D5 household denominator.
**Artefacts:** `datasets/processed/district_analysis.csv`,
`datasets/processed/feature_manifest.csv`,
`evaluation/metrics/phase2_*.csv|json`.

## E-11 — Defects found and fixed during the build

1. `GroupKFold.split` in scikit-learn 1.9 requires `groups=`; the positional call
   put the state array into `y` and raised `ValueError`. Fixed.
2. The checks list was extended with itself (the screen returned the same list it
   appended to), so every check printed twice. Fixed by returning a separate list.
3. **Education composition was invalid.** Summing the seven attainment levels plus
   `Literate_Education` and `Illiterate_Education` gave ratio 1.6878. Diagnosis:
   `sum(levels) / Literate_Education = 1.0000`, i.e. `Literate_Education` is the
   subtotal of the levels. Removed it; levels + `Illiterate_Education` partition the
   table at exactly 1.0000.
4. The first baseline set could not distinguish Configurations A and B, because all
   baselines ignored features and k-NN used six features present in both. Added
   `knn_all_features` so the comparison tests the added features.

## E-12 — Baselines (Phase 2)

**Command:** `py src/data/build_dataset.py` (baseline section)
**Result, grouped by state (primary):** national/state mean MAE 2.440, R² −0.137;
A k-NN all features MAE 2.124 (640) / 2.111 (628); B k-NN all features MAE 1.855,
R² 0.329, Spearman 0.577.
**Result, random 5-fold (mean of 5 seeds):** A 1.483, B 1.364; state-mean baseline
1.556; R² 0.603 → 0.646.
**Finding:** grouped CV is materially harder (MAE 2.12 vs 1.50 for the same model),
because state context carries much of the signal. The state-mean baseline is
exactly equal to the national mean under grouped CV by construction (unseen
states), which is documented rather than "fixed".

## E-13 — Phase 2 contract tests

**Command:** `py tests/test_phase2_contracts.py`
**Result:** 19/19 passing. Covers target reconstruction from D2, key alignment with
D1, manifest coverage, deny-list compliance, A-completeness, B missingness exactly
for the 12 unresolvable districts, no zero-filling, share bounds, absence of a
single feature explaining the target, fold validity and A/B fold identity.

## E-14 — Phase 3 grouped nested CV (primary)

**Command:** `py src/training/run_nested_cv.py --scheme grouped`
**Setup:** 5-fold outer `GroupKFold` on states, 4-fold inner; ten families; 76
candidate configurations per outer fold; seed 42.
**Result:** 30 model-configuration rows, 150 fold rows, 18,960 out-of-fold predictions.
Best per configuration — A 640: `svr_rbf` MAE 2.0293 (R² 0.1699, ρ 0.4626);
A 628: `extra_trees` 2.0888 (R² 0.1837); B 628: `svr_rbf` 1.7248 (R² 0.3992,
ρ 0.6258). Dummy 2.4402 / 2.4495 / 2.4495 reproduces the Phase 2 mean baseline.
**Side check:** grouped folds regenerated from `src.training.splits` are identical to
`phase2_folds.csv`.

## E-15 — Phase 3 random nested CV (reference)

**Command:** `py src/training/run_nested_cv.py --scheme random` (two resumable stages)
**Result:** A 640 `svr_rbf` 1.3257 (R² 0.6661); A 628 1.3390; B 628 1.1401
(R² 0.7316, ρ 0.8319). Grouped-minus-random gap 0.58–0.75 pp, i.e. roughly a third
of the random-CV accuracy is state context.
**Timing:** 570.5 s (A 640 + A 628) and 373.1 s (B 628).

## E-16 — Defects and incidents during Phase 3

1. **Lost sweep stage.** The first `--scheme random` launch was killed when the
   execution environment restarted; the grouped stage had already completed and was
   preserved. Only the random stage was re-run, which is why the runner merges on
   `(configuration, model)` instead of overwriting.
2. **Two incorrect test assertions, not pipeline faults.**
   - `test_inner_tuning_folds_stay_inside_outer_training` compared inner-fold indices
     (positions within the training subset) against outer-test indices (positions in
     the full array). Fixed by mapping inner positions back to global rows before
     intersecting.
   - `test_metric_arithmetic` asserted a negative R² for a toy case whose R² is
     actually 0.55 (SS_res 2.25, SS_tot 5.0). Fixed by asserting the exact value and
     adding a perfect-prediction case.
   Both were assertion errors in newly written tests; no result changed because of them.
3. **Notebook execution dependency.** No Jupyter stack was installed. `nbformat`,
   `nbclient` and `ipykernel` were installed successfully on Python 3.14 and the
   notebooks were then executed in a real kernel.
4. **Shell heredoc truncation.** An inline notebook generator was truncated by the
   shell and failed with a syntax error; regenerated from a temporary script file,
   which succeeded, and the script was then removed.

## E-17 — Phase 3 finalisation, plots and error analysis

**Command:** `py src/evaluation/finalize_phase3.py` (75 s)
**Result:** 30-row comparison table, paired A/B table, selection table, 17 plots,
per-district residuals, errors by state and by target decile, extreme-district
summary, largest errors, permutation importance (3 configurations × 5 folds).
**Key error finding:** the models compress the distribution — A 640 over-predicts the
lowest decile by 2.44 pp and under-predicts the highest by 4.89 pp; 100% of the top
5% are under-predicted (B: 97%, MAE 4.38 vs 1.57 for the remainder).
**Key importance finding:** only 6/43 (A 640), 6/43 (A 628) and 10/77 (B) features are
positive in all five folds; `hh_size_3_share` ranks first on average but is positive
in 2/5 folds with SD 0.150 vs mean 0.066.

## E-18 — Notebook execution

**Command:** `py tests/run_notebooks.py`
**Result:** 5/5 notebooks executed end-to-end in a Jupyter kernel (nbclient) with
outputs saved — 06 (8 code cells), 07 (6), 08 (6), 09 (7), 10 (8); no errors.

## E-19 — Reproducibility audit

- Re-running `finalize_phase3.py` reproduced every metric CSV, JSON and error table
  **byte-for-byte** (sha256 before/after identical).
- Re-running `ridge` end-to-end reproduced its stored grouped MAE to 1e-9.
- Each fold's MAE recomputed from the OOF file using the deterministic fold assignment
  matches the saved fold table to 1e-9 (3 configurations × 2 models).
- Phase 2 outputs retain their original timestamps; raw data untouched.
- Recorded: seed 42, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1,
  Python 3.14.3; fold signatures and per-fold hyperparameters in the meta files.

## E-20 — Ridge coefficient stability on Configuration B

**Command:** `py src/evaluation/coefficient_stability.py`
**Method:** Ridge refitted inside each outer fold with that fold's chosen α, coefficients
read on the standardised scale. Output: `evaluation/metrics/phase3_coefficient_stability.csv`.
**Result:** α = 100 in all five folds; **54/77 predictors (70%) keep their sign in all
five folds**; median SD/|mean| = 0.55. Most stable: `tv_share` −0.473 ± 0.039,
`hh_size_2_share` −0.449 ± 0.026, `electricity_share` −0.394 ± 0.031, `radio_share`
−0.360 ± 0.075, `scooter_share` −0.354 ± 0.057, `hh_size_3_share` +0.313 ± 0.045,
`marginal_worker_share` −0.307 ± 0.036, `other_cooking_fuel_share` +0.301 ± 0.074.
Sign-flipping: `muslim_share` +0.299 ± 0.192 (negative in 1 fold), `sikh_share`,
`primary_share` −0.143 ± 0.130, `bicycle_share`, `literacy_rate_male`, `car_share`.
**Use:** these are predictive associations, not causal effects; unstable coefficients
must not be interpreted individually, and complementary indicators interpenetrate.

## E-21 — Phase 3 model ratification recorded

**Decision (project owner):**
1. Configuration B (628 districts, 77 predictors) is the primary modelling configuration;
   A (640 districts, 43 predictors) is the secondary full-coverage configuration.
2. Ridge on B is the primary interpretable model; Elastic Net on B is secondary
   interpretable; SVR-RBF on B is the retained best-performing benchmark. No statistical
   superiority is claimed.
3. Coefficients and permutation importance may be discussed as predictive associations
   only, with coefficient stability checked first.

**Documentation updated:** `Admin/MODEL_DECISIONS.md` (D-024), `Admin/PHASE_3_MODEL_DECISIONS.md`
(P3-D-13, P3-D-14, statuses of P3-D-06 and P3-D-07), `Admin/METHODOLOGY.md` (§11.8),
`reports/ml_report/PHASE_3_MODELLING_REPORT.md`.
**Not changed:** Phase 1 and Phase 2 data, the approved target, and all completed
Phase 3 metrics, predictions and plots.

## E-22 — Phase 4 tail-aware investigation (authorised)

**Authorisation:** project owner; bounded investigation defined in
`reports/proposals/EXTREME_DISTRICT_REMEDY_PROPOSAL.md`.
**Commands:** `py tests/test_phase4_isolation.py` → `py src/training/run_tail_experiments.py`
→ `py src/evaluation/tail_evaluation.py`.
**Configuration:** `B_augmented_628` only — 628 districts x 77 predictors, approved target
`target_deprivation_rate_pct`. No feature, dataset or transformation was added.
**Folds:** the exact Phase 3 grouped outer folds (5) and inner splits (4); verified by
reproducing the frozen Phase 3 Ridge OOF predictions to < 1e-8.
**Seed:** 42 (no randomness in the weighted arms; bootstrap seeded). **Fits:** 325.
**Runtime:** 11.7 s total (E1 0.37 s, E2 0.34 s, E3 0.43 s, E4a 4.33 s, E4b 3.56 s).

**Parameters actually selected (per outer fold 1-5):**

| Arm | Definition | Selected α | Outer-refit training threshold / weighted rows |
| --- | --- | --- | --- |
| E0 | ratified Phase 3 Ridge, reused | n/a | n/a (not refitted) |
| E1 | q75, λ = 2, w normalised to mean 1 | 100, 30, 100, 300, 300 | 6.250-7.100; 126-129 rows |
| E2 | q75, λ = 4 | 100, 30, 100, 300, 100 | 6.250-7.100; 126-129 rows |
| E3 | q90, λ = 4 | 300, 300, 100, 300, 300 | 8.70-9.68; 51-54 rows |
| E4a | quantile regression τ = 0.75 | 0.01 x5 | n/a |
| E4b | quantile regression τ = 0.90 | 0.01, 0.01, 0.1, 0.01, 0.01 | n/a |

**Isolation verification:** 9/9 tests pass, including weight routing, inner-validation
target override (split weights unchanged), outer-test target override (α, threshold,
weights, coefficients and predictions bit-identical), and fold reproduction.
Every fit is logged in `evaluation/metrics/tail_fit_diagnostics.csv` (325 rows: 60 inner
+ 5 outer-refit per arm).

**Results (pooled OOF, frozen thresholds p90 = 9.200, p95 = 11.200):**

| Arm | MAE | RMSE | R² | ρ | p90 MAE | p90 bias | p95 MAE | p95 bias | sens | prec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 1.750 | 2.284 | 0.468 | 0.667 | 3.648 | +3.593 | 4.527 | +4.510 | 0.371 | 0.520 |
| E1 | 1.915 | 2.431 | 0.398 | 0.615 | 3.032 | +2.935 | 3.786 | +3.693 | 0.543 | 0.388 |
| E2 | 1.975 | 2.525 | 0.350 | 0.602 | 2.831 | +2.724 | 3.545 | +3.435 | 0.686 | 0.387 |
| E3 | 1.985 | 2.531 | 0.347 | 0.631 | 2.807 | +2.501 | 3.530 | +3.271 | 0.600 | 0.300 |
| E4a | 1.914 | 2.476 | 0.375 | 0.672 | 2.624 | +2.303 | 3.431 | +3.109 | 0.629 | 0.324 |
| E4b | 2.912 | 3.574 | -0.302 | 0.545 | 2.331 | +1.827 | 3.102 | +2.994 | 0.543 | 0.170 |

Quantile calibration: E4a pinball 0.746 at τ = 0.75, coverage 0.670 (−7.96 pp);
E4b pinball 0.541 at τ = 0.90, coverage 0.825 (−7.52 pp).
State-clustered bootstrap (2,000 reps): ΔMAE +0.165 [+0.022, +0.327] (E1), +0.225
[+0.041, +0.428] (E2), +0.235 [+0.055, +0.457] (E3), +0.165 [−0.023, +0.374] (E4a),
+1.162 [+0.718, +1.653] (E4b); Δp95 bias −0.817 (E1), −1.074 (E2), −1.239 (E3),
−1.401 (E4a), −1.516 [−3.782, +1.251] (E4b).

**Gates (C1-C6 as registered):** C1 FAIL for E1-E3 (p90 MAE met; p95 bias 3.693/3.435/3.271
against the 3.157 limit), NOT APPLICABLE for E4a/E4b (conditional quantile). C2 FAIL for
every arm (+9.4% to +13.5% MAE; E4b +66%). C3 PASS (4/4 folds for E1-E4a; 3/4 for E4b).
C4 PASS for E1-E4a (4 states, largest share 37-57%), FAIL for E4b (West Bengal worse,
Assam 63% of the reduction). C5 PASS for every arm. C6 FAIL for every arm (precision
0.17-0.39 below the 0.40 floor).
**Decision:** no challenger accepted; no challenger promoted; ratified Phase 3 roles
unchanged. Registered null-result wording recorded in the report, with the measured
0.8-1.4 pp tail-bias reduction also reported. No information ceiling is claimed.
**Reproducibility:** full re-run produced byte-identical artefacts (sha256 on all nine).
**Not changed:** every `phase3_*` artefact, the target, the folds, and the 77-predictor
feature set. No observation was deleted, capped or winsorised.

## Not done (deliberately)

- No cleaning, no imputation, no feature engineering beyond reading column semantics.
- No model, no tuning, no predictions, no plots of results.
- No dataset was downloaded, deleted or overwritten.
- Notebooks 02–11 are placeholders stating exactly that.

---

## Correction round — 2026-11-12 (post-audit)

Executed in response to `reports/ml_report/FINAL_MASTER_AUDIT.md` (verdict C). No model
was re-selected, no fold, threshold, metric or ratified decision was changed, and no
frozen artefact was overwritten. Frozen Phase 3/4 files were hashed before and after.

| ID | Action | Output | Verified |
| --- | --- | --- | --- |
| E-23 | Implemented the **registered district-level** paired bootstrap alongside the executed state-clustered one | `evaluation/metrics/tail_bootstrap_district.csv`, `tail_bootstrap_comparison.csv`, `src/evaluation/bootstrap_paired.py`, `src/evaluation/tail_bootstrap_registered.py` | state path reproduces `tail_bootstrap.csv` to < 1e-9; frozen file sha256 unchanged; district result reproducible at seed 42 |
| E-24 | Corrected the q75 threshold range 6.58–7.10 → **6.250–7.100** in report, verification report and this log | `PHASE_4_TAIL_REMEDY_REPORT.md` §2/§11.2, `PHASE_4_VERIFICATION.md`, `EXPERIMENT_LOG.md` | values re-read from `tail_selected_params.json` (6.575/6.700/6.250/6.700/7.100) |
| E-25 | Added binomial (Wilson + Jeffreys) intervals for quantile coverage | `evaluation/metrics/tail_quantile_coverage_ci.csv`, `src/evaluation/tail_coverage_intervals.py` | every recomputed coverage matches `tail_quantile_metrics.csv` to 1e-12 |
| E-26 | Added overall bias and p95 calibration gap per arm, with sign conventions | `PHASE_4_TAIL_REMEDY_REPORT.md` §11.3 | from `tail_experiments.csv`; documented the bias reversal (E0 +0.092 → E4b −2.287) |
| E-27 | Named the outcome explicitly in Phase 4 documentation | `PHASE_4_TAIL_REMEDY_REPORT.md` §11.1 | "district-level prevalence of dilapidated census houses; NOT individual homelessness" |
| E-28 | Investigated E4b's 29.44 prediction against an observed maximum of 17.7; documented as linear extrapolation, not a defect; **not clipped** | `PHASE_4_TAIL_REMEDY_REPORT.md` §11.6 | solver exact, outputs finite, arm reproduces byte-identically |
| E-29 | Added the two missing regression tests | `tests/test_phase4_quantile_and_bootstrap.py` | 14/14 pass: hand-computed pinball values, direction asymmetry, coverage, tie/boundary cases, paired resampling, seed reproducibility, both interval constructions |
| E-30 | Implemented Phase 6: fitted, persisted and reloaded the ratified Ridge B; built inference and deliverables | `models/final/*`, `evaluation/predictions/final_predictions_ridge_b_628.csv`, `district_ranking_ridge_b_628.csv`, `src/prediction/*` | reload max abs difference 0.0e+00; 628 unique districts; ranking monotone with ranks 1–628 |
| E-31 | Added Phase 6 tests | `tests/test_phase6_inference.py` | 19/19 pass: schema rejections, permuted columns, scaler provenance, ranking, bands, flagged out-of-range values |
| E-32 | Corrected the Phase 2 row-count statement 192/48 → **288/72** | `reports/dataset_report/PHASE_2_DATA_PREPARATION_REPORT.md` | `phase2_baselines.csv` 288 rows; `phase2_baselines_wide.csv` 72 rows |
| E-33 | Rewrote README, PROJECT_SCOPE, config.yaml and requirements.txt to describe the completed project and the verified interpreter | root files | all documented commands executed under `py -3.14` |
| E-34 | Produced the final research report, presentation and viva material | `reports/final_report/FINAL_RESEARCH_REPORT.md`, `reports/presentation/FINAL_PRESENTATION.pptx` (29 slides), `reports/ml_report/VIVA_PREPARATION.md` | every figure traced to a CSV; deck built from the project's own plots |

**Test totals after the correction round:** 17 + 19 + 21 + 9 + 14 + 19 = **99 passing, 0 failed, 0 skipped**.

**Frozen artefacts:** `evaluation/metrics/phase3_*.csv`, `phase2_*.csv`, `tail_*.csv` (original),
`evaluation/predictions/phase3_oof_predictions.csv`, `tail_oof_predictions.csv` — unchanged by hash.
