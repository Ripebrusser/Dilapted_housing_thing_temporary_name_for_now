# Methodology

Phase 1 methodology is *audit and feasibility*. Phases 2–4 methodology is planned
here so that the design is fixed before any model is fitted.

---

## 1. Design

Cross-sectional, district-level, observational study.

- **Unit:** district (n = 640), Census 2011.
- **Outcome:** `target_deprivation_rate_pct` — share of households living in
  dilapidated census houses (D2 `Total` row).
- **Predictors:** district socioeconomic structure from D1 (plus D7–D9 if they earn
  their place).
- **Claim type:** association and ranking, not causation.

## 2. Pipeline

| Stage | Module | Output |
| --- | --- | --- |
| 0. Audit | `src/data/*` | metadata, quality report, target feasibility (done) |
| 1. Ingest | `src/data/loaders.py` (Phase 2) | typed frames with canonical keys |
| 2. Clean | `src/preprocessing/` | validated, no silent coercion |
| 3. Target | `src/features/target.py` | 640-row target vector + provenance |
| 4. Features | `src/features/` | share-based predictors, documented formulas |
| 5. Split | `src/training/splits.py` | grouped-by-state folds, fixed seed |
| 6. Train | `src/training/` | fitted pipelines, checkpoints |
| 7. Evaluate | `src/evaluation/` | metrics, plots, error tables |
| 8. Predict | `src/prediction/` | district scores + metadata |

## 3. Data-integrity rules

1. **Never invent a value.** Missing stays missing; unreported is not zero.
2. **Never join by name alone** when a code exists; `src/data/naming.py` is the only
   place name normalisation is allowed to happen.
3. **Never divide by `D1.Households`.** It is the census-house stock (see
   `DATASET_AUDIT.md` §3).
4. **Never build a feature from the target's own cross-tabulation.**
5. Raw data are read-only; every derived file is regenerable from code.
6. Every processed column carries a formula and a source in `feature_manifest.csv`.

## 4. Validation design

- **Primary:** 5-fold **grouped by state** cross-validation, repeated 5 times with
  distinct seeds. Grouping prevents a model from memorising state-level context and
  then being scored on the same state.
- **Reference:** stratified/random 5-fold, reported alongside, to quantify how much
  of the apparent skill is within-state.
- **Tuning:** nested CV (inner folds for hyperparameters, outer for scoring). No
  test-fold peeking.
- **Baselines that must be beaten:** (a) predict the national mean; (b) predict the
  state mean; (c) k-NN on two or three headline features. A model that does not beat
  the state-mean baseline has no district-level value.
- **Stability:** report fold-to-fold spread, not just the mean score.

## 5. Metrics

Regression (primary): MAE (pp), RMSE (pp), R², Spearman ρ. Ranking metrics are
weighted equally with error metrics because the intended use is prioritisation.

Classification (secondary, optional): if a top-quintile flag is reported, use
ROC-AUC, PR-AUC and precision@20 districts — not accuracy, given class imbalance.

No metric is reported on training data.

## 6. Leakage controls (checklist enforced in code)

- [ ] Target columns dropped by name from the feature frame (explicit deny-list).
- [ ] Deny-list checked at build time; a test fails if a forbidden column appears.
- [ ] No target-derived ratio (Good/Livable shares, structure type) used as a feature.
- [ ] Preprocessing fitted **inside** each CV fold, never on the full dataset.
- [ ] Scaler/imputer statistics come only from the training fold.
- [ ] No feature computed using the full-sample distribution of the target.
- [ ] Planted-leakage test: adding the true target as a feature must push CV R² → 1;
      if it does not, the pipeline is broken.

## 7. Reproducibility

- Fixed seeds (`config.yaml: seed`), recorded in every metrics artefact.
- Deterministic splits saved to disk before training.
- Library versions pinned in `requirements.txt`.
- Relative paths only; no machine-specific absolute paths in code or config.
- Every experiment appended to `Admin/EXPERIMENT_LOG.md` with its command and artefact.

## 8. Error analysis (Phase 4)

- Residuals by state, by deprivation decile, by district size (population).
- Compare predicted vs actual for the largest misses; check for data-entry artefacts
  before calling them model error.
- Spatial clustering of residuals (state adjacency) reported descriptively.
- Segment analysis: does the model systematically under-predict small districts?

## 9. Reporting standards

- State n, folds, seed and metric spread for every number.
- Distinguish *explained variance* from *predictive skill on unseen states*.
- Always disclose: 2011 vintage, ecological design, dilapidated-proxy (not
  homelessness), 12 districts without a count-based denominator.

---

## 10. Phase 2 implementation (as built)

Code: `src/data/loaders.py`, `src/features/target.py`,
`src/features/predictors.py`, `src/training/splits.py`,
`src/evaluation/baselines.py`, `src/data/build_dataset.py`.

### 10.1 Target

D2 `Total Number of Dilapidated` on each district's `Total` row; 640 districts;
validated for range, nulls, duplicates and component additivity (Good + Livable +
Dilapidated = 100 ± 0.5). Built by `src/features/target.py:build_target`.

### 10.2 Predictors

Two normalisation strategies (D-010): within-table totals for closed compositions
(640 districts), D5 `NO_HH` for standalone household counts (628 districts).
81 features built, 4 dropped by the collinearity screen, 77 retained: 43 in
Configuration A and 34 additional in Configuration B. Every feature has a formula,
denominator, denominator source and universe in `feature_manifest.csv`.

### 10.3 Screens

- **Collinearity:** pairwise |r| > 0.98, computed on the full sample, frozen before
  modelling (D-013).
- **Indirect leakage:** |r| with the target > 0.90 → flagged for review.
- **Deny-list:** `FORBIDDEN_SOURCES` blocks every target component; a test fails if
  one appears.

### 10.4 Splits

Primary: `GroupKFold(5)` grouped by state. Reference: `KFold(5, shuffle=True)`,
5 seeds. `assert_folds_valid` enforces (a) no train/test row overlap, (b) every row
tested exactly once, (c) no state split across train and test. Identical fold
objects are reused for A, A-restricted and B.

### 10.5 Baselines

`national_mean`, `state_mean` (falls back to the global training mean for an unseen
state), `knn_headline` (6 features) and `knn_all_features` (the configuration's full
feature set). All scaling is fitted inside the training fold. Metrics: MAE, RMSE,
R², Spearman.

### 10.6 Known methodological limits of this phase

1. Screens are computed once on the full sample, not per fold; acceptable for a
   frozen feature set, not for target-driven selection inside folds.
2. Only k-NN uses features, so the A/B comparison rests on one learner until
   Phase 3.
3. Structured missingness (12 districts) is handled by exclusion from B, not by
   imputation; this makes B's sample slightly non-random with respect to state.
4. No spatial or temporal validation is possible with a single cross-section.

---

## 11. Phase 3 implementation (as built)

Code: `src/models/registry.py`, `src/training/nested_cv.py`,
`src/training/run_nested_cv.py`, `src/evaluation/metrics.py`,
`src/evaluation/compare.py`, `src/evaluation/error_analysis.py`,
`src/evaluation/finalize_phase3.py`.

### 11.1 Validation design

- Outer: 5-fold `GroupKFold` on the 35 states; inner: 4-fold `GroupKFold` on the
  outer-training states only, consumed by `GridSearchCV`.
- Held-out rows never enter tuning. Fold sizes 512/128 (640) and 502–503 / 125–126 (628).
- Random 5-fold nested CV (seed 42) is run as a labelled reference and is never used
  for selection.
- A, A-restricted and B use identical fold objects, verified in tests and by identical
  fold signatures in the metadata.

### 11.2 Models and tuning

Ten families (dummy, ridge, elastic net, k-NN, decision tree, random forest,
extra trees, gradient boosting, hist gradient boosting, SVR-RBF); 76 candidate
configurations in total, `scoring="neg_mean_absolute_error"`, refit per outer fold.
XGBoost excluded (not installed, not required). Search spaces are capped at 12
candidates per family.

### 11.3 Leakage controls

Imputation and scaling are pipeline steps fitted per fold; there is no target-driven
feature selection; district identifiers, names, codes and row order are never
predictors; the Phase 2 deny-list is enforced by test. Verified empirically: the
scaler learns training-fold means and differs from full-sample means.

### 11.4 Metrics and reporting

MAE and RMSE in percentage points, R² and Spearman. Mean and SD across outer folds
plus pooled metrics over all out-of-fold predictions. Negative R² retained.
Mean-of-fold and pooled R² are reported side by side because they answer different
questions (equal weight per state group vs per district).

### 11.5 Selection

Ranked by grouped mean MAE (RMSE tie-break), fixed before the sweep. Stability,
error structure and complexity are qualitative checks. Where two models are within a
fraction of the fold SD, they are reported as co-leading and no significance is
claimed.

### 11.6 Error and importance analysis

All analysis uses out-of-fold predictions only. Residuals are decomposed by state,
by target decile and for the extreme 5%. Permutation importance is measured on each
fold's held-out districts with that fold's hyperparameters (10 repeats), and
rank stability is reported. Built-in tree impurity importance is not used as evidence.

### 11.7 Limits of this phase

1. Five folds cannot support significance testing; paired fold differences are
   descriptive.
2. Configuration B excludes 12 districts selected by name ambiguity, so its error is
   not guaranteed to generalise to the 640-district population.
3. The models compress the target distribution and under-predict the most deprived
   districts; they cannot support targeting.
4. Importance ranks are unstable under the correlated compositional blocks.
5. One compact grid per family; a wider search would shift numbers slightly without
   changing conclusions.

### 11.8 Ratified Phase 3 specification (P3-D-13)

At Phase 3 close the project owner ratified the following specification. It governs
Phase 4 and must be quoted with its coverage caveat.

| Role | Configuration | Model | Grouped MAE | R² | Spearman |
| --- | --- | --- | ---: | ---: | ---: |
| Primary interpretable | B, 628 districts, 77 predictors | Ridge (α = 100) | 1.7490 | 0.3992 | 0.6311 |
| Secondary interpretable | B, 628 districts, 77 predictors | Elastic Net | 1.7476 | 0.3923 | 0.6444 |
| Best-performing benchmark | B, 628 districts, 77 predictors | SVR-RBF | 1.7248 | 0.3992 | 0.6258 |
| Secondary coverage | A, 640 districts, 43 predictors | SVR-RBF (best A) | 2.0293 | 0.1699 | 0.4626 |

- **Coverage.** Configuration B excludes 12 districts with unresolved D5 geographic
  matching. Its coverage is not nationally complete.
- **No significance claim.** The three B models differ by 0.024 pp in grouped MAE
  against a fold SD of ≈0.17 pp; five outer folds cannot separate them.
- **Coefficient interpretation.** Ridge coefficients are reported on the standardised
  scale (per training-fold SD). Stability is computed by refitting Ridge inside each
  outer fold with that fold's α (`src/evaluation/coefficient_stability.py`, output
  `evaluation/metrics/phase3_coefficient_stability.csv`): 54/77 predictors keep their
  sign in all five folds, median SD/|mean| = 0.55. Stable: assets (television,
  electricity, radio, scooter), household size, cooking fuel, water source and latrine
  type. Unstable and not to be interpreted individually: religion shares, primary
  education, bicycle and car ownership, male literacy.
- **Causality.** Coefficients and permutation importance are predictive associations.
  They do not identify causal effects, and correlated indicators (e.g. `tapwater_share`
  with `handpump_share`) must be interpreted jointly.
