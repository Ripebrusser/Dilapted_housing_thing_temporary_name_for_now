# District-Level Prevalence of Dilapidated Census Houses in India — Machine Learning

Predicting **how prevalent dilapidated housing is in each Indian district**, from
Census 2011 socioeconomic and housing-stock indicators, using grouped nested
cross-validation and a regularised linear model.

> **Scope in one sentence.** The project predicts the *share of a district's
> census houses classified as Dilapidated* in Census 2011. It is a structural
> condition measure of the housing stock. **It is not a homelessness model**, it
> makes no individual or household-level claim, and it makes no causal claim.

**Status:** Phases 1–4 complete and independently audited; Phase 6 (final
prediction system) implemented; Phase 5 (further modelling) deliberately not
started. Every number in this README comes from a file in `evaluation/metrics/`.

---

## 1. Research problem

Which Indian districts show high prevalence of dilapidated housing, and how much
of the between-district variation can be captured by district-level
socioeconomic and housing-stock structure? The practical question is whether a
transparent model can rank districts well enough to be useful for screening —
not whether it can explain *why* dilapidation occurs.

## 2. Objectives

1. Construct a defensible district-level target from Census 2011 housing data.
2. Build leakage-free feature configurations from district socioeconomic data.
3. Compare ten model families under validation that respects state clustering.
4. Quantify how much is lost when the model must generalise to unseen states.
5. Test whether tail-aware remedies reduce systematic under-prediction of the
   most dilapidated districts, under pre-registered acceptance gates.
6. Deliver a reproducible, inspectable prediction artefact for all eligible
   districts.

## 3. Datasets

| ID | Source | Shape | Role |
| --- | --- | --- | --- |
| D1 | `india-districts-census-2011.csv` | 640 × 118 | primary predictors (household amenities, demographics) |
| D2 | `india_census_housing-hlpca-full.csv` | 1,908 × 156 | **target source** (housing condition), district × Rural/Urban/Total |
| D3 | `hlpca-colnames.csv` | 156 cols | dictionary for D2 only |
| D4 | `Slum Population Of India.csv` | 32 × 12 | **excluded** — state level, 4 states absent |
| D5 | `census_2011.csv` (PCA) | 1,342 × 89 | household denominator + 34 housing predictors |

D6–D10 (2001 PCA, education/marital/age tables, 2016 geography) are registered
and not used for modelling; see `config.yaml`.

## 4. Target definition

```
target_deprivation_rate_pct
  = share of census houses classified "Dilapidated"
  = D2["Total Number of Dilapidated"] on the Rural/Urban == "Total" row
```

- 640 districts; range 0.2 – 17.7; mean 5.035; median 4.40; skew 1.154.
- Verified independently: reconstructed from raw files with an independent naming
  implementation, **difference 0.00e+00**; agrees with a count-based
  reconstruction from D1 ÷ D5 at r = 0.999368 (median difference 0.053 pp).
- **Rejected denominator:** D1 `Households` sums to 330,835,767, which is India's
  *census-house stock*, not its 246,740,228 households. Using it would have
  inflated the rate by ~34%. Decision record D-001.

## 5. Feature configurations

| Config | Districts | Predictors | Notes |
| --- | ---: | ---: | --- |
| A (`A_primary_640`) | 640 | 43 | D1 + derived shares; no D5 household denominator |
| B (`B_augmented_628`) | 628 | 77 | A's 43 + 34 D5 housing/asset predictors |

B loses 12 districts whose district name is duplicated across states
(Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh) and therefore
cannot be matched to a D5 household denominator. **Those districts are excluded,
never zero-filled**; their predictions are unavailable by design.

Preprocessing: median imputation (a no-op in B, which has no missing values) and
standardisation, both fitted **inside** every fold.

## 6. Model families

Ten families, 76 candidate configurations, hyperparameters chosen on inner folds
only: dummy mean, Ridge, Elastic Net, k-NN, decision tree, random forest, extra
trees, gradient boosting, histogram gradient boosting, SVR-RBF.

Ratified roles after Phase 3: **Ridge B = primary interpretable model**,
**Elastic Net B = secondary**, **SVR-RBF B = accuracy benchmark**.
SVR's MAE advantage (1.7248 vs 1.7490) is 0.024 pp, is not significant, and is
not worth the loss of interpretability.

## 7. Validation strategy

5-fold **GroupKFold on state** (35 groups), 4-fold inner GroupKFold on
outer-training states only — 20 model fits per configuration per outer fold, all
seeded at 42. No state ever appears in both training and test. Random 5-fold CV
is reported separately as a reference only: it flatters every model (SVR B:
MAE 1.140 random vs 1.725 grouped) and is **not** the generalisation estimate.

Metrics are reported as **fold means** unless labelled *pooled* (computed on all
out-of-fold rows at once). The two differ and are never mixed.

## 8. Primary results (grouped nested CV, fold means)

| Config | Model | MAE | RMSE | R² | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| A (640) | SVR-RBF | 2.0293 | 2.6144 | 0.1699 | 0.4625 |
| A (640) | dummy mean | 2.4495 | 3.1438 | −0.1157 | — |
| B (628) | **SVR-RBF** (benchmark) | **1.7248** | **2.2556** | **0.3992** | 0.6258 |
| B (628) | **Ridge α = 100** (primary) | 1.7490 | 2.2703 | 0.3992 | 0.6311 |
| B (628) | Elastic Net | 1.7476 | 2.2946 | 0.3923 | — |
| B (628) | extra trees | 1.8101 | 2.3309 | 0.3727 | — |
| B (628) | decision tree | 2.5055 | 3.2127 | −0.1840 | — |

Ridge B, pooled on all 628 out-of-fold rows: MAE 1.7496, RMSE 2.2842,
R² 0.4680, Spearman 0.6672.

Adding the 34 D5 predictors improves every family by 10–21% MAE (ridge −19.9%,
5/5 folds). **Configuration B is therefore the ratified configuration.**

## 9. Phase 4 — extreme-district investigation

The chosen model systematically under-predicts the most dilapidated districts:
the 35 districts at or above the 95th percentile (11.2%) are under-predicted by
4.51 pp on average, while the model is almost unbiased overall (+0.09 pp).

Five pre-registered remedies were tested (3 weighted Ridge arms, 2 linear
quantile regressions). **No challenger was promoted; Ridge B stands unchanged.**
They moved the tail (p95 bias −18% to −34%) but none met the registered gates:

- C1 (≥30% tail-bias reduction): best weighted arm −27.5% — **FAIL**
- C2 (≤5% overall MAE cost): every arm cost +9.4% to +66% — **FAIL**
- C4 (no single state > 60% of the gain): E4b **FAIL** (West Bengal worsened)
- C6 (precision ≥ 0.40): every arm **FAIL** (0.17–0.39)

Registered conclusion, verbatim: *"No improvement was demonstrated using the
tested tail-aware approaches. Limited predictor information is one possible
explanation, but cannot be established from these experiments alone."*

Honest reading: the tail moved, but not far enough, and at a measurable cost to
overall accuracy. A methodological deviation (the executed bootstrap resampled
states; the proposal registered districts) is documented and both are reported —
see `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md` §11.

## 10. Phase 6 — final prediction system

The ratified **Ridge B (α = 100, 77 predictors, 628 districts)** is fitted on all
628 districts and persisted, then used to score every eligible district.

| Artefact | Path |
| --- | --- |
| Fitted pipeline | `models/final/ridge_b_pipeline.joblib` |
| Metadata (versions, hashes, target definition) | `models/final/ridge_b_metadata.json` |
| Feature manifest (order matters) | `models/final/feature_manifest.csv` |
| Predictions (628 rows) | `evaluation/predictions/final_predictions_ridge_b_628.csv` |
| Ranking | `evaluation/predictions/district_ranking_ridge_b_628.csv` |
| Predictions beside observed (retrospective only) | `evaluation/predictions/final_predictions_retrospective_628.csv` |

**These fitted values are not a validation result.** They are in-sample
predictions from a model fitted on all 628 districts; the generalisation
evidence is the grouped nested CV in §8. Every file is labelled
`refit_in_sample`.

Two honest limitations are visible in the output and are **not** hidden:

1. **Two districts receive a negative fitted value** (−0.286, −0.636 pp).
   Ridge is an unconstrained linear model, so it can extrapolate just outside
   the physically admissible [0, 100] range. The values are reported as computed
   and flagged in `physical_range_flag`; they are not clipped. A floor at 0
   would not change the ranking — both districts rank last.
2. **Risk bands are relative, not absolute.** `relative_band` splits predicted
   prevalence into five equal-sized quintiles (B1 lowest … B5 highest). No
   validated external threshold for "severe" district prevalence exists in these
   sources, so these are **relative ranking bands, not risk classes**.

Ranking: predicted prevalence descending, ties broken by district key ascending,
ranks 1–628 with no duplicates. Rank 1 is a position in this list, not a
national percentile.

## 11. Repository structure

```
Admin/            scope, methodology, decisions (D-001…), experiment log
datasets/         raw (read-only) · metadata (crosswalk, dictionary) · interim · processed
src/data/         loaders, canonical naming, audits, denominator resolution
src/features/     feature builders for configurations A and B
src/models/       registry of ten families, tail-aware arms
src/training/     nested CV, configuration builders, Phase 4 runner
src/evaluation/    metrics, comparison, error analysis, bootstrap, coverage intervals
src/prediction/    final model fitting, persistence, inference, deliverables
models/final/     persisted Ridge B pipeline + metadata + manifest
evaluation/       metrics (CSV) · predictions (CSV) · plots (PNG)
reports/          dataset report · ml reports · proposals · final report
notebooks/        drivers; 11_final_predictions documents the Phase 6 run
tests/            99 tests across six files
```

## 12. Installation and environment

Verified interpreter: **Python 3.14.3**. On this machine `py` resolves to
Python 3.15, which does **not** have the scientific stack — use `py -3.14`
explicitly, as every command below does.

```bash
py -3.14 -m pip install -r requirements.txt
```

Pinned: pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1,
matplotlib 3.11.2, seaborn 0.13.2. `joblib` 1.6.0 ships with scikit-learn and is
used for persistence. Python 3.15 compatibility is **not** claimed or tested.

## 13. Reproduction

```bash
# Phase 1 — data audit (writes datasets/metadata, reports/dataset_report)
py -3.14 src/data/audit_datasets.py
py -3.14 src/data/audit_geography.py
py -3.14 src/data/target_feasibility.py
py -3.14 src/data/resolve_denominator.py

# Phase 2 — preprocessing, feature construction (A and B), baselines, folds
py -3.14 src/data/build_dataset.py

# Phase 3 — grouped nested CV (all ten families, 76 configs) + comparison + reports
py -3.14 src/training/run_nested_cv.py
py -3.14 src/evaluation/finalize_phase3.py

# Phase 4 — tail investigation
py -3.14 tests/test_phase4_isolation.py
py -3.14 src/training/run_tail_experiments.py
py -3.14 src/evaluation/tail_evaluation.py
py -3.14 src/evaluation/tail_bootstrap_registered.py
py -3.14 src/evaluation/tail_coverage_intervals.py

# Phase 6 — final model, predictions, ranking
py -3.14 src/prediction/final_model.py
py -3.14 src/prediction/build_predictions.py
py -3.14 src/prediction/inference.py

# Tests
py -3.14 tests/test_phase1_contracts.py          # 17
py -3.14 tests/test_phase2_contracts.py          # 19
py -3.14 tests/test_phase3_contracts.py          # 21
py -3.14 tests/test_phase4_isolation.py          #  9
py -3.14 tests/test_phase4_quantile_and_bootstrap.py  # 14
py -3.14 tests/test_phase6_inference.py          # 19
```

Deterministic: seed 42, `n_jobs=1`, no network access, relative paths only.
Phase 1–4 artefacts regenerate byte-identically (sha256 verified).

## 14. Outputs

- `evaluation/metrics/*.csv` — every reported metric (fold-level and pooled).
- `evaluation/predictions/final_predictions_ridge_b_628.csv` — 628 districts.
- `evaluation/predictions/district_ranking_ridge_b_628.csv` — ranked with bands.
- `models/final/` — the fitted pipeline and its provenance.
- `reports/ml_report/PHASE_3_MODELLING_REPORT.md` — the modelling report.
- `reports/final_report/FINAL_RESEARCH_REPORT.md` — the full paper.

## 15. Limitations

1. **Cross-sectional, 2011 only.** No temporal validation; the numbers describe
   the 2011 Census, not the present.
2. **Ecological design.** District averages say nothing about any household.
3. **n = 628 with 77 predictors.** R² ≈ 0.40 (fold mean) is modest; roughly 60%
   of between-district variance is unexplained.
4. **Grouped CV is hard.** Out-of-state generalisation costs ~0.58 pp MAE
   compared with random CV; the grouped figure is the honest one.
5. **12 districts unscored.** Duplicated names defeat the D5 join.
6. **Only 35 districts define the tail.** Tail conclusions rest on a small,
   state-concentrated group (24 in Assam and West Bengal).
7. **Structural condition, not homelessness.** Census "Dilapidated" is an
   enumerator classification of the dwelling.
8. **No causality.** Coefficients and permutation importance are predictive
   associations between correlated features.
9. **Tail under-prediction remains.** The ratified model under-predicts the top
   5% by ~4.5 pp; every attempted remedy failed its registered cost gate.

## 16. Academic disclaimer

The analysis uses **secondary, historical Census of India 2011 data**. It
describes 2011 conditions in the 2011 administrative geography (640 districts,
35 states/UTs); several districts have since been split and no attempt is made to
map them forward. Nothing in this repository should be read as a current
assessment of housing condition, as a policy recommendation, or as a statement
about homelessness.

## 17. Documentation map

| Document | Contents |
| --- | --- |
| `reports/final_report/FINAL_RESEARCH_REPORT.md` | full paper: method, results, discussion |
| `reports/ml_report/PHASE_3_MODELLING_REPORT.md` | modelling detail and limitations |
| `reports/ml_report/PHASE_4_TAIL_REMEDY_REPORT.md` | tail experiment + §11 corrections |
| `reports/ml_report/PHASE_6_PREDICTION_SYSTEM.md` | prediction system verification |
| `reports/ml_report/FINAL_MASTER_AUDIT.md` | independent audit and defect register |
| `reports/ml_report/VIVA_PREPARATION.md` | defence questions with worked answers |
| `reports/presentation/` | final slide deck |
| `Admin/` | scope, methodology, decisions, experiment log |
