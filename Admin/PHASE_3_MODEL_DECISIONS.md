# Phase 3 — Model Decisions

Decision records for the modelling phase. Numbering continues from the Phase 2 records (D-001 … D-013) in `Admin/MODEL_DECISIONS.md`; these are referenced as **P3-D-01 … P3-D-12**.

---

## P3-D-01 — Grouped nested CV is the primary evaluation

**Status:** adopted
**Evidence:** random-fold MAE is 0.58–0.75 pp lower than grouped for the same models (e.g. B: 1.1401 vs 1.7248).

**Decision.** Report 5-fold `GroupKFold` on the 35 states, with a 4-fold `GroupKFold` inner loop, as the generalisation estimate. Report random nested CV separately as a reference and never quote it as the headline.

**Reason.** Holding out whole states is the closest available proxy for predicting a district in a region the model has not seen. Random CV reuses states across folds and therefore measures interpolation within an observed state mix.

## P3-D-02 — Nested CV rather than a single hold-out or a tuned-then-tested split

**Status:** adopted

**Decision.** Every model is tuned inside each outer fold (`GridSearchCV`, inner 4-fold, `neg_mean_absolute_error`). Outer test rows are never seen by the search. The selected configuration may differ per fold and that variation is reported.

**Consequence.** 76 candidate configurations × 4 inner folds × 5 outer folds = 1,520 fits per configuration-scheme; 9,120 fits across the six sweeps. Reported metrics are therefore not optimistic from tuning on the evaluation data.

## P3-D-03 — Ten model families; XGBoost excluded

**Status:** adopted

**Decision.** Dummy, Ridge, Elastic Net, k-NN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, SVR-RBF. XGBoost is not used: it is not installed and adding it would introduce an installation risk without methodological benefit.

**Reason.** The set spans linear, distance, single-tree, bagged, boosted and kernel learners, which is what the A/B question and the baseline comparison require. A sixth boosted variant would add runtime, not evidence.

## P3-D-04 — All learned preprocessing inside pipelines

**Status:** adopted

**Decision.** Median imputation and `StandardScaler` are pipeline steps, fitted per fold. No target-driven feature selection is performed anywhere.

**Evidence.** `test_scaling_is_fitted_on_the_training_fold_only` fits a pipeline on one outer-training fold and asserts the learned means equal the training-fold means and differ from the full-sample means. `test_inner_tuning_folds_stay_inside_outer_training` maps inner indices back to global rows and asserts no held-out row appears.

**Note.** The imputer is retained even though the evaluated matrices are complete, so a future re-run cannot silently bypass in-fold fitting.

## P3-D-05 — Selection rule fixed in advance

**Status:** adopted

**Decision.** Rank by grouped nested-CV mean MAE; RMSE breaks ties. Stability (fold SD / coefficient of variation), error structure and complexity are qualitative checks, not post-hoc re-ranking criteria.

**Reason.** Fixing the rule before seeing results prevents selecting on a favourable metric after the fact. R² is not the selection metric, because a model can have a high R² and a worse typical error.

## P3-D-06 — Configuration B is the better-specified model, but its adoption is a coverage decision

**Status:** **ratified at Phase 3 close** — Configuration B is the primary modelling configuration; Configuration A is retained as the secondary full-coverage configuration.

**Evidence.** On identical districts and identical folds, B improves 8 of 10 families, 7 of them on all five folds, by 11–21% MAE. Largest gains are for regularised linear models (elastic net −20.4%, SVR −20.6%, ridge −19.9%). Only the decision tree degrades (+11.2%). B also fixes the fold-5 failure: A's models have R² −0.44 and −0.17 there, B has +0.08.

**Decision (provisional).** Recommend B for Phase 4 modelling, **conditional** on accepting that the model covers 628 districts. Keep A on 640 districts as the full-coverage fallback, and report both.

**Why this is not automatic.** The 12 excluded districts were removed because their names occur in two states — not at random. B is therefore estimated on a slightly selected sample, and its advantage cannot be assumed to extend to those 12 districts. More features is not the reason for preferring B; the paired evidence is.

## P3-D-07 — SVR and Ridge are co-leading; no significance claim

**Status:** superseded by P3-D-13 — Ridge is the retained primary interpretable model, Elastic Net the secondary interpretable model, and SVR-RBF the best-performing benchmark. No statistical superiority is claimed.

**Decision.** Report `svr_rbf` as the selected model on B (lowest grouped MAE 1.7248) while stating that `ridge` (1.7490) and `elastic_net` (1.7476) are within 0.025 pp — about 1.4% of MAE and far inside the fold SD (≈0.17). Recommend the regularised linear model where interpretability is required.

**Reason.** Five folds cannot separate models whose difference is an order of magnitude below fold-to-fold variation. Claiming otherwise would be a metric artefact, not a finding.

## P3-D-08 — Pooled and fold-averaged R² are both reported

**Status:** adopted

**Decision.** Report mean-of-fold R² and pooled out-of-fold R² side by side (e.g. A 640: 0.1699 vs 0.2895), and retain negative R².

**Reason.** They answer different questions. The fold mean weights each state group equally and exposes folds where the model fails (A 640 fold 5: R² −0.4447); the pooled value weights districts. Reporting only one would hide either the failure or the overall fit.

## P3-D-09 — Permutation importance on held-out folds only

**Status:** adopted

**Decision.** Importance is the MAE increase when a feature is shuffled on held-out districts, with the estimator refitted per fold using that fold's chosen hyperparameters (10 repeats, seed 42). Built-in tree impurity importance is not used as evidence.

**Reason.** Impurity importance is computed on training data, is biased toward high-cardinality and correlated features, and cannot support a generalisation claim. Permutation on held-out folds at least measures out-of-sample reliance, though it remains associational.

## P3-D-10 — District predictions are not risk rankings

**Status:** adopted

**Decision.** Do not publish per-district predictions as a validated ranking. Error analysis shows the top 5% of districts are under-predicted 97–100% of the time and the model compresses the distribution by ~2.4 pp at the bottom and ~4.9 pp at the top.

**Reason.** A ranking that systematically misses the worst-off districts is worse than no ranking for targeting purposes.

## P3-D-11 — The Phase 2 screens are reused, not recomputed per fold

**Status:** adopted with disclosure

**Decision.** Keep the frozen Phase 2 collinearity screen (|r| > 0.98, 4 features dropped) and the leakage deny-list. Do not add any target-driven selection. Disclose that the screen was computed once on the full sample.

**Reason.** The screen is unsupervised and was frozen before modelling, so it cannot leak the target. Recomputing it per fold would change the feature set between folds and make the A/B comparison incoherent. A future Phase 4 model that selects features by target association must do so inside folds.

## P3-D-12 — Runtime and resumability

**Status:** adopted

**Decision.** The sweep runner merges results on `(configuration, model)`, so an interrupted run can be completed without repeating finished work. Seeds, package versions, fold signatures and per-fold hyperparameters are written to `_phase3_{scheme}_meta.json`.

**Reason.** A background sweep was in fact lost when the execution environment restarted. Only the missing random-CV stage was re-run; the completed grouped stage was preserved and reused. Reproducibility was then demonstrated by hash comparison rather than asserted.

---

## P3-D-13 — Final Phase 3 model ratification

**Status:** **ratified by the project owner at Phase 3 close** (supersedes the provisional elements of P3-D-06 and P3-D-07)

**Decision 1 — Configuration.** Configuration B is the primary modelling configuration: 628 districts, 77 approved predictors (D1 socioeconomic and demographic features plus valid D5-normalised household features). Configuration A is retained as the secondary full-coverage configuration: 640 districts, 43 predictors.

Configuration B excludes 12 districts because of unresolved D5 geographic matching. **Its coverage must not be described as nationally complete.**

**Decision 2 — Models.**

| Role | Configuration | Model | Grouped MAE | RMSE | R² | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Primary interpretable | B (628 d, 77 f) | **Ridge** | 1.7490 | 2.2703 | 0.3992 | 0.6311 |
| Secondary interpretable | B (628 d, 77 f) | Elastic Net | 1.7476 | 2.2946 | 0.3923 | 0.6444 |
| Best-performing benchmark | B (628 d, 77 f) | SVR-RBF | 1.7248 | 2.2556 | 0.3992 | 0.6258 |

**No statistical superiority is claimed between these three models.** The spread in grouped MAE is 0.024 pp (1.4% of the metric) against a fold SD of ≈0.17 pp, and five outer folds cannot support a significance test. The choice is therefore on interpretability and parsimony, not on accuracy.

**Decision 3 — Interpretation.** Ridge coefficients and permutation importance may be used to discuss **predictive associations, not causal effects**. Coefficient stability across outer folds must be checked before any strong statement about an individual predictor (see P3-D-14), and coefficients of correlated features must not be read independently without qualification.

**Consequences.**
- The Phase 4 headline model is Ridge on B; SVR-RBF remains the accuracy benchmark that any new model must match.
- Every headline number inherits B's 628-district coverage caveat.
- No Phase 1 or Phase 2 data, target definition or completed Phase 3 result was altered by this ratification.

---

## P3-D-14 — Coefficient stability evidence for the interpretation rules

**Status:** adopted (evidence for P3-D-13, Decision 3)

**Method.** Ridge was refitted inside each outer fold on Configuration B with that fold's chosen α (100 in all five folds), and coefficients were read on the standardised scale (one unit = one training-fold SD). Output: `evaluation/metrics/phase3_coefficient_stability.csv`.

**Result.**
- **54 of 77 predictors (70%) keep the same sign in all five folds.**
- Median SD / |mean coefficient| across folds is **0.55**, i.e. typical coefficient uncertainty of about half its magnitude.
- Stable, tightly estimated coefficients (mean ± SD): `tv_share` −0.473 ± 0.039, `hh_size_2_share` −0.449 ± 0.026, `electricity_share` −0.394 ± 0.031, `radio_share` −0.360 ± 0.075, `scooter_share` −0.354 ± 0.057, `hh_size_3_share` +0.313 ± 0.045, `marginal_worker_share` −0.307 ± 0.036, `other_cooking_fuel_share` +0.301 ± 0.074, `handpump_share` +0.299 ± 0.095, `pit_latrine_share` +0.281 ± 0.122.
- **Unstable coefficients that must not be interpreted individually:** `muslim_share` +0.299 ± 0.192 (negative in 1 of 5 folds), `sikh_share` +0.143 ± 0.103, `primary_share` −0.143 ± 0.130 (positive in 1 of 5), `bicycle_share` +0.125 ± 0.103, `literacy_rate_male` −0.079 ± 0.059, `car_share` +0.075 ± 0.079.

**Consequence.** Discussion may rely on the sign and approximate magnitude of the stable block — assets, household size, cooking fuel, water source and latrine type — and must treat religion shares, primary education, vehicle ownership and male literacy as unstable. Complementary indicators (`tapwater_share` negative with `handpump_share` positive; `water_near_premises_share` negative with `water_away_share` positive) are entangled by construction and must be interpreted as a group.

---

## Open decisions for Phase 4

1. **Extreme-district handling:** accept the compression, or address it explicitly (e.g. quantile loss, two-stage model). Any such change must be evaluated on the same grouped folds.
2. **D7–D9:** decide whether the unused education, marital-status and age extracts may be admitted, and require them to beat these baselines.
3. **Reporting of district-level output:** decide whether any district-level table may be published, given P3-D-10.
4. **B rate overshoot:** decide whether the five B features exceeding 1.0 should be clipped, left as-is (current), or replaced by a D1-internal household total.
