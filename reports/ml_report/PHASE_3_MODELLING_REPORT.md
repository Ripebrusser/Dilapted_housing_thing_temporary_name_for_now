# Phase 3 — Machine Learning Modelling Report

**Project:** Predicting District-Level Prevalence of Dilapidated Census Houses in India Using Machine Learning
**Phase:** 3 (modelling, nested cross-validation, model selection)
**Status:** complete — **ratified at Phase 3 close (P3-D-13)**; Phase 4 not started
**Ratified specification:** Configuration B (628 districts, 77 predictors) primary; Ridge primary interpretable model; Elastic Net secondary; SVR-RBF accuracy benchmark; Configuration A secondary full-coverage.
**Target (unchanged from the approved Phase 2 definition):** `target_deprivation_rate_pct`

---

## Disclosure summary

1. **All source data are from the 2011 Census of India.** Every result describes 2011 conditions.
2. **This is district-level ecological prediction**, not prediction of individual or household homelessness. The target is the Census-computed share of dilapidated census houses.
3. **Results do not establish causality.** Feature importance is predictive association under correlation.
4. **Generalisation to present-day India has not been demonstrated.** The model has been validated on held-out 2011 states only.
5. **Grouped nested cross-validation is the primary evaluation.** Random-fold results are reported as a reference and must not be quoted as the generalisation estimate.

## Reproduce

```bash
py src/training/run_nested_cv.py --scheme grouped     # primary, ~8 min
py src/training/run_nested_cv.py --scheme random      # reference, ~16 min
py src/evaluation/finalize_phase3.py                  # tables, error analysis, plots (~75 s)
py tests/run_notebooks.py                             # executes notebooks 06-10 in a Jupyter kernel
py tests/test_phase1_contracts.py
py tests/test_phase2_contracts.py
py tests/test_phase3_contracts.py
```

The runner is resumable: results merge on `(configuration, model)`, so an interrupted sweep can be completed without repeating finished work.

---

## 1. Objective and scope

Estimate the district-level prevalence of dilapidated census houses from socioeconomic, demographic and household-amenity composition variables, and determine whether household-amenity features available for 628 districts improve out-of-state generalisation over a 640-district socioeconomic-only specification.

In scope: ten regression families, state-grouped nested CV, a matched 628-district A-versus-B comparison, residual and importance analysis.
Out of scope by instruction: deployment, website, dashboard, API, classification conversion, alternative targets, causal claims.

## 2. Target

`target_deprivation_rate_pct` — D2 `Total Number of Dilapidated` for each district's `Total` row.

- 640 districts, continuous, percentage points, no nulls, 124 distinct values.
- Median 4.40, mean 5.035, sd 3.12, range 0.2–17.7, skew 1.15.
- Verified additive: Good + Livable + Dilapidated = 100 ± 0.5 for every district.
- No target component, D2 housing-condition column or mathematical equivalent enters any feature matrix (enforced by `FORBIDDEN_SOURCES` and tested).

## 3. Dataset and feature configurations

| Configuration | Districts | Predictors | Content |
| --- | ---: | ---: | --- |
| A (primary) | 640 | 43 | D1 socioeconomic and demographic shares only |
| A (restricted) | 628 | 43 | Same features, restricted to districts with a D5 denominator |
| B (augmented) | 628 | 77 | A plus 34 D5-`NO_HH`-normalised household indicators |

Features: religion, education attainment, age structure, income bands and household size as closed compositions (one reference category each, all additive at 1.0000); literacy, caste, work participation and tenure shares; amenities, assets, water source, latrine type and cooking fuel normalised by D5 `NO_HH`.

**Ratified allocation (P3-D-13).** Configuration B is the primary modelling configuration and Configuration A the secondary full-coverage configuration. **B's 628-district coverage is not nationally complete and must not be described as such.**

The 12 districts without a reliable D5 match are preserved as missing and excluded from B only. Five B rates exceed 1.0 by at most 6.2% because D1 asset counts and D5 household counts are different per-district universes; values are not clipped.

## 4. Cross-validation methodology

**Primary — state-grouped nested CV.**
- Outer: 5-fold `GroupKFold` on the 35 states.
- Inner: 4-fold `GroupKFold` on the outer-training states only, used exclusively by `GridSearchCV`.
- A state never appears in both training and test. For every fold, training states + held-out states = 35, with no overlap (asserted in code and in tests).
- Fold sizes: 512/128 for 640 districts; 502–503 / 125–126 for 628.

**Reference — random nested CV.** Same ten families, same grids, same budgets, shuffled 5-fold outer and inner splits (seed 42). Reported separately; never used for selection.

**Outer folds (640-district configuration)**

| Fold | Districts held out | States held out |
| ---: | ---: | --- |
| 1 | 128 | Arunachal Pradesh, Daman and Diu, Jammu and Kashmir, Pondicherry, Uttar Pradesh, Uttarakhand |
| 2 | 128 | Andaman and Nicobar Islands, Assam, Haryana, Kerala, Madhya Pradesh, Manipur, Tripura |
| 3 | 128 | Andhra Pradesh, Bihar, Karnataka, Lakshadweep, Mizoram, NCT of Delhi, West Bengal |
| 4 | 128 | Chhattisgarh, Dadra and Nagar Haveli, Goa, Gujarat, Himachal Pradesh, Maharashtra, Orissa, Sikkim |
| 5 | 128 | Chandigarh, Jharkhand, Meghalaya, Nagaland, Punjab, Rajasthan, Tamil Nadu |

These folds are byte-identical to the Phase 2 fold file, and A/B use the same fold objects.

**Leakage controls.**
- Imputation (median) and scaling (`StandardScaler`) live inside a `Pipeline`; both are fitted on training rows only. Verified by test: the learned scaler means equal the training-fold means and differ from full-sample means.
- No feature selection uses the target. The Phase 2 collinearity screen is a frozen, unsupervised screen; no target-driven selection is performed.
- Hyperparameters are chosen on inner folds only; outer test rows never enter `GridSearchCV`.
- District keys, names, codes, state labels and row order are never predictors.

## 5. Models evaluated

| Family | Candidates | Scaling | Complexity rank |
| --- | ---: | :---: | ---: |
| Dummy (mean) | 1 | no | 1 |
| Ridge | 5 | yes | 2 |
| Elastic Net | 12 | yes | 3 |
| k-NN | 8 | yes | 3 |
| Decision Tree | 12 | no | 4 |
| Random Forest (200 trees) | 8 | no | 6 |
| Extra Trees (200 trees) | 8 | no | 6 |
| Gradient Boosting (150/300 trees, subsample 0.8) | 8 | no | 7 |
| HistGradientBoosting (no early stopping) | 8 | no | 7 |
| SVR (RBF) | 6 | yes | 5 |

**XGBoost was not used.** It is not installed and is not required; adding it would have introduced an installation risk for no methodological gain.

## 6. Hyperparameter tuning

- Grid search, `scoring="neg_mean_absolute_error"`, inner 4-fold.
- 76 candidate configurations across the ten families are evaluated inside each outer fold → 304 inner fits per outer fold per configuration-scheme, **1,520 fits per configuration-scheme**, 9,120 across the six sweeps.
- The chosen configuration can differ by fold; that variation is reported rather than hidden.

**Selected models, per-fold choices**

`A_primary_640 / svr_rbf` (6 candidates): C = 1, 10, 1, 10, 1 across folds 1–5; epsilon 0.5 except fold 5 (0.1); gamma = scale throughout.

`A_restricted_628 / extra_trees` (8 candidates): max_depth 10 / None / 10 / 10 / None; max_features sqrt / sqrt / sqrt / 0.5 / sqrt; min_samples_leaf 1 / 3 / 1 / 3 / 1.

`B_augmented_628 / svr_rbf`: C = 10 in all five folds; epsilon 0.5 / 0.5 / 0.1 / 0.5 / 0.1; gamma = scale.

## 7. Grouped nested-CV results (primary)

Mean of the five fold metrics, with fold SD. `pooled_R²` is computed on all 640 (or 628) out-of-fold predictions pooled, which is why it differs from the mean of fold R².

### A — primary, 640 districts, 43 predictors

| Model | MAE | SD | RMSE | R² | Spearman | pooled R² | random MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **svr_rbf** | **2.0293** | 0.1664 | 2.6144 | 0.1699 | 0.4626 | 0.2895 | 1.3257 |
| extra_trees | 2.0919 | 0.1399 | 2.6705 | 0.1631 | 0.5032 | 0.2566 | 1.3587 |
| random_forest | 2.1128 | 0.1764 | 2.7039 | 0.1387 | 0.4722 | 0.2353 | 1.5105 |
| hist_gradient_boosting | 2.1444 | 0.2585 | 2.7271 | 0.0905 | 0.4627 | 0.2236 | 1.4152 |
| knn | 2.1636 | 0.3826 | 2.8073 | 0.0123 | 0.4470 | 0.1594 | 1.3350 |
| ridge | 2.1877 | 0.2400 | 2.7957 | 0.0715 | 0.4484 | 0.1847 | 1.6728 |
| elastic_net | 2.2541 | 0.3261 | 2.9061 | −0.0424 | 0.4368 | 0.1202 | 1.6646 |
| gradient_boosting | 2.2586 | 0.2715 | 2.9154 | −0.0124 | 0.4445 | 0.1098 | 1.4475 |
| dummy_mean | 2.4402 | 0.1656 | 3.1486 | −0.1371 | — | −0.0355 | 2.4122 |
| decision_tree | 2.4960 | 0.3460 | 3.3277 | −0.3360 | 0.2752 | −0.1500 | 1.9089 |

### A — restricted to the 628 districts of Configuration B

| Model | MAE | SD | RMSE | R² | Spearman | pooled R² | random MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **extra_trees** | **2.0888** | 0.1084 | 2.6540 | 0.1837 | 0.4966 | 0.2721 | 1.3698 |
| random_forest | 2.0991 | 0.1244 | 2.6772 | 0.1724 | 0.4920 | 0.2579 | 1.4924 |
| gradient_boosting | 2.1426 | 0.1062 | 2.7205 | 0.1277 | 0.4281 | 0.2378 | 1.4419 |
| knn | 2.1460 | 0.2898 | 2.7856 | 0.0317 | 0.4581 | 0.1897 | 1.3480 |
| hist_gradient_boosting | 2.1506 | 0.1607 | 2.7037 | 0.1157 | 0.4245 | 0.2485 | 1.4715 |
| svr_rbf | 2.1714 | 0.2357 | 2.8104 | 0.0377 | 0.3989 | 0.1843 | 1.3390 |
| ridge | 2.1845 | 0.1701 | 2.7949 | 0.0897 | 0.4498 | 0.1908 | 1.6859 |
| elastic_net | 2.1963 | 0.2003 | 2.8717 | 0.0656 | 0.4769 | 0.1406 | 1.6828 |
| decision_tree | 2.2528 | 0.2673 | 3.0075 | −0.0676 | 0.2639 | 0.0595 | 1.8717 |
| dummy_mean | 2.4495 | 0.2345 | 3.1438 | −0.1157 | — | −0.0289 | 2.4269 |

### B — augmented, 628 districts, 77 predictors

| Model | MAE | SD | RMSE | R² | Spearman | pooled R² | random MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **svr_rbf** | **1.7248** | 0.1506 | 2.2556 | 0.3992 | 0.6258 | 0.4755 | 1.1401 |
| elastic_net | 1.7476 | 0.1723 | 2.2946 | 0.3923 | 0.6444 | 0.4562 | 1.4468 |
| ridge | 1.7490 | 0.1775 | 2.2703 | 0.3992 | 0.6311 | 0.4680 | 1.4363 |
| extra_trees | 1.8101 | 0.1751 | 2.3309 | 0.3727 | 0.6289 | 0.4378 | 1.2329 |
| hist_gradient_boosting | 1.8413 | 0.2769 | 2.3754 | 0.3437 | 0.5856 | 0.4125 | 1.2710 |
| knn | 1.8633 | 0.1620 | 2.4175 | 0.3135 | 0.5695 | 0.3978 | 1.2632 |
| gradient_boosting | 1.8677 | 0.1976 | 2.4338 | 0.3251 | 0.5886 | 0.3833 | 1.2614 |
| random_forest | 1.8708 | 0.1343 | 2.4251 | 0.3273 | 0.6268 | 0.3905 | 1.3456 |
| dummy_mean | 2.4495 | 0.2345 | 3.1438 | −0.1157 | — | −0.0289 | 2.4269 |
| decision_tree | 2.5055 | 0.5869 | 3.2127 | −0.1840 | 0.3752 | −0.1143 | 1.7855 |

**Fold detail, selected models (grouped)**

| Config / model | Fold | n_test | States | MAE | RMSE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A640 / svr_rbf | 1 | 128 | 6 | 2.1399 | 2.6132 | 0.2379 | 0.5410 |
| | 2 | 128 | 7 | 2.0903 | 2.9401 | 0.2605 | 0.4584 |
| | 3 | 128 | 7 | 2.0235 | 2.7935 | 0.3259 | 0.7402 |
| | 4 | 128 | 8 | 1.7450 | 2.1499 | 0.4699 | 0.6253 |
| | 5 | 128 | 7 | 2.1476 | 2.5752 | **−0.4447** | **−0.0522** |
| A628 / extra_trees | 1 | 126 | 6 | 2.1949 | 2.5779 | 0.1946 | 0.5380 |
| | 2 | 126 | 7 | 2.1559 | 3.1154 | 0.1987 | 0.4660 |
| | 3 | 125 | 7 | 2.1261 | 2.8804 | 0.3310 | 0.6791 |
| | 4 | 126 | 8 | 2.0456 | 2.4324 | 0.3682 | 0.5750 |
| | 5 | 125 | 7 | 1.9216 | 2.2640 | **−0.1743** | 0.2251 |
| B628 / svr_rbf | 1 | 126 | 6 | 1.9184 | 2.3362 | 0.3385 | 0.5869 |
| | 2 | 126 | 7 | 1.7757 | 2.5907 | 0.4459 | 0.6002 |
| | 3 | 125 | 7 | 1.7857 | 2.3771 | 0.5444 | 0.7608 |
| | 4 | 126 | 8 | 1.5872 | 1.9738 | 0.5840 | 0.7326 |
| | 5 | 125 | 7 | 1.5574 | 2.0004 | 0.0833 | 0.4487 |

Fold 5 is the hard case for A: it mixes Punjab and Rajasthan with Chandigarh, Nagaland and Meghalaya, and both A models lose their ranking there (ρ ≈ −0.05 and R² < 0). Configuration B keeps a positive R² on that same fold. This asymmetry is evidence for the added features, not a numerical accident: the same districts and the same fold are used.

## 8. Random nested-CV results (reference only)

| Configuration | Best model | MAE | SD | RMSE | R² | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A 640 | svr_rbf | 1.3257 | 0.0983 | 1.7753 | 0.6661 | 0.7834 |
| A 628 | svr_rbf | 1.3390 | 0.0522 | 1.8562 | 0.6396 | 0.7767 |
| B 628 | svr_rbf | 1.1401 | 0.0321 | 1.6032 | 0.7316 | 0.8319 |

The grouped-minus-random MAE gap is 0.70 pp (A 640), 0.75 pp (A 628) and 0.58 pp (B). Roughly a third of the apparent random-CV accuracy is state context: when the same states appear in training, a district's regional level is available almost for free. This is why grouped CV is the reported estimate.

## 9. Configuration A versus B

Paired by fold on the identical 628 districts with identical folds. Delta = B − A; negative means B had the lower fold MAE.

| Model | A MAE | B MAE | Δ | SD | B better (folds) | Δ% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| elastic_net | 2.1963 | 1.7476 | −0.4487 | 0.1240 | 5/5 | −20.4 |
| svr_rbf | 2.1714 | 1.7248 | −0.4465 | 0.1661 | 5/5 | −20.6 |
| ridge | 2.1845 | 1.7490 | −0.4355 | 0.1044 | 5/5 | −19.9 |
| hist_gradient_boosting | 2.1506 | 1.8413 | −0.3093 | 0.2692 | 4/5 | −14.4 |
| knn | 2.1460 | 1.8633 | −0.2827 | 0.2559 | 5/5 | −13.2 |
| extra_trees | 2.0888 | 1.8101 | −0.2787 | 0.0904 | 5/5 | −13.3 |
| gradient_boosting | 2.1426 | 1.8677 | −0.2749 | 0.1681 | 5/5 | −12.8 |
| random_forest | 2.0991 | 1.8708 | −0.2283 | 0.0632 | 5/5 | −10.9 |
| dummy_mean | 2.4495 | 2.4495 | 0.0000 | 0.0000 | 0/5 | 0.0 |
| decision_tree | 2.2528 | 2.5055 | +0.2527 | 0.4225 | 2/5 | +11.2 |

**Reading.** Eight of ten families improve, seven of them on all five folds, and the improvement is largest for the regularised linear models (−20%) — which is the signature of features that carry information the socioeconomic shares do not, rather than of extra dimensionality. The decision tree degrades, as expected when 34 weak, correlated splits are added to an unstable learner. The dummy is unchanged by construction (it ignores features), which is a useful control that the comparison itself is not biasing the result.

**This does not by itself adopt Configuration B.** B covers 628 districts, not 640, and the excluded 12 are exactly the districts with ambiguous names. B is therefore the better *specified* model on the available evidence but the weaker *coverage* choice. That trade-off is a project decision (see `Admin/PHASE_3_MODEL_DECISIONS.md`, P3-D-06).

## 10. Selected model and justification

| Configuration | Selected | Grouped MAE | SD | R² | Spearman | vs dummy | vs Phase 2 kNN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A 640 | svr_rbf | 2.0293 | 0.1664 | 0.1699 | 0.4626 | −16.8% | −4.4% |
| A 628 | extra_trees | 2.0888 | 0.1084 | 0.1837 | 0.4966 | −14.7% | −1.0% |
| **B 628** | **svr_rbf** | **1.7248** | 0.1506 | 0.3992 | 0.6258 | **−29.6%** | **−7.0%** |

Selection rule (fixed in advance, P3-D-05): lowest grouped nested-CV mean MAE, RMSE as tie-break, with stability, error structure and complexity used as qualitative checks.

**Ratified Phase 3 specification (P3-D-13).** At Phase 3 close the project owner adopted a specification that is deliberately not identical to the metric ranking above:

| Role | Configuration | Model | Grouped MAE | RMSE | R² | Spearman |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Primary interpretable | B (628 d, 77 f) | **Ridge** | 1.7490 | 2.2703 | 0.3992 | 0.6311 |
| Secondary interpretable | B (628 d, 77 f) | Elastic Net | 1.7476 | 2.2946 | 0.3923 | 0.6444 |
| Best-performing benchmark | B (628 d, 77 f) | SVR-RBF | 1.7248 | 2.2556 | 0.3992 | 0.6258 |

Configuration A remains the secondary full-coverage configuration (its best model is SVR-RBF at 2.0293). Ridge was chosen as the primary model because its accuracy is indistinguishable from the benchmark while its coefficients are directly interpretable. The spread among the three B models is 0.024 pp against a fold SD of ≈0.17 pp, so **no statistical superiority is claimed**.

**Why `svr_rbf` on B.** It has the lowest grouped MAE (1.7248) and the joint-highest grouped R² (0.3992) of all 30 model-configuration pairs, the highest grouped Spearman (0.6258) among B models, and a fold-to-fold coefficient of variation of 0.087 — comparable to the alternatives. Its hyperparameters are stable across folds (C = 10 in all five).

**Honest caveat on that choice.** `ridge` (1.7490) and `elastic_net` (1.7476) are within 0.025 pp — about 1.4% of the MAE and far inside the fold noise (SD ≈ 0.17). Elastic Net even has the highest rank correlation (0.6444). The SVR's advantage is therefore **not established**; on parsimony and interpretability the regularised linear model is the better default. Both are reported as co-leading; the report does not claim SVR is significantly better. Following ratification, **Ridge is the retained primary model**, Elastic Net the secondary interpretable model, and SVR-RBF the best-performing benchmark (P3-D-13).

**No model beats the baselines on A in a decisive way.** Under grouped CV, the best A model improves on the dummy by 16.8% but only 4.4% on the Phase 2 k-NN all-feature baseline (2.1236 → 2.0293). For A, the socioeconomic shares alone buy limited out-of-state signal, and that is reported as a finding rather than hidden.

## 11. Residual and error analysis

Out-of-fold predictions only. `residual = actual − predicted`; positive = underprediction.

| Configuration / model | MAE | RMSE | Bias | Over-predicted | Under-predicted | p95 |error| | Max |error| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 640 / svr_rbf | 2.0293 | 2.6279 | +0.327 | 297 | 343 | 5.201 | 9.602 |
| A 628 / extra_trees | 2.0890 | 2.6720 | +0.151 | 327 | 301 | 5.127 | 11.022 |
| B 628 / svr_rbf | 1.7250 | 2.2682 | +0.141 | 322 | 306 | 4.399 | 11.591 |

**Systematic distribution compression.** Bias by target decile (A 640):

| Band | n | Mean actual | MAE | Bias |
| --- | ---: | ---: | ---: | ---: |
| 0.2–1.6 | 66 | 1.18 | 2.443 | −2.443 |
| 1.6–2.3 | 65 | 2.01 | 2.072 | −2.041 |
| 2.3–3.0 | 64 | 2.74 | 1.296 | −1.228 |
| 3.0–3.8 | 68 | 3.46 | 1.218 | −0.759 |
| 3.8–4.4 | 59 | 4.15 | 1.170 | −0.247 |
| 4.4–5.1 | 63 | 4.81 | 1.267 | +0.200 |
| 5.1–5.9 | 66 | 5.58 | 1.523 | +1.041 |
| 5.9–7.4 | 62 | 6.69 | 1.693 | +1.508 |
| 7.4–9.1 | 63 | 8.23 | 2.699 | +2.550 |
| 9.1–17.7 | 64 | 11.77 | 4.885 | +4.885 |

The model over-predicts the least deprived districts by ~2.4 pp and under-predicts the most deprived by ~4.9 pp. It behaves as a regression-to-the-mean predictor, which is the expected failure of squared/absolute-loss learners on a right-skewed target with 640 observations.

**Extreme districts are the hard case, consistently.**

| Configuration | Group | n | Mean actual | MAE | Bias | Share under-predicted |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A 640 | top 5% | 35 | 13.23 | 5.863 | +5.863 | 1.000 |
| A 640 | remainder | 605 | 4.56 | 1.807 | +0.007 | 0.509 |
| A 628 | top 5% | 35 | 13.23 | 6.532 | +6.532 | 1.000 |
| A 628 | remainder | 593 | 4.58 | 1.827 | −0.226 | 0.449 |
| B 628 | top 5% | 35 | 13.23 | 4.378 | +4.298 | 0.971 |
| B 628 | remainder | 593 | 4.58 | 1.568 | −0.104 | 0.459 |

Configuration B halves the gap between extreme and typical error (4.38 vs 1.57 instead of 5.86 vs 1.81) but still fails to reach the extremes.

**Largest errors** (A 640): Dhubri, Assam (17.0 → 7.40), Barpeta, Assam (17.7 → 8.30), Purba Medinipur, West Bengal (17.2 → 8.04), Murshidabad, West Bengal (16.0 → 7.04), Darrang, Assam (17.2 → 8.53), Jalpaiguri, West Bengal (14.5 → 5.92), Kurung Kumey, Arunachal Pradesh (12.3 → 3.76), Hailakandi, Assam (16.1 → 8.21).

**By state** (A 640, worst MAE): Tripura 5.22 (4 districts, bias −5.22 → over-predicted), West Bengal 4.88 (19, +4.88), Assam 3.60 (27, +3.46), Punjab 3.39 (20, +3.39), Tamil Nadu 2.79 (32, −2.79). Under B: Assam 3.43, Punjab 2.71, West Bengal 2.70, Meghalaya 2.67.

Errors concentrate in the eastern high-deprivation belt (Assam, West Bengal) and in Punjab — precisely the districts whose actual deprivation is far from the national relationship. This is a substantive finding, not only a modelling artefact: the same socioeconomic composition that predicts 8% elsewhere corresponds to 17% in these districts.

## 12. Feature importance and stability

Permutation importance, measured on each fold's held-out districts with that fold's hyperparameters, refitted per fold (10 repeats, seed 42).

### A 640 / svr_rbf

| Rank | Feature | Mean | SD | Folds positive | Rank SD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | hh_size_3_share | 0.0657 | 0.1499 | 2/5 | 21.8 |
| 2 | owned_share_of_reported_tenure | 0.0604 | 0.0351 | 5/5 | 5.4 |
| 3 | hh_size_2_share | 0.0491 | 0.0613 | 4/5 | 15.0 |
| 4 | secondary_share | 0.0443 | 0.0260 | 5/5 | 4.0 |
| 5 | income_90k_150k_share | 0.0429 | 0.0642 | 4/5 | 15.3 |
| 6 | worker_share | 0.0427 | 0.0292 | 4/5 | 5.3 |

### B 628 / svr_rbf

| Rank | Feature | Mean | SD | Folds positive | Rank SD |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | tv_share | 0.0836 | 0.0837 | 5/5 | 4.1 |
| 2 | hh_size_2_share | 0.0609 | 0.0485 | 4/5 | 26.6 |
| 3 | mobile_only_share | 0.0527 | 0.0320 | 5/5 | 8.6 |
| 4 | tapwater_share | 0.0434 | 0.0158 | 5/5 | 6.9 |
| 5 | telephone_share | 0.0385 | 0.0282 | 5/5 | 12.4 |
| 6 | scooter_share | 0.0384 | 0.0485 | 4/5 | 31.0 |

**Stability is limited and must be stated.** Only 6 of 43 features (A 640), 6 of 43 (A 628) and 10 of 77 (B) have positive importance in all five folds. `hh_size_3_share` ranks first on average but is positive in only 2 of 5 folds with SD (0.150) more than twice its mean — its top rank is not a reliable finding. The most stable signals are tenure (`owned_share`, 5/5), education (`secondary_share`, 5/5), household size 1 and 2, and in B the asset block (`tv_share`, `mobile_only_share`, `tapwater_share`, `telephone_share`, all 5/5 with modest rank SD).

**Correlated predictors.** The compositional blocks are internally correlated (religion, education, age, income, household size), and the amenity block shares one denominator. Permutation importance is biased toward correlated groups: shuffling one member leaves the information available through its partners, which both deflates individual scores and destabilises ranks (visible as large rank SD). The Phase 2 screen removed four near-duplicate features (|r| > 0.98) but substantial correlation remains.

**Predictive association, not causation.** Nothing here identifies a causal effect. `tv_share` predicting deprivation more strongly than income does not mean television ownership causes deprivation.

### 12.1 Ridge coefficient stability on Configuration B

Because Ridge is the retained primary model, its coefficients were refitted inside each outer fold with that fold's α (**100 in all five folds**) and read on the standardised scale (one unit = one training-fold SD). Output: `evaluation/metrics/phase3_coefficient_stability.csv`.

- **54 of 77 predictors (70%) keep the same sign in all five folds.** Median SD / |mean| is **0.55** — typical coefficient uncertainty of about half the magnitude.
- **Stable and interpretable** (mean ± SD): `tv_share` −0.473 ± 0.039, `hh_size_2_share` −0.449 ± 0.026, `electricity_share` −0.394 ± 0.031, `radio_share` −0.360 ± 0.075, `scooter_share` −0.354 ± 0.057, `hh_size_3_share` +0.313 ± 0.045, `marginal_worker_share` −0.307 ± 0.036, `other_cooking_fuel_share` +0.301 ± 0.074, `handpump_share` +0.299 ± 0.095, `water_near_premises_share` −0.288 ± 0.080, `pit_latrine_share` +0.281 ± 0.122, `tapwater_share` −0.236 ± 0.057, `lpg_share` −0.213 ± 0.046.
- **Unstable — must not be interpreted individually:** `muslim_share` +0.299 ± 0.192 (negative in one fold), `sikh_share` +0.143 ± 0.103, `primary_share` −0.143 ± 0.130, `bicycle_share` +0.125 ± 0.103, `literacy_rate_male` −0.079 ± 0.059, `car_share` +0.075 ± 0.079.
- **Direction is coherent with deprivation** — fewer televisions, radios, scooters, electricity connections and clean cooking fuel, and less secure water, associate with more dilapidated housing. `handpump_share` is positive because it marks the absence of tapped water, and `pit_latrine_share` positive because it marks poor sanitation. These are conditional associations among correlated indicators, not causal effects.

## 13. Baseline comparisons

| Reference | Grouped MAE | This phase |
| --- | ---: | --- |
| Dummy (mean), A 640 | 2.4402 | 2.0293 (−16.8%) |
| Phase 2 k-NN all features, A 640 | 2.1236 | 2.0293 (−4.4%) |
| Dummy (mean), A 628 | 2.4495 | 2.0888 (−14.7%) |
| Phase 2 k-NN all features, A 628 | 2.1107 | 2.0888 (−1.0%) |
| Phase 2 k-NN all features, B 628 | 1.8547 | 1.7248 (−7.0%) |

Under random CV the Phase 2 state-mean baseline reached 1.556; the grouped dummy is 2.44 because no held-out state's mean is available. Every Phase 3 model that beats the dummy under grouped CV has learned something that transfers across states — but the margin over Phase 2's k-NN on A is small.

## 14. Limitations

1. **2011 vintage, single cross-section.** No temporal validation is possible; nothing here supports present-day inference.
2. **Ecological design.** District aggregates cannot identify household-level relationships.
3. **Small n.** 640 observations with 43–77 correlated predictors; all metrics carry substantial sampling uncertainty. No confidence intervals are claimed; five folds cannot support significance testing.
4. **B's coverage.** B excludes 12 districts chosen by name ambiguity, not at random. If those districts differ systematically, B's error is optimistically biased relative to the 640-district population. Because B is now the primary configuration by ratification, **every headline number inherits this caveat**, and B's coverage must not be described as nationally complete.
5. **Rate overshoot.** Five B features exceed 1.0 by up to 6.2%; interpret B coefficients as "per D5 household".
6. **Extreme under-prediction.** MAE in the top 5% is 4.4–6.5 pp with 97–100% under-prediction. The model must not be used to rank the worst-off districts.
7. **Selection noise.** The difference between `svr_rbf` and `ridge`/`elastic_net` on B (0.02 pp) is far below fold noise and is not a demonstrated ranking.
8. **Importance instability.** Only 6/43 and 10/77 features are positive in every fold; importance ranks are not stable enough to be quoted individually.
9. **Screening.** The collinearity screen was computed once on the full sample (unsupervised and frozen before modelling, but not per fold).
10. **One grid per family.** Hyperparameter grids are small by design; a wider search would change numbers slightly without changing conclusions.
11. **Fold 5 instability.** Both A models invert their ranking there (ρ ≈ −0.05, R² < 0). The reported mean is honest but conceals a fold where A fails.
12. **No calibration or prediction intervals.** Point estimates only.

## 15. Reproducibility

| Item | Value |
| --- | --- |
| Seeds | 42 (models, fold shuffles, permutation importance) |
| Outer / inner folds | 5 `GroupKFold` / 4 `GroupKFold` (grouped); 5 / 4 `KFold` (random) |
| Package versions | numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, Python 3.14.3 |
| Fold assignments | `phase3_fold_results.csv`; grouped folds identical to `phase2_folds.csv` |
| Hyperparameters | `phase3_best_parameters.json` (per fold, per model, per configuration) |
| OOF predictions | `phase3_oof_predictions.csv`, one row per district per model per scheme |
| Run metadata | `_phase3_{scheme}_meta.json` (versions, fold signatures, elapsed time) |

**Checks actually performed**
- Re-running `finalize_phase3.py` reproduced every metric CSV, JSON and error table **byte-for-byte** (sha256 compared before/after).
- Re-running `ridge` end-to-end reproduced its stored grouped MAE to within 1e-9 (`test_ridge_result_is_reproducible`).
- Regenerated grouped folds are identical to the Phase 2 fold file.
- Each fold's MAE recomputed from the OOF file using the deterministic fold assignment matches the saved fold table to 1e-9 (all three configurations, two models).
- Ridge coefficient stability was recomputed from the saved folds and per-fold α; the file reproduces identically on re-run.
- Phase 2 outputs retain their original timestamps; no raw dataset or Phase 2 artefact was modified.
- No machine-specific paths: every path derives from `Path(__file__).resolve().parents[...]`.

**Runtime (measured).** Model fitting: 210.7 s of smoke/verification runs + 478.9 s (grouped sweep, 30 model-configurations) + 943.6 s (random sweep) = **1,633 s ≈ 27 min** of cross-validation compute. Finalisation 75 s per run; notebook execution completed cleanly in one pass. Wall-clock was longer because diagnostics and other commands shared the machine.

## 16. Conclusion

- Ten model families were trained with state-grouped nested cross-validation; no state was ever split across train and test, and all preprocessing and tuning stayed inside the folds.
- The dummy baseline reproduces the Phase 2 mean baseline exactly (2.4402), confirming the evaluation harness is consistent across phases.
- Under grouped CV, Configuration A supports only modest transferable signal: best MAE 2.0293 (R² 0.170, ρ 0.463), a 16.8% improvement on the mean but only 4.4% on the Phase 2 k-NN baseline.
- Configuration B improves every regularised and ensemble model, on all five folds for seven of them, by 11–21% MAE, and lifts grouped R² from 0.184 to 0.399 and Spearman from 0.497 to 0.626. Because the comparison uses identical districts, identical folds and identical procedures, this is evidence that the D5-normalised household features add trans-state signal — not merely dimensionality.
- The best-performing model is **SVR (RBF) on Configuration B**: grouped MAE 1.7248, RMSE 2.2556, R² 0.3992, Spearman 0.6258, 29.6% better than the mean baseline and 7.0% better than Phase 2's k-NN. **Ridge (1.7490) and Elastic Net (1.7476) are statistically indistinguishable from it and materially more interpretable**, which is why Ridge is the retained primary model.
- **Ridge coefficients are interpretable but only partly stable.** 54 of 77 predictors keep their sign across all five folds; assets, household size, cooking fuel, water source and latrine type are the reliable block, while religion shares, primary education, bicycle/car ownership and male literacy flip sign and must not be read individually.
- The model compresses the distribution: it over-predicts the least deprived decile by ~2.4 pp and under-predicts the most deprived by ~4.9 pp, with 97–100% of the top 5% under-predicted. It is not a tool for identifying the worst-off districts.
- Feature importance is only moderately stable; the enduring signals are tenure, education, household size and the asset block, and they are predictive associations, not causal effects.

**Ratified at Phase 3 close (P3-D-13).** Configuration B is the primary modelling configuration and Ridge the retained primary interpretable model, with Elastic Net secondary and SVR-RBF as the accuracy benchmark; Configuration A remains the secondary full-coverage specification. No statistical superiority among the three B models is claimed.

Phase 3 stops here. No website, dashboard, API or deployment was built, and Phase 4 has not begun.

---

## Appendix A — Post-audit addition: permutation importance for the primary model

The Phase 3 importance sweep covered the selected benchmark (SVR-RBF) and extra
trees, which left a reader looking for the **primary** model's importance with
nothing to read (audit defect D-21). That gap is now closed by
`src/evaluation/ridge_importance.py`, which uses the identical protocol — each
fold's Ridge (α = 100) refitted on that fold's training rows, importance measured
on that fold's **held-out** districts by shuffling one feature at a time, 20
shuffles, seed 42. Output: `evaluation/metrics/phase3_ridge_permutation_importance.csv`.
No frozen Phase 3 artefact was modified.

| Rank | Feature | Mean importance | SD | Folds positive |
| ---: | --- | ---: | ---: | ---: |
| 1 | `hh_size_2_share` | 0.1255 | — | 4/5 |
| 2 | `tv_share` | 0.1050 | — | 5/5 |
| 3 | `electricity_share` | 0.0740 | — | 5/5 |
| 4 | `handpump_share` | 0.0586 | — | 5/5 |
| 5 | `radio_share` | 0.0487 | — | 4/5 |
| 6 | `marginal_worker_share` | 0.0483 | — | 4/5 |
| 7 | `pit_latrine_share` | 0.0419 | — | 5/5 |
| 8 | `scooter_share` | 0.0406 | — | 4/5 |
| 9 | `water_within_premises_share` | 0.0393 | — | 4/5 |
| 10 | `sc_share` | 0.0384 | — | 4/5 |

**26 of 77 features** are positive in at least four of five folds, and **7 of the
10** most important features are among the 10 largest |coefficients| in §12.1, so
the two attributions agree on the dominant block. The features that are important
but not large in coefficient terms (`handpump_share`, `pit_latrine_share`,
`water_within_premises_share`) are the ones that carry information their
correlated partners do not, which is exactly what permutation importance detects
and a coefficient cannot. As with every other attribution in this report, these
are predictive associations, not causal effects.
