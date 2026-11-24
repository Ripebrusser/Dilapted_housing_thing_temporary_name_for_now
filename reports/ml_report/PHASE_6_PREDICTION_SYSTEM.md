# Phase 6 — Final Prediction System: Implementation and Verification

**Model:** Ridge Regression, Configuration B, α = 100 (ratified in Phase 3, re-used unchanged)
**Scope:** district-level prevalence of dilapidated census houses, Census 2011
**Status:** implemented, executed, and verified (19/19 Phase 6 tests pass)

---

## 1. Why this phase existed

The independent audit found that the project had a validated *evaluation* but no
prediction system: no persisted model, no inference path, no district-level
output, and a placeholder notebook. The scope document promised a model that
"ranks districts"; nothing ranked anything. This report documents the
implementation that closes that gap and the checks applied to it.

## 2. What was built

| File | Role |
| --- | --- |
| `src/prediction/final_model.py` | Fits the ratified recipe on all 628 districts, validates the schema, fingerprints the data, persists model + metadata + manifest |
| `src/prediction/inference.py` | Reusable scoring interface with a strict input contract; never refits anything |
| `src/prediction/build_predictions.py` | Produces the prediction table, the ranking and the relative bands |
| `tests/test_phase6_inference.py` | 19 tests: persistence, schema contract, deliverables, ranking, bands |

No model selection, tuning or feature change occurred. `alpha = 100`, the 77
predictors and their order, and the target are exactly Phase 3's.

## 3. The fitted model

| Property | Value |
| --- | --- |
| Model id | `ridge_b_alpha100_v1` (version 1.0.0) |
| Estimator | `Ridge(alpha=100.0)`, `random_state=42` |
| Preprocessing | `SimpleImputer(median)` → `StandardScaler`, fitted with the model |
| Training rows | 628 districts, 35 states |
| Predictors | 77, in frozen manifest order (`feature_manifest.csv`) |
| Target | `target_deprivation_rate_pct` — share of dilapidated census houses |
| Feature matrix sha256 | recorded in `ridge_b_metadata.json` |
| Raw source sha256 | D1, D2, D5 hashes recorded |
| Dependencies | numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, Python 3.14.3 |
| Trained at | recorded in `trained_at_utc` |

Artifacts: `models/final/ridge_b_pipeline.joblib`, `ridge_b_metadata.json`,
`feature_manifest.csv`, `training_config.json`.

**Security.** The `.joblib` file is a pickle payload and can execute code when
loaded. It must only be loaded from this repository. The metadata carries the
same warning.

## 4. In-sample predictions are not validation

The model was fitted on all 628 districts, so scoring those districts reproduces
the training data. In-sample residuals are optimistically small and are **not**
reported as performance. Every output row is labelled
`prediction_type = "final_refit_in_sample(full_628)"`.

The generalisation evidence is unchanged and remains the Phase 3 grouped nested
CV: MAE 1.7490, RMSE 2.2703, R² 0.3992 (fold means); MAE 1.7496, RMSE 2.2842,
R² 0.4680, Spearman 0.6672 (pooled out-of-fold).

## 5. Deliverables and what they contain

| File | Rows | Columns |
| --- | ---: | --- |
| `final_predictions_ridge_b_628.csv` | 628 | district_key, district_name, state_name, predicted_deprivation_rate_pct, physical_range_flag, model_id, prediction_type, configuration, n_features, model_version, coverage_note |
| `district_ranking_ridge_b_628.csv` | 628 | rank, district_key, district_name, state_name, predicted_deprivation_rate_pct, physical_range_flag, relative_band, rank_scope, band_basis |
| `final_predictions_retrospective_628.csv` | 628 | predictions beside `observed_rate_pct_2011` and the residual, marked `RETROSPECTIVE ONLY` |

Observed values are kept out of the prediction file entirely and appear only in
the retrospective file, so the two cannot be confused.

### 5.1 Distribution of the predictions

| Statistic | Value (pp) |
| --- | ---: |
| Minimum | −0.636 |
| 10th percentile | 2.24 |
| 25th percentile | 3.74 |
| Median | 4.584 |
| 75th percentile | 6.08 |
| 90th percentile | 8.14 |
| Mean | 5.058 |
| Maximum | 13.535 |

Mean predicted (5.058) equals the training mean by construction of ordinary
least squares with an unpenalised intercept; this is not evidence of accuracy.

### 5.2 Top of the ranking

| Rank | District | State | Predicted (%) | Band |
| ---: | --- | --- | ---: | --- |
| 1 | Nadia | West Bengal | 13.535 | B5 |
| 2 | Koch Bihar | West Bengal | 13.423 | B5 |
| 3 | Murshidabad | West Bengal | 13.109 | B5 |
| 4 | Maldah | West Bengal | 13.027 | B5 |
| 5 | Barpeta | Assam | 12.499 | B5 |

These are the same states that dominate the extreme group in Phase 4 (24 of the
35 districts at or above p95 are in Assam and West Bengal), so the ranking is at
least internally consistent with the evaluation.

### 5.3 Two predictions are physically impossible, and are flagged

Districts rank 627 (Chamba, Himachal Pradesh, −0.286) and 628 (Daman, Daman and
Diu, −0.636) receive negative fitted values. A share cannot be negative.

**Cause.** Ridge is an unconstrained linear model. For feature vectors at the
low-precipitation edge of the space it extrapolates just below zero. This is a
property of a linear model, not a coding error: the same pipeline reproduces the
values bit-for-bit, all inputs are finite and in-schema, and the equivalent
Phase 3 out-of-fold predictions behave the same way and further from zero
(minimum **−2.4465**, maximum 14.2643), because each fold model is fitted on
fewer rows.

**Handling.** The values are published as computed and marked
`physical_range_flag = "below 0 (linear extrapolation)"`. They are **not
clipped**. Clipping would (a) hide a real property of the ratified model,
(b) have no effect on the ranking — both districts rank last either way — and
(c) contradict the treatment of out-of-range values in Phase 4. A reader who
wants a floor at zero can apply it; the raw value is preserved.

### 5.4 Bands are relative, not risk classes

`relative_band` partitions the 628 predictions into five equal groups by
predicted rank: B1 (lowest predicted) 126 districts, B2 125, B3 126, B4 125,
B5 (highest predicted) 126.

No validated external threshold for "severe district prevalence" exists in the
sources used here, and inventing cut-points such as "above 10% is severe" would
be unfounded. The bands are therefore **relative ranking quintiles**, computed
from predictions only — never from the observed target — and every row carries
`band_basis = "relative predicted quintile (equal-sized by rank), not a
validated risk class"`. They support prioritisation, not classification.

### 5.5 Ranking policy

Descending predicted prevalence; ties broken by district key ascending; stable
sort produces ranks 1…628 with no duplicates. `rank_scope` records that the rank
is a position within the 628 eligible districts, not a national percentile.
Ranking order is unaffected by the two negative values, by clipping, or by any
monotone transform of the predictions.

## 6. Verification performed

All checks below were executed; the automated versions live in
`tests/test_phase6_inference.py` (19/19 pass).

| Check | Result | Evidence |
| --- | --- | --- |
| Exactly 628 predictions | PASS | 628 rows, 628 unique `district_key` |
| No duplicate districts | PASS | `is_unique` on keys |
| No missing predictions | PASS | 0 nulls, all finite |
| Correct feature order | PASS | permuted input columns give identical predictions |
| Correct district–state mapping | PASS | 35 states, keys `STATE\|\|DISTRICT` preserved from the ratified configuration |
| Correct ranking | PASS | monotonically non-increasing; ranks exactly 1…628 |
| Correct prediction units | PASS | percentage points of census houses; `units` field in the model output |
| Reproducibility | PASS | two independent fits and two loads agree to 0.0e+00 |
| Model reload | PASS | `joblib` round-trip; max abs difference 0.0e+00 |
| Saved scaler is the training scaler | PASS | `n_samples_seen_ == 628`, means equal training means |
| Alpha unchanged | PASS | 100.0 in pipeline and metadata |
| Missing feature rejected | PASS | `SchemaError` naming the column |
| NaN / inf rejected | PASS | `SchemaError` naming the column |
| Extra column rejected | PASS | unless `allow_extra_columns=True`, then reported |
| Non-numeric rejected | PASS | `SchemaError` |
| Input frame not mutated | PASS | deep comparison before/after |
| Observed target absent from prediction file | PASS | only in the retrospective file |

## 7. How to use it

```python
from src.prediction.inference import load_scorer, predict_districts
import pandas as pd

scorer = load_scorer()                     # model + metadata + manifest order
frame = pd.read_csv("my_district_features.csv")   # must contain the 77 manifest columns
scored = predict_districts(frame)          # rejects bad schemas loudly
```

`predict_districts` accepts any number of rows, preserves `key`/`district`/`state`
when supplied, and returns `predicted_deprivation_rate_pct`. It never fits a new
scaler and never reorders features by position — order always comes from the
saved manifest.

Rebuild all deliverables:

```bash
py -3.14 src/prediction/final_model.py        # fit + persist + verify reload
py -3.14 src/prediction/build_predictions.py  # predictions, ranking, bands
py -3.14 tests/test_phase6_inference.py       # 19/19
```

## 8. Limitations of this system

1. **In-sample values.** Not performance estimates; the OOF evaluation stands.
2. **12 districts unscorable.** No D5 household denominator; excluded by design.
3. **Top-tail unreliability.** The model under-predicts the highest-prevalence
   districts by ~4.5 pp on out-of-fold data. The top of the ranking is therefore
   directionally useful but not calibrated, and should not be read as a
   magnitude.
4. **Ranks are ordinal within 628 districts.** Two districts separated by
   0.01 pp are not meaningfully ordered differently.
5. **Relative bands only.** No absolute risk classification is supported.
6. **No uncertainty intervals per district.** Bootstrap intervals were computed
   for Phase 4 arm comparisons, not for individual district predictions.
7. **2011 only.** No applicability to later years is claimed.
