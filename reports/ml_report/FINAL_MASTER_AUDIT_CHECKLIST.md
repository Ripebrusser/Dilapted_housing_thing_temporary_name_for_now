# Final Master Audit — Checklist

Companion to `reports/ml_report/FINAL_MASTER_AUDIT.md`. Every item was executed against the repository with `py -3.14`; nothing is taken from a prior summary.

Legend: **[PASS]** verified · **[FAIL]** not met · **[PARTIAL]** verified with a documented gap · **[N/V]** not verifiable.

---

## A. Repository inventory and hygiene

- [PASS] Complete inventory produced: 176 files, sizes, timestamps, 17 extensions
- [PASS] Excluded categories stated: `__pycache__`, `*.pyc`, `.freebuff/`, binary zips, plot PNGs
- [PASS] Every `.py` file located and classified (37 files incl. 8 empty `__init__.py`)
- [FAIL] No final-prediction, ranking or inference artefact exists (see §L)
- [FAIL] `models/`, `models/final/`, `Docs/`, `reports/final_report/`, `datasets/interim/` are empty
- [FAIL] `config.yaml` is never read by any code and carries a stale phase marker
- [PARTIAL] Notebooks 06–10 executed with outputs; 02–05 and 11 are placeholders
- [PARTIAL] ≈49 MB of duplicated raw data (`samrus (1–3)`, `downloads`, `raw`); spot-hashed pairs identical
- [PASS] No untracked source file essential to the pipeline was found
- [PASS] No conflicting duplicate report versions; Phase 4 has report + verification + audit, each distinct

## B. Phase 1 — datasets

- [PASS] D1 = 640 × 118, 35 states, unique canonical keys, no duplicate rows
- [PASS] D2 = 1,908 × 156, three Rural/Urban levels, 640 `Total` rows, keys unique
- [PASS] D3 = 156-column dictionary, used for D2 only
- [PASS] D4 = 32 × 12, state-level, correctly excluded from modelling
- [PASS] D5 = 1,342 × 89; 1,268 district rows; 634 distinct names; every name exactly 2 rows
- [PASS] D1 `Households` sum = 330.8 M = census-house stock, not 246.7 M households
- [PASS] Target components sum to 99.9–100.1 for every D2 `Total` row
- [PASS] D4 is unusable as a predictor and is not used (4 states absent, not zeros)

## C. Target

- [PASS] Target = D2 `Total Number of Dilapidated` on the `Total` row, 640 districts, 0 nulls
- [PASS] Independently reconstructed target is **identical** to the stored column (0.00e+00)
- [PASS] Distribution verified: 0.2–17.7, mean 5.0348, median 4.40, sd 3.12, skew 1.154, 124 distinct
- [PASS] No use of D1's invalid `Households` denominator anywhere (deny-list + tests)
- [PASS] Agreement with independent D1÷D5 reconstruction: r = 0.999368, median |diff| 0.0530 pp
- [PASS] No target leakage: no target component or equivalent is a predictor source
- [PASS] Target is a prevalence percentage, explicitly **not** a homelessness measure in every document

## D. Geography and matching

- [PASS] D1↔D2 complete: 640/640 on canonical `(state, district)`
- [PASS] Six duplicated district names identified (Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh)
- [PASS] D5 resolves exactly 628 districts; the 12 unresolved are exactly the duplicate names
- [PASS] The 12 unresolved districts are `NaN`, never zero-filled or borrowed
- [PASS] `has_d5_households` flag in the processed table matches an independent resolution exactly
- [PASS] Aliases are additive and verified; no district is silently renamed
- [PASS] Phase 2 and Phase 3 fold assignments identical for all 640 districts

## E. Phase 2 — preprocessing

- [PASS] Type conversion and numeric parsing correct; no silent coercion to zero
- [PASS] Missing values: 0 in Configuration A; exactly 12 (same rows) per B feature
- [PASS] No clipping, capping, winsorising or outlier removal
- [PASS] No zero-fill of any denominator
- [PASS] Rural/urban aggregation for D5 correct (sum of the two district rows)
- [PASS] `Total`-row extraction and `District - ` prefix stripping correct
- [PASS] Closed compositions additive at 1.0000 (religion, education, age, income, household size)
- [PASS] Five B rates exceed 1.0 by ≤ 6.2% and are disclosed, not clipped
- [PASS] Feature construction documented for all 81 built features (formula, denominator, source, universe)
- [PARTIAL] Collinearity screen computed once on the full sample (disclosed in 3 documents)
- [PASS] No preprocessing happens inside the CV when it should be in-fold; imputer/scaler are pipeline steps
- [PASS] Rebuild reproduces the stored table exactly

## F. Phase 3 — features and configurations

- [PASS] Configuration A = 640 × 43
- [PASS] Configuration B = 628 × 77 (43 + 34)
- [PASS] Manifest retained set equals the model matrices exactly, both configurations
- [PASS] No duplicate feature name; no identifier column in any matrix
- [PASS] `Literate_Education` subtotal correctly excluded; levels ÷ literate = 1.000000
- [PASS] Levels + `Illiterate_Education` partition `Total_Education` = 1.000000
- [PASS] 4 collinearity drops reproduce exactly (`literacy_rate_female`, `nonworker_share`, `no_bathing_share`, `open_defecation_share`)
- [PASS] Target-correlation screen: max |r| = 0.528 (`electricity_share`), **0 flagged**
- [PASS] Feature names stable between training and inference (both read the frozen manifest)
- [FAIL] No EDA notebook artefact: `03_eda.ipynb` and `04_feature_engineering.ipynb` are placeholders

## G. Model implementations (all ten families)

- [PASS] All ten registered: dummy, ridge, elastic net, kNN, decision tree, random forest, extra trees, gradient boosting, HistGB, SVR-RBF
- [PASS] Candidate counts 1/5/12/8/12/8/8/8/8/6 = **76**, matching the report
- [PASS] Correct estimator classes and parameters; HistGB `early_stopping=False`
- [PASS] `Pipeline(SimpleImputer(median), [StandardScaler], model)` for every family; scaling where required
- [PASS] Seeds fixed at 42; `n_jobs=1`; deterministic across repeated runs
- [PASS] Target and features correctly separated; no identifier leakage
- [PASS] All estimators fitted on their designated training rows only
- [PASS] Dummy baseline correct: 2.4401665 reproduces the Phase 2 mean baseline 2.4402
- [PASS] Ridge on B refits **bit-for-bit**, α = 100 in all five folds
- [PASS] Elastic Net secondary (1.7476) and SVR-RBF benchmark (1.7248) match the ratified roles
- [PASS] Primary configuration is B; Ridge remains the primary interpretable model
- [FAIL] Model persistence: **none**; no `.pkl`/`.joblib` anywhere, so no reload can be verified
- [PARTIAL] `n_configs` docstring misleading (returns grid size, not "crossed with inner CV")
- [PARTIAL] Grid-free spec (`dummy_mean`) fitted on a shared object rather than a clone

## H. Validation architecture

- [PASS] 5-fold outer `GroupKFold` on 35 states; 4-fold inner on outer-training states only
- [PASS] No state appears in both train and test of any fold (all 5 folds, both configurations)
- [PASS] Inner folds carved from outer-training rows only (tested)
- [PASS] A628/B628 share districts, order, targets, groups and fold index arrays
- [PASS] Phase 2 and Phase 3 grouped folds identical (640/640)
- [PASS] Random nested CV reported separately and never used for selection
- [PASS] Hyperparameters chosen on inner folds; per-fold choices recorded and honestly varying
- [PASS] Fold sizes 512/128 (640) and 502–503/125–126 (628) match the report
- [PASS] OOF predictions complete, unique per district, finite, no duplicates

## I. Metric reproduction

- [PASS] 30 pooled metric sets recomputed from OOF: **max |Δ| = 0.00e+00**
- [PASS] 150 fold-level MAE values recomputed from rebuilt folds: **0 mismatches**
- [PASS] Spearman reproduces for all 30 configuration × model cells
- [PASS] Reference table in the brief matches `mae_mean` exactly (A640/SVR, A628/ET, B628/SVR)
- [PASS] Fold-mean vs pooled R² distinction documented in the Phase 3 report (0.3992 vs 0.4755)
- [PASS] Negative R² retained, not clipped (decision tree, fold 5 cases)
- [PASS] Baseline comparisons use identical districts and folds

## J. Importance and stability

- [PASS] Permutation importance for B628/SVR-RBF reproduces to **0.00e+00** (held-out rows, per-fold refit)
- [PASS] Ridge coefficient means reproduce to **0.00e+00**
- [PASS] Sign-consistency count 54/77 reproduces exactly
- [PASS] Report discloses limited stability (6/43, 6/43, 10/77 positive in all folds)
- [PASS] No causal language: associations only, correlated blocks flagged
- [FAIL] Ridge — the primary interpretable model — has **no** permutation importance entry; not stated in the report

## K. Phase 4 (tail experiments) and its audit

- [PASS] Five registered arms (E1, E2, E3, E4a, E4b) plus frozen E0; no extra arm
- [PASS] 325 fits reconcile: 180 weighted inner + 120 quantile inner + 25 outer refits
- [PASS] Audit trail 325 rows, 65/arm, no duplicates; α is the argmin of mean inner score in 25/25 cells
- [PASS] 180/180 inner thresholds equal the quantile of that split's inner-training rows
- [PASS] Outer-test corruption leaves α, threshold, weights, coefficients and predictions Δ = 0.000e+00
- [PASS] Gates C1–C6: independently derived matrix identical to the reported one, 30/30 cells
- [PASS] No challenger eligible for promotion; Ridge B unchanged; no later modelling authorised
- [PASS] Phase 4 OOF (3,140 rows) and gate table (30 rows) intact
- [FAIL] **D-06** q75 threshold range still stated as 6.58–7.10 in three documents (actual 6.250–7.100)
- [FAIL] **D-07** pinball/coverage synthetic test and bootstrap reproducibility test were never added (0 matches)
- [FAIL] **D-08** no binomial interval for quantile coverage (required for n = 35)
- [FAIL] **D-09** overall bias and p95 calibration gap remain absent from the Phase 4 report
- [FAIL] **D-10** Phase 4 documents still never name the outcome (0 occurrences of "dilapidated")
- [FAIL] **D-05** bootstrap remains state-clustered while the proposal still registers district-level; no amendment recorded
- [PASS] Mandated null-result sentence present verbatim and not strengthened into an information-ceiling claim
- [PASS] E4b extrapolation (max 29.44 vs target 17.7) is documented in the independent audit, though not in the Phase 4 report

## L. Phase 6 — final predictions and district outputs

- [FAIL] No district prediction table (only OOF evaluation files exist)
- [FAIL] No district ranking, no rank column, no sorted deliverable
- [FAIL] No risk classification and no documented deployment thresholds
- [FAIL] No reusable inference pipeline (`src/prediction/` contains only `__init__.py`)
- [FAIL] `notebooks/11_final_predictions.ipynb` is a 786-byte placeholder
- [PASS] On the OOF surrogate: keys unique, 628/640 coverage exact, actual values equal the target, no district duplicated
- [PASS] Units are percentage points throughout; no actual/predicted swap
- [PASS] No unvalidated classifier was invented to fill the gap

## M. Code quality

- [PASS] No destructive operations; nothing writes to `datasets/raw/`
- [PASS] No hidden execution-order dependency; all paths derive from `Path(__file__).resolve()`
- [PASS] No machine-specific absolute paths in code, config or notebooks
- [PASS] Functions have single responsibilities; no unreachable code; no abandoned imports
- [PASS] Error handling: loaders raise on missing/renamed files; row-count and uniqueness guards on every join
- [PASS] No credentials, keys, tokens, network calls, `pickle`/`joblib`, or `subprocess`
- [PARTIAL] `exec`/`eval` in `tests/run_notebooks.py` (local notebook sources only)
- [PARTIAL] Resumable merge keeps the last duplicate row; no run provenance per row
- [PARTIAL] Four Phase 1 profiling scripts inspected at function level only, not line by line

## N. Test suite

- [PASS] Phase 1: **17/17** (executed by the auditor)
- [PASS] Phase 2: **19/19**
- [PASS] Phase 3: **21/21**
- [PASS] Phase 4: **9/9**
- [PASS] 66/66 total; 0 failed, 0 skipped, 0 errors
- [PASS] Tests are substantive: numeric target reproduction, deny-list, no-zero-imputation, in-fold scaling, fold recomputation, end-to-end ridge reproduction
- [FAIL] No synthetic pinball/coverage calibration test (proposal requirement)
- [FAIL] No bootstrap reproducibility test (proposal requirement)
- [PARTIAL] Tests validate against stored artefacts; no test-time independent recomputation
- [PARTIAL] Notebook test asserts execution but not that notebook numbers equal the CSV numbers

## O. Reproducibility

- [PASS] Target, D5 resolution, feature counts, folds, OOF, metrics, gates all reproduce
- [PASS] Ridge B refit, coefficients and SVR-B permutation importance reproduce exactly
- [PASS] Phase 4 artefacts byte-identical on re-run (12/12, timing excluded)
- [PASS] Seeds, versions, fold signatures recorded (`_phase3_*_meta.json`, Phase 3 report)
- [FAIL] **D-11** Phase 2 report claims 192/48 baseline rows; files hold 288/72
- [FAIL] **D-13** documented commands use `py`, which now resolves to Python 3.15 without the stack
- [PARTIAL] `scipy` used but not pinned in `requirements.txt`
- [PARTIAL] Phase 4 runtime figures not recoverable from the stored CSV (timing only)

## P. End-to-end pipeline and documentation

- [PASS] Stages 1–11 traceable input → schema → transformation → output → consumer
- [PASS] Phases 1–4 executable from documented commands in the correct order
- [FAIL] Stages 12–14 (final predictions, ranking, persistence/inference) do not exist
- [FAIL] **D-01** README, PROJECT_SCOPE and config.yaml describe a Phase 1 project with no trained model
- [FAIL] **D-02** Scope promises district ranking; Phase 3 forbids ranking the worst-off
- [PASS] Phase 3 report numbers all trace to stored artefacts; no unsupported claim found
- [PASS] Phase 2 report is accurate apart from D-11
- [FAIL] No presentation artefact of any kind (no ppt/pptx/pdf/key)
- [N/V] Cited literature: no bibliography exists, so nothing to verify and nothing supporting the framing

## Q. Repository and submission integrity

- [PASS] Hash baseline taken before audit: 176 files (`_final_audit/hashes_before.txt`)
- [PASS] No project artefact modified by the audit; all audit output confined to `_final_audit/`
- [PASS] No Phase 5/6/7 modelling exists; no unapproved later experiment
- [PASS] Phase 1/2/3/4 artefact timestamps are chronologically consistent
- [FAIL] No git repository: history, diffs and immutability cannot be established
- [PARTIAL] `.gitignore` excludes processed data, predictions and plots — delivery mechanism must be chosen deliberately
- [PARTIAL] `_audit/` (previous audit scratch) and several empty directories remain in the tree

---

## Outcome summary

| Outcome | Count |
| --- | ---: |
| PASS | 116 |
| FAIL | 26 |
| PARTIAL | 14 |
| NOT VERIFIABLE | 1 |

**The FAIL items are the defect register in §15 of `FINAL_MASTER_AUDIT.md`.** None of them concerns the correctness of the data, the target, the features, the models, the validation or the reported metrics; they concern missing deliverables (Phase 6, presentation), stale status documents, unapplied Phase 4 errata, and two absent tests.
