# Phase 4 — Audit Checklist

Companion to `reports/ml_report/PHASE_4_INDEPENDENT_AUDIT.md`. Every item was executed against the repository; none is taken from a prior summary.
Verdict: **VERIFIED WITH LIMITATIONS**. Reproduce with `py -3.14 _audit/audit_phase4.py` and `py -3.14 _audit/corrupt_test.py`.

Legend: **PASS** verified · **PARTIAL** verified with a documented gap · **FAIL** not met · **N/V** not verifiable.

## A. Data identity
- [PASS] 628 districts and 77 predictors match the ratified configuration
- [PASS] Target is `target_deprivation_rate_pct`, equal to `district_analysis.csv` bit-for-bit
- [PASS] p90 = 9.200 and p95 = 11.200 recomputed exactly; groups n = 64 / 35
- [PASS] 5 arms × 628 rows in OOF; no duplicate keys; no missing predictions
- [PASS] No E0 rows stored in the OOF file (reference is not a fitted arm)

## B. Folds
- [PASS] A fresh `GroupKFold(5)` on the 35 states reproduces the stored `fold` column, 0 mismatches
- [PASS] All five arms share that one fold column
- [PASS] No state appears on both sides of any fold (6/7/7/8/7 held out)
- [PASS] E0's folds identical to the Phase 4 folds

## C. Reference integrity
- [PASS] E0 is `array_equal` to the frozen Phase 3 `ridge`/grouped/B predictions
- [PASS] E0 counts as 628 rows, not refitted anywhere in the Phase 4 code

## D. Arm registration
- [PASS] E1/E2/E3 = q75 λ2, q75 λ4, q90 λ4 with α ∈ {30,100,300}
- [PASS] E4a/E4b = τ 0.75/0.90 with L1 α ∈ {0.001,0.01,0.1}
- [PASS] Runner rejects unregistered arm names; no code path widens a grid
- [PASS] Only registered α values appear in the audit trail

## E. Weight isolation
- [PASS] Weights passed as `model__sample_weight`; pipeline ≡ direct weighted `Ridge` (< 1e-10)
- [PASS] Imputer/scaler fitted inside the pipeline per fit; no global preprocessing anywhere
- [PASS] 180/180 inner thresholds equal `quantile(inner-TRAIN rows)` — recomputed from scratch
- [PASS] 180/180 inner `n_train`, `n_weighted`, `weight_mean` correct
- [PASS] Inner thresholds are split-specific (only 9/180 coincide with the outer quantile)
- [PASS] 25/25 outer refits recompute threshold on the complete outer training fold
- [PASS] Thresholds vary per fold (q75 {6.250,6.575,6.700,7.100}; q90 {8.700,8.790,9.200,9.680})
- [PASS] Outer-test corruption: α, threshold, weights, coef, intercept, predictions all Δ = 0.000e+00
- [PASS] Inner-validation corruption: threshold and weights unchanged; inner coefficients Δ = 0.000e+00
- [PARTIAL] Shipped inner test checks diagnostics but not coefficients with bit-identity (gap closed by auditor)

## F. Fit accounting
- [PASS] 325 = 180 (weighted inner) + 120 (quantile inner) + 25 (outer refits)
- [PASS] Trail holds exactly 325 rows, 65 per arm (60 inner + 5 refit)
- [PASS] No duplicate `(arm, fold, stage, inner_split, alpha)`; `inner_split` covers 1–4
- [PASS] Exactly 5 `selected=True` rows per arm
- [PASS] Recorded α is the argmin of mean inner score in 25/25 fold×arm cells
- [PASS] Stored OOF reproduces from the registered code, max |Δ| = 0.000e+00 (E1, E2, E3, E4b refit)

## G. Metrics
- [PASS] 30 headline cells × 6 arms reproduce, max |Δ| = 0.00e+00
- [PASS] Spearman reproduces for all six arms
- [PASS] 30 fold-level MAE values reproduce
- [PASS] 210 state-level biases reproduce (incl. Punjab 2.53/2.62, Assam −0.31, WB 2.79 for E4b)
- [PASS] Flag counts 25/49/62/70/68/112 and sensitivity/precision reproduce
- [PASS] Trade-off ratios reproduce (0.2024 / 0.2099 / 0.1899 / 0.1175 / 0.7667)
- [FAIL] Runtime "11.7 s" not recoverable from `tail_runtime.csv` (now 8.652 s; timing only) — D7

## H. Quantile regression
- [PASS] `pinball_loss` matches an independent piecewise definition (< 1e-12) and `pinball(0.5) = 0.5·MAE`
- [PASS] τ = 0.75 / 0.90 correctly wired into `QuantileRegressor`
- [PASS] E4a coverage 0.670382 and E4b 0.824841 reproduce to 1e-9
- [PASS] Pinball 0.746461 / 0.541137 and calibration error +7.96 / +7.52 pp reproduce
- [PASS] Reliability MAD 11.78 / 8.63 pp reproduce
- [PASS] Both arms **under-cover by ~8 pp** → correctly *not* described as calibrated
- [PARTIAL] No binomial interval for coverage, required for n = 35 — D4
- [PARTIAL] E4b extrapolates to 29.44 against a target maximum of 17.7; not mentioned — D13

## I. Bootstrap
- [PASS] States resampled, not districts; 35 clusters; 2,000 reps; seed 42 recorded
- [PASS] Paired: arm and E0 evaluated on the same resample before differencing
- [PASS] Every reported interval reproduced to |Δ| = 0.0000
- [PASS] E1 ΔMAE +0.165 [+0.022, +0.327] and Δp95 bias −0.817 [−1.077, −0.487] confirmed
- [FAIL] Pre-registration specifies a **district-level** bootstrap; execution is state-clustered — D1
- [PASS] Direction of the deviation is conservative (intervals 2–3× wider; no manufactured significance)
- [PARTIAL] Two report statements ("E4a cost interval includes zero", "E4b tail gain unstable") hold only under the executed clustering

## J. Extreme districts
- [PASS] Per-fold counts 11/20/21/10/2 (p90) and 4/12/14/5/0 (p95) match the proposal
- [PASS] Fold 5 has no p95 district — tail estimable on 4 folds only
- [PASS] Assam 12, West Bengal 12, UP 4, Orissa 4, Bihar 2, Arunachal 1; two-state share 68.6%
- [PASS] Report's geographic-concentration caveat is accurate and not softened

## K. Gates C1–C6
- [PASS] Cuts reconstructed from the proposal's multipliers: 3.1009 / +3.1567 / 1.8370 / 2.3984 / 0.4714 / 0.40
- [PASS] Independently derived gate matrix identical to the reported one — 30/30 cells
- [PASS] N/A on C1/C2 for E4a/E4b justified (different estimand); raw values still reported
- [PASS] C1 fails for E1–E3 on the p95-bias half only, as reported
- [PASS] C2 fails for all arms (+9.4% to +66.4% MAE)
- [PASS] C4 fails only E4b (WB worse, Assam 63.2%)
- [PARTIAL] E4 "specialised targeting tool" rule (C5 ∧ C6) invented in code, unspecified in the proposal — D12
- [PASS] No challenger eligible for promotion; no marginal case

## L. Conclusions
- [PASS] Tail improvement real and consistently reported (not overstated, not hidden)
- [PASS] Overall cost reported and quantified
- [PASS] Precision collapse (0.17–0.39 vs 0.40 floor) disclosed for every arm
- [PASS] Retaining Ridge B as primary is evidence-led
- [PASS] Mandated null-result sentence present verbatim, once, unaltered; no information ceiling claimed
- [PARTIAL] Report omits overall bias (E0 +0.09 → E4b −2.29) and the p95 calibration gap (10.24 vs 13.23) — D5
- [PARTIAL] Phase 4 documents never name the outcome (dilapidated census houses); repo-wide framing is correct and no homelessness claim exists — D6

## M. Phase boundary
- [PASS] No Phase 1/2 artefact postdates Phase 3 (15:35)
- [PASS] All Phase 3 metrics 16:23–16:34; OOF predictions byte-unchanged
- [PASS] Phase 3 decision documents untouched (16:33)
- [PASS] Phase 4 writes only `tail_*` files
- [PASS] No Phase 5 source, metric, prediction or report exists
- [PARTIAL] E-22 accurately describes the work but repeats the wrong q75 lower bound — D2
- [N/V] Full immutability: no git repository, no pre-Phase-4 hash manifest — timestamps only

## N. Test deliverables
- [PASS] 9/9 isolation suite passes on independent execution; Phases 1–3 at 17/17, 19/19, 21/21
- [FAIL] Pinball/coverage synthetic-calibration test required by the proposal is absent — D3
- [FAIL] Bootstrap reproducibility test required by the proposal is absent — D3
- [FAIL] Documented commands (`py ...`) no longer run: `py` now resolves to Python 3.15 without numpy — D8

## Summary

| Outcome | Count |
| --- | ---: |
| Checks passed | 69 |
| Partial (documented gap) | 8 |
| Failed (deviation / missing deliverable) | 5 |
| Not verifiable | 1 |

**Blocking for closure: none.** Fix before reuse: D1 (record the clustering change), D2 (correct 6.25), D3 (add the two tests), D4 (binomial interval), D5 (add the two omitted rows).
