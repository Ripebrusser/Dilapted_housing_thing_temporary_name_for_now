# Viva Preparation — Questions and Worked Answers

Every figure quoted here comes from `evaluation/metrics/`. Answers are written to be
explained at a whiteboard, not recited: each gives the reasoning, the number, and the
limitation that goes with it. Where the honest answer is "we do not know", it says so.

---

## A. Research framing

### A1. Why this topic?

Housing condition is one of the few deprivation dimensions that a census measures directly and comparably for every district. The Census of India classifies each census house as Good, Livable or Dilapidated, so the share classified Dilapidated is a prevalence measure with a single definition applied nationwide by one agency. That makes it a legitimate target for a prediction exercise, unlike constructs assembled from heterogeneous surveys.

### A2. What is the research problem, precisely?

Given 77 district-level socioeconomic and housing indicators from Census 2011, predict the percentage of that district's census houses classified Dilapidated, and estimate how well the prediction transfers to states the model has never seen. It is a supervised regression problem with n = 628, p = 77, and a right-skewed continuous outcome.

The *statistical* problem is not "fit a model" — that is easy. It is estimating generalisation honestly when observations are clustered by state, and characterising where the model fails.

### A3. What is your contribution?

Three things, none of which is a new algorithm:

1. A **pre-registered, state-grouped predictive benchmark** for this outcome, with the naive random-CV figure reported alongside so the leakage gap is visible (1.140 vs 1.725 MAE).
2. A **quantified tail failure**: the model under-predicts the top 5% of districts by 4.51 pp while being nearly unbiased overall (+0.09 pp), and five remedies under fixed gates all failed.
3. A **reproducible deliverable**: persisted model, 628 district predictions, ranking, schema-validated inference, and an audit trail where every number regenerates.

### A4. What is the research gap?

Studies of district housing conditions in India are mostly descriptive or correlational. What is missing is a predictive benchmark evaluated under validation that respects administrative clustering, with explicit reporting of failure modes. I do not claim no prior work exists on any component; the claim is about this combination and this level of verification.

### A5. Why district-level prevalence rather than household-level?

Because the data are district aggregates. The census publishes counts per district; there are no household records in these files. A household-level claim could not be supported, and attempting one would be the ecological fallacy in reverse.

### A6. Why 2011 data, which is old?

It is the only year with both the housing-condition classification and a consistent district geography across the predictor tables. The 2001 round does not carry the same target variable in these extracts, so a panel is impossible. The consequence is stated everywhere: the results describe 2011, not the present, and the 640-district geography has since changed.

---

## B. Dataset questions

### B1. Explain D1–D5.

- **D1** — district census 2011, 640 × 118: household amenities, demographics, education, employment. Primary predictors.
- **D2** — Houselisting and Housing Census, 1,908 × 156: district × {Rural, Urban, Total} housing condition and amenities. Source of the target.
- **D3** — 156 column labels: a dictionary for D2 only.
- **D4** — slum population, 32 × 12: state-level, and four states are simply absent, not zero.
- **D5** — Primary Census Abstract, 1,342 × 89: household counts plus 34 housing/asset predictors.

D6 (2001 PCA) is the wrong vintage for the target; D7–D9 are optional supporting tables not used; D10 is 2016 geography and does not match the 2011 districts.

### B2. Why D2 for the target?

Because it is the only table containing the Census's own classification of house condition, already aggregated to the district and already expressed as a share. Deriving the target from D1 counts would require a denominator that D1 does not correctly provide — which is exactly the error described next.

### B3. Why was D1's `Households` denominator rejected?

Because it is not a household count. The column sums to **330,835,767** across 640 districts, which matches India's **census-house stock** (330,983,625), not its **246,740,228 households**. Dividing dilapidated *houses* by that number would have produced rates about 34% too low... and, more importantly, would have been dimensionally meaningless.

The diagnostic that settled it: D1's asset columns (electricity 67.2%, television 47.2%, latrine 46.9%) are consistent with 247 million households, so the asset columns are household-based while `Households` is house-based. A single table mixes two units. The decision is recorded as D-001.

### B4. Why only 628 districts in Configuration B?

Configuration B adds 34 predictors that need a household denominator from D5. D5 is matched by district name, and six names occur in more than one state — Aurangabad, Bijapur, Bilaspur, Hamirpur, Pratapgarh, Raigarh. Those six names produce **12 districts** whose D5 row cannot be identified without guessing. Rather than guess, they are excluded.

The exclusion is non-random (it selects particular states), which is disclosed as a limitation, and Configuration A (all 640) exists precisely so the comparison can be made on the same footing.

### B5. Why not impute the 12?

Because imputing a district's household denominator from other states' districts would fabricate the very quantity that determines 34 of its predictors. Missing is not zero, and it is not an invitation to interpolate. The correct handling of unresolvable identity is exclusion with documentation.

---

## C. Machine learning

### C1. Why Ridge as the primary model?

Three reasons. The predictors are strongly collinear (asset shares measured on the same household base), so ordinary least squares would produce unstable, sign-flipping coefficients; the L2 penalty shrinks them toward zero and stabilises them. Second, the goal includes interpretation, and a linear model on standardised features gives coefficients on a comparable scale. Third, its accuracy is within noise of the best model: MAE 1.7490 against SVR's 1.7248.

### C2. Why SVR as benchmark, and why not adopt it?

SVR with an RBF kernel is the strongest non-linear candidate and is the natural check that the linear specification is not missing large structure. It wins by **0.024 pp MAE** — about 1.4% relative — which is smaller than the fold-to-fold standard deviation (0.15–0.18). Adopting it would trade a fully interpretable coefficient vector for an uninterpretable kernel expansion in exchange for a difference indistinguishable from noise.

### C3. Why not Random Forest as the primary?

Random forest reached MAE 1.8708 — clearly worse than both Ridge (1.7490) and SVR (1.7248). With n = 628 and 77 correlated predictors, tree ensembles have too little data per split to beat a well-regularised linear model. It also offers no coefficient-level interpretation, only importance.

### C4. Why use ten model families at all?

Because the choice of Ridge must be defensible rather than assumed. If only linear models had been tried, "Ridge is best" would be uninformative. Comparing ten families across 76 configurations establishes that (a) the linear model is competitive, and (b) the ceiling is a property of the data, not of one estimator's misspecification.

### C5. What is regularisation, mathematically?

For Ridge, the objective is

  minimise ‖y − Xβ‖² + α‖β‖²

The solution is β̂ = (XᵀX + αI)⁻¹Xᵀy. Adding αI to the diagonal makes the matrix invertible and shrinks coefficients; when predictors are nearly collinear, XᵀX is close to singular and small perturbations of the data cause huge swings in β̂ — the penalty removes that instability by trading a little bias for a large reduction in variance.

### C6. What does α = 100 mean here?

It is the strength of that penalty, on standardised features so the scale is meaningful: the penalty is comparable to a sum of squared coefficients of order 100. It was selected on inner folds from {1, 10, 30, 100, 300} and was chosen in all five outer folds, so the choice is stable rather than an artefact of one split. It is large enough to shrink aggressively — which contributes to the prediction-range compression discussed later — but the inner folds preferred it to α = 300.

### C7. What is overfitting, and how do you know you have avoided it?

Overfitting is fitting sample-specific noise so that training error falls while out-of-sample error rises. Evidence against it here: (a) all hyperparameters are chosen on inner folds that never contain the outer test rows; (b) the reported metrics are computed only on held-out states; (c) the random-CV figure (1.140) versus grouped (1.725) shows what over-optimism looks like, and it is not quoted as performance; (d) corruption tests confirm that outer test targets cannot influence any fitted quantity.

### C8. What is nested cross-validation, and why nested?

Two loops: an outer loop that estimates generalisation and an inner loop that selects hyperparameters. If you select hyperparameters and report performance from the same splits, the reported number is optimistically biased because the model has seen the validation data during selection. Nesting keeps selection and evaluation separate: for each outer fold, the inner loop sees only outer-training data.

Here the outer loop is 5-fold GroupKFold on state, the inner is 4-fold GroupKFold on the outer-training states, giving 25 fits per configuration per scheme.

---

## D. Metrics

### D1. Explain MAE.

Mean absolute error: (1/n)Σ|yᵢ − ŷᵢ|, in percentage points. It is the average magnitude of error, robust to outliers relative to squared error, and directly interpretable: 1.75 pp means the typical district's predicted prevalence is off by about 1.75 points. It is the primary metric because the outcome is in percentage points and because the tail's influence should not be silently amplified.

### D2. Explain RMSE.

Root mean squared error: √((1/n)Σ(yᵢ − ŷᵢ)²). It penalises large errors quadratically, so RMSE ≥ MAE always, and the gap between them indicates how concentrated the errors are in a few cases. Ridge B: MAE 1.749, RMSE 2.270 — the ratio 1.30 signals a moderately heavy-tailed error distribution, consistent with the tail problem.

### D3. Explain R².

R² = 1 − SS_res/SS_tot, the fraction of variance explained relative to predicting the mean. It can be negative — the decision tree gets −0.184 — which means it is worse than the mean, and that value is reported rather than clipped. R² 0.399 means about 40% of between-district variance is captured and 60% is not.

### D4. Explain Spearman correlation.

Pearson correlation of ranks. It measures whether the model orders districts correctly, independent of magnitude. For a ranking use case it is arguably more relevant than R²: Ridge B's 0.631 means the ordering is substantially better than chance, even though the magnitudes are compressed. Elastic Net scores 0.6444, the best of the set.

### D5. Why is macro F1 not the primary metric?

Because there are no classes. F1 is defined on a confusion matrix; using it here would require dichotomising a continuous outcome at an arbitrary threshold, and the threshold choice would drive the metric. Macro-averaging would add a second arbitrary choice (which classes to average). Where a detection framing is genuinely needed — Phase 4's "does this arm flag the extreme districts?" — sensitivity and precision are reported against the frozen p90/p95 cuts, explicitly as a read-out rather than a model metric.

### D6. What does a negative R² mean?

The model's squared error exceeds that of predicting the training mean, i.e. it is actively worse than a constant. The decision tree reaches −0.184: its piecewise-constant fit transfers badly across states, so on unseen states it adds error rather than removing it.

### D7. Why is grouped CV harder than random CV, numerically?

Because districts within a state share unobserved conditions — administrative capacity, climate, building materials, reporting practice. A random split places near-duplicates on both sides, so the model can interpolate within a state it has already seen. Grouping by state forces extrapolation to a new state context. The measured cost is 1.725 − 1.140 = 0.585 pp MAE, a 51% relative increase. Only the grouped number is a generalisation estimate.

### D8. Why are there two sets of numbers, fold mean and pooled?

They answer different questions. The fold mean averages five independent state-held-out MAE values and describes typical performance on a new set of states; the pooled value computes one MAE over all 628 out-of-fold predictions and weights larger folds more heavily. They differ (1.7490 vs 1.7496 for MAE; 0.3992 vs 0.4680 for R²) because R² is nonlinear. The report labels which is which and never mixes them.

---

## E. Feature interpretation

### E1. Explain permutation importance.

Shuffle one feature's values in the held-out rows, recompute the error, and measure the increase. If the model relies on the feature, shuffling degrades predictions. It is model-agnostic and measured out of fold, so it reflects predictive reliance rather than fitted structure.

### E2. Explain coefficient stability.

For each of the 77 features, record the coefficient in each of the five outer-fold models. If the sign is the same in all five, the association is at least consistent across state subsets; if it flips, the feature's contribution depends on which states are in training. Here **54 of 77** keep their sign; **23 flip** and are not interpreted. The top features by magnitude — `tv_share` (−0.473), `hh_size_2_share` (−0.449), `electricity_share` (−0.394) — are negative in 5/5 folds, with small standard deviations (0.026–0.039).

### E3. Why are correlated features hard to interpret?

When two predictors carry nearly the same information, the loss surface is flat along the direction that trades one against the other. The coefficients can be individually large and opposite-signed while their sum is stable — the model is identified only in combination. Ridge shrinks this instability but does not remove the ambiguity: a coefficient is a partial association *given the rest of the correlated block*, not an isolated effect.

### E4. Why is correlation not causation here, specifically?

Because the design cannot support a causal claim: the data are cross-sectional, observational and aggregated. There is no intervention, no temporal ordering, no instrument, and no control for unobserved district characteristics. Additionally, several predictors are measured on the same household base as the outcome's determinants, so reverse and common-cause pathways are indistinguishable. The correct statement is "districts with more televisions have less dilapidated housing, conditional on the other features" — and nothing about what would happen if televisions increased.

---

## F. Phase 4

### F1. What was the extreme-district problem?

Aggregate MAE hid a systematic failure. Among the 35 districts at or above the 95th percentile (11.20%), the mean actual prevalence is 13.231 but the mean prediction is 8.722 — an under-prediction of **4.510 pp** — while the model is nearly unbiased overall (+0.092). Variance-reducing estimators compress toward the mean, so the districts of greatest interest are precisely the ones predicted worst.

### F2. Why was tail weighting attempted?

Because the failure is directional, not random. If errors were symmetric noise, nothing could be done; because errors are systematically negative in the tail, reweighting the loss toward high-target districts is the natural first remedy: it changes what the estimator optimises without changing the estimator class.

### F3. Why did overall MAE get worse?

By construction. Reweighting shifts the fitted function toward high-target districts, which necessarily moves predictions away from the bulk of the distribution where most districts live. The cost is measurable: +9.5% (E1), +12.9% (E2), +13.5% (E3), +66.4% (E4b). The relevant question is not "did it get worse" but "was the trade acceptable", and the pre-registered answer was no: the cost allowance was 5% and the tail requirement was a 30% reduction, and no arm met both.

### F4. What are sensitivity and precision here?

Both are defined against frozen thresholds, not fitted classes. An arm "flags" a district if its prediction is ≥ 9.20 (p90). Sensitivity at p95 = of the 35 truly extreme districts, what fraction was flagged. Precision at p95 = of the districts flagged, what fraction were truly extreme. E2 found 24 of 35 (sensitivity 0.686) but flagged 62 districts, so precision was 0.387 — below the 0.40 floor. More flags buy sensitivity and destroy precision.

### F5. What is quantile regression?

Instead of estimating the conditional mean E[y|x], it estimates the conditional τ-quantile Q_τ(y|x) by minimising the asymmetric absolute loss

  ρ_τ(r) = max(τr, (τ−1)r),  r = y − ŷ

Under-prediction is penalised with weight τ and over-prediction with weight 1−τ. At τ = 0.5 this reduces to median regression. Here τ = 0.75 and 0.90 estimate the upper part of each district's conditional distribution.

### F6. What is pinball loss, and why is it the right metric for those arms?

Pinball loss is the mean of ρ_τ above. It is the loss the estimator minimises, so comparing it to a mean-regression model's pinball loss is the like-for-like comparison: E4b scores 0.541 against the reference's 0.912 at the same τ. This is exactly why the MAE-based gates were marked NOT APPLICABLE for quantile arms — a τ-quantile is not a mean predictor, and penalising it with MAE would be a category error.

### F7. What did the bootstrap deviation mean?

The proposal registered a district-level paired bootstrap; the executed analysis resampled **states**. The stated reason — tail districts cluster in four states, so districts are not independent — is statistically defensible, but it was not what was pre-registered, and it must not be presented as if it were.

The consequence is interpretive. District-level intervals are 1.07–3.55× narrower. Under district resampling, E4a's cost interval excludes zero (+0.165 [+0.078, +0.250]) and E4b's tail interval excludes zero (−2.404, −0.553); under state clustering, both include zero. The direction of every point estimate is unchanged, so no gate flips — but the claim "E4b's tail gain is unstable" is method-dependent, and both results are now reported.

### F8. Why were no challengers promoted?

Because the acceptance rule was fixed before the results existed. A candidate required C1 (≥30% p95-bias reduction) and C2 (≤5% overall cost) plus fold consistency, state spread and no over-correction. Weighted arms reached 18–28% on C1; every arm exceeded C2; E4b failed C4 by worsening West Bengal; all failed C6 on precision. Loosening a threshold after seeing the numbers would convert a test into a search.

### F9. If the tail bias fell, why say "no improvement"?

Because the registered statement is about improvement *sufficient to meet the criteria*. The measured fall of 0.8–1.4 pp is real, consistent across folds and, under district resampling, distinguishable from zero — but it is below the 30% bar and was bought with 9–66% more overall error. Both halves are reported; that is the whole point of pre-registration.

### F10. Could a better weighting pass?

Possibly — a continuous rather than step weighting, or a weight tied to the p95 group rather than q75/q90. That is a new experiment requiring new authorisation; adding arms after seeing these results is the post-hoc search pre-registration exists to prevent. The measured exchange rate (0.12–0.21 pp overall MAE per pp of tail bias) suggests the cost would remain substantial.

---

## G. Final predictions

### G1. What does the final model output?

For each eligible district, a predicted prevalence in percentage points: the model's estimate of the share of that district's census houses classified Dilapidated in 2011. Plus identifiers, a model id, a prediction-type label and a physical-range flag.

### G2. What is a district ranking?

The 628 predictions sorted descending, with rank 1 = highest predicted prevalence. Ties are broken by district key so the order is deterministic. Rank is a position in this list, not a national percentile, and adjacent ranks separated by hundredths of a point are not meaningfully ordered.

### G3. How does inference work?

`load_scorer()` reads the persisted pipeline, metadata and manifest. `predict_districts()` then: checks every one of the 77 manifest features is present; rejects non-numeric columns; rejects NaN and infinite values; rejects unexpected columns unless explicitly tolerated; reorders columns to manifest order (so input order cannot corrupt results); applies the *saved* scaler — never refitting; predicts; checks the output is finite; and returns identifiers with predictions. Failures raise a `SchemaError` naming the offending columns.

### G4. What is the difference between OOF and final fitted predictions?

Out-of-fold predictions come from five models, each fitted without the districts it scores, so they are honest evidence of generalisation: MAE 1.749. Final predictions come from one model fitted on all 628 districts and scored on those same districts: they reproduce the training data and are **not** a performance estimate — hence the `refit_in_sample` label. Their residuals would look impressive and would be meaningless as validation.

### G5. What are the model's limitations?

Under-prediction of the top 5% by ~4.5 pp; R² 0.40 leaving most variance unexplained; 12 districts unscorable; predictions compressed (max 13.535 against an observed 17.7); two fitted values slightly negative because Ridge is unconstrained — flagged, not clipped; relative bands rather than risk classes; no per-district uncertainty; and 2011-only applicability. Ranks are ordinal, and the model should not be the sole basis for allocating resources to the extreme group.

---

## H. Deliberately difficult questions

### H1. "Your R² is 0.40. Why should anyone use this?"

Because the use case is ranking, not forecasting magnitudes, and 0.40 with Spearman 0.63 is a usable ordering signal — better than a mean baseline by 0.70 pp MAE. But the honest answer includes the second half: the ordering is worst exactly where it matters most, since the top 5% is under-predicted by 4.5 pp, so the ranking identifies broad priority regions rather than individual severe districts. If the decision requires precise magnitudes at the top, this model is not sufficient.

### H2. "Your best model is only 1.4% better than Ridge. Isn't the whole model comparison pointless?"

The comparison's purpose is to establish that Ridge is not a lazy choice. Ten families and 76 configurations were evaluated; the best alternative wins by less than the fold-level standard error and at the cost of interpretability. A negative result on "is a more complex model needed?" is a legitimate finding, and it also bounds what further modelling on these features can achieve.

### H3. "You tried five remedies and all failed. Was Phase 4 wasted?"

No — it converts a suspicion into a measurement. Before Phase 4 the tail problem was known but its trade-off was not: we now know the tail bias can be reduced 18–34%, that doing so costs 9–66% more overall error, and that the exchange rate is 0.12–0.21 pp of MAE per pp of tail bias. That rules out a whole class of fixes and points to missing predictors rather than missing loss functions.

### H4. "Your bootstrap was not what you registered. Isn't that a fatal flaw?"

It is a real deviation and is reported as one. It is not fatal because: the point estimates are identical under both procedures; the gates are point-estimate rules unaffected by either interval; and both intervals are now reported, with the two conclusions that depend on the choice explicitly flagged. What would have been fatal is presenting the state-clustered result as if it had been pre-registered.

### H5. "Two districts have negative predicted prevalence. Isn't that a bug?"

It is a property of an unconstrained linear model, not a coding defect: the same numbers reproduce bit-for-bit, all inputs are finite and in-schema, and the out-of-fold predictions also go negative (minimum −2.4465). Ridge extrapolates linearly, and for feature vectors at the low-dilapidation edge the hyperplane crosses zero. I report the values and flag them rather than clipping, because clipping would hide the model's behaviour and would not change the ranking — both districts rank last either way.

### H6. "Why should we believe your target is right? You derived it from the same data you predict."

Three independent checks. The target was reconstructed from the raw files with a separate naming implementation and matched to 0.00e+00. A count-based reconstruction (D1 dilapidated ÷ D5 households) correlates at r = 0.999368 with a median difference of 0.053 pp. And the component shares (Good + Livable + Dilapidated) sum to 100 ± 0.1 for every district. The target is the Census's own computed share; I did not compute it from a formula of my own.

### H7. "Your features are just prosperity proxies. What does the model actually learn?"

Largely a single socioeconomic gradient — asset ownership, housing materials and household structure load on one dimension, which is why 23 coefficients flip sign and why the top-10 permutation features carry 57% of importance. That is a substantive finding about the data: district housing condition in 2011 is strongly associated with a broad development gradient, and the 77 predictors add only modest independent information beyond it. It is also why the tail is hard: the extreme districts differ from the bulk in ways these proxies do not capture.

### H8. "If someone re-ran this, would they get the same numbers?"

Yes, under the documented environment: Python 3.14.3 with the pinned versions, seed 42, `n_jobs = 1`, and documented commands. This was tested rather than assumed — the target, all 30 metric sets, 150 fold values, the Ridge refit, coefficients, importance and the Phase 4 artefacts all reproduce to 0.00e+00 or byte-identically. The one practical trap is that `py` on this machine resolves to Python 3.15 without the scientific stack; every command specifies `py -3.14`.

### H9. "You are one student with no domain expert. Why trust the feature definitions?"

You should check them, and the repository is built to make that possible: every feature has a documented formula, numerator, denominator and universe in `PHASE_2_DATA_PREPARATION_REPORT.md`, the deny-list is enforced in code and tested, and the one construct that was ambiguous (the education subtotal) is excluded with the arithmetic shown. Where I could not verify something — provenance of the delivered CSV extracts against censusindia.gov.in — the limitation is stated rather than papered over.

### H10. "What would you do differently with another three months?"

Replace features, not models. The tail failure and the 0.40 R² both point at missing information: housing age, construction material, climate exposure, migration. I would also add per-district prediction intervals, since a ranking without uncertainty invites over-confident allocation, and a multilevel model with state random effects to exploit the clustering that grouped CV currently only defends against.

---

## Quick-reference numbers

| Item | Value |
| --- | --- |
| Districts / states | 640 (A) · 628 (B) · 35 |
| Predictors | 43 (A) · 77 (B) |
| Target | mean 5.058, median 4.40, sd 3.12, range 0.2–17.7 |
| Ridge B (fold mean) | MAE 1.7490 · RMSE 2.2703 · R² 0.3992 · ρ 0.6311 |
| SVR B (benchmark) | MAE 1.7248 · RMSE 2.2556 · R² 0.3992 · ρ 0.6258 |
| Dummy baseline | MAE 2.4495 |
| Random vs grouped (SVR) | 1.1401 vs 1.7248 |
| p95 group | n = 35, actual 13.231, predicted 8.722, bias +4.510 |
| Best tail remedy | E4b, −33.6% bias, +66.4% MAE — rejected |
| Exchange rate | 0.12–0.21 pp MAE per pp tail bias |
| Final predictions | 628, range −0.636 to 13.535, 2 flagged |
| Tests | 99 passing, 0 skipped |
