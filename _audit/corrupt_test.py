"""Independent corruption tests, written from scratch (not the project's suite)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.models.tail_aware import (  # noqa: E402
    build_arms, build_estimator, fit_estimator, fit_fold, inner_folds, inner_search, sample_weights,
)
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

cfg = build_configurations(["b628"])["b628"]
X, y, g = cfg["X"], np.asarray(cfg["y"], float), np.asarray(cfg["groups"])
folds = build_outer_folds(g, "grouped")
tr, te = folds[0]
print(f"fold 1: {len(tr)} train / {len(te)} test, {len(set(g[te]))} states held out\n")

for name in ("E2", "E3"):
    arm = build_arms()[name]
    base_p, base_r, base_e = fit_fold(X, y, g, tr, te, arm, 1, return_estimator=True)

    corrupt = y.copy()
    corrupt[te] = 999.0
    corr_p, corr_r, corr_e = fit_fold(X, corrupt, g, tr, te, arm, 1, return_estimator=True)

    print(f"[{name}] OUTER-TEST CORRUPTION (y on held-out rows -> 999)")
    print(f"  alpha      {base_r['best_alpha']} -> {corr_r['best_alpha']}")
    print(f"  threshold  {base_r['threshold']:.6f} -> {corr_r['threshold']:.6f}")
    print(f"  n_weighted {base_r['n_weighted']} -> {corr_r['n_weighted']}")
    print(f"  coef max|d|      {np.max(np.abs(base_e[-1].coef_ - corr_e[-1].coef_)):.3e}")
    print(f"  intercept |d|    {abs(base_e[-1].intercept_ - corr_e[-1].intercept_):.3e}")
    print(f"  prediction max|d| {np.max(np.abs(base_p - corr_p)):.3e}")
    print(f"  -> {'ISOLATED' if np.array_equal(base_p, corr_p) and base_r['best_alpha'] == corr_r['best_alpha'] else 'LEAK'}\n")

arm = build_arms()["E2"]
X_tr, y_tr = X.iloc[tr], y[tr]
splits = inner_folds(g[tr])
i_tr, i_va = splits[0]

clean_diag: list[dict] = []
inner_search(X_tr, y_tr, g[tr], arm, 1, diagnostics=clean_diag)
corrupted = y_tr.copy()
corrupted[i_va] = 999.0
corr_diag: list[dict] = []
inner_search(X_tr, corrupted, g[tr], arm, 1, diagnostics=corr_diag)

print("[E2] INNER-VALIDATION CORRUPTION (rows that are validation for inner split 1 -> 999)")
same = True
for a in (30.0, 100.0, 300.0):
    ra = [r for r in clean_diag if r["inner_split"] == 1 and r["alpha"] == a][0]
    rb = [r for r in corr_diag if r["inner_split"] == 1 and r["alpha"] == a][0]
    ok = (ra["threshold"] == rb["threshold"] and ra["n_weighted"] == rb["n_weighted"]
          and ra["weight_mean"] == rb["weight_mean"])
    same &= ok
    print(f"  alpha {a:>5}: threshold {ra['threshold']:.6f} -> {rb['threshold']:.6f}, "
          f"n_weighted {ra['n_weighted']} -> {rb['n_weighted']}, weights identical: {ok}")

# coefficients of that split's fit at fixed alpha must be bit-identical
w, _ = sample_weights(y_tr[i_tr], arm.quantile, arm.lam)
e_a = fit_estimator(build_estimator(arm, 100.0), X_tr.iloc[i_tr], y_tr[i_tr], w)
w2, _ = sample_weights(corrupted[i_tr], arm.quantile, arm.lam)
e_b = fit_estimator(build_estimator(arm, 100.0), X_tr.iloc[i_tr], corrupted[i_tr], w2)
print(f"  inner-split coefficients at alpha=100: max|d| = "
      f"{np.max(np.abs(e_a[-1].coef_ - e_b[-1].coef_)):.3e}")
print(f"  -> {'ISOLATED' if same else 'LEAK'}")

# the other inner splits legitimately change: those rows are their TRAINING data
other = [s for i, s in enumerate(splits) if i != 0]
changed = sum(1 for i, _ in other
              if [r for r in clean_diag if r["inner_split"] == i + 1 and r["alpha"] == 100.0][0]["threshold"]
              != [r for r in corr_diag if r["inner_split"] == i + 1 and r["alpha"] == 100.0][0]["threshold"])
print(f"  (expected) other inner splits whose threshold moved: {changed}/3 — those rows were their training data")
