"""Independent Phase 3/4 audit: recompute metrics, refit, verify importance.

Read-only. Nothing is written into project directories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance

R = Path.cwd()
sys.path.insert(0, str(R))

from src.models.registry import build_models  # noqa: E402
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

M = R / "evaluation" / "metrics"
P = R / "evaluation" / "predictions"
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def check(name, cond, detail=""):
    say(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def H(t):
    say("")
    say("=" * 76)
    say(t)
    say("=" * 76)


def met(y, p):
    r = np.asarray(y, float) - np.asarray(p, float)
    out = {"mae": float(np.mean(np.abs(r))), "rmse": float(np.sqrt(np.mean(r * r))),
           "r2": float(1 - np.sum(r * r) / np.sum((y - np.mean(y)) ** 2))}
    out["spearman"] = float(stats.spearmanr(y, p).statistic) if np.std(p) > 0 else np.nan
    return out


H("A. OOF predictions: completeness and configuration coverage")
oof = pd.read_csv(P / "phase3_oof_predictions.csv")
say("columns:", list(oof.columns))
counts = oof.groupby(["configuration", "scheme", "model"]).size()
say("rows per (config, scheme, model):", counts.value_counts().to_dict())
expected = {"A_primary_640": 640, "A_restricted_628": 628, "B_augmented_628": 628}
ok = all(int(counts.loc[(c, s, m)]) == n
         for (c, s, m), n in counts.items() for c in [c] for n in [expected[c]])
check("every configuration x scheme x model has the expected row count", ok)
check("10 model families present", oof["model"].nunique() == 10, sorted(oof["model"].unique()))
check("no missing predictions", oof["predicted"].notna().all())
check("predictions finite", bool(np.isfinite(oof["predicted"]).all()))
dupe = oof.duplicated(["configuration", "scheme", "model", "key"]).sum()
check("no duplicated district predictions", int(dupe) == 0, int(dupe))

proc = pd.read_csv(R / "datasets" / "processed" / "district_analysis.csv")
tmap = proc.set_index("key")["target_deprivation_rate_pct"]
for c in expected:
    block = oof[oof["configuration"] == c]
    same_actual = np.allclose(block["actual"].to_numpy(), tmap.loc[block["key"]].to_numpy(), atol=0, rtol=0)
    check(f"{c}: 'actual' equals the processed target exactly", bool(same_actual))

sub = proc[proc["has_d5_households"].astype(bool)]
a_keys = set(oof[oof["configuration"] == "A_restricted_628"]["key"])
b_keys = set(oof[oof["configuration"] == "B_augmented_628"]["key"])
check("A_restricted_628 and B_augmented_628 cover the same 628 districts", a_keys == b_keys)
check("A_restricted_628 equals the has_d5_households=True subset", a_keys == set(sub["key"]))
check("A_primary_640 covers all 640 districts", len(set(oof[oof["configuration"] == "A_primary_640"]["key"])) == 640)

H("B. Feature matrices")
for cid, n_expect in (("a640", 43), ("a628", 43), ("b628", 77)):
    cfg = build_configurations([cid])[cid]
    X = cfg["X"]
    check(f"{cid}: {n_expect} predictors, {len(cfg['y'])} rows, no missing",
          X.shape == (len(cfg["y"]), n_expect) and not X.isna().any().any(), X.shape)
    check(f"{cid}: no identifier columns in the matrix",
          not ({"key", "state", "district", "district_code", "target_deprivation_rate_pct"} & set(X.columns)))

cfg_a628, cfg_b628 = build_configurations(["a628"])["a628"], build_configurations(["b628"])["b628"]
check("A628/B628 share districts, order, targets and groups",
      np.array_equal(cfg_a628["keys"], cfg_b628["keys"]) and np.array_equal(cfg_a628["y"], cfg_b628["y"])
      and np.array_equal(cfg_a628["groups"], cfg_b628["groups"]))
fa = build_outer_folds(cfg_a628["groups"], "grouped")
fb = build_outer_folds(cfg_b628["groups"], "grouped")
check("A628/B628 use identical outer folds",
      all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(fa, fb)))
f640 = build_outer_folds(build_configurations(["a640"])["a640"]["groups"], "grouped")
whole = all(len(set(np.asarray(cfg_a628["groups"])[te])) * 0 == 0 for _, te in fa)
overlap = sum(len(set(np.asarray(cfg_a628["groups"])[tr]) & set(np.asarray(cfg_a628["groups"])[te])) for tr, te in fa)
check("no state appears in both train and test of any 628 fold", overlap == 0, overlap)
overlap640 = sum(len(set(np.asarray(build_configurations(['a640'])['a640']['groups'])[tr])
                     & set(np.asarray(build_configurations(['a640'])['a640']['groups'])[te])) for tr, te in f640)
check("no state appears in both train and test of any 640 fold", overlap640 == 0, overlap640)

H("C. Independent metric recomputation vs phase3_nested_cv_results.csv")
res = pd.read_csv(M / "phase3_nested_cv_results.csv")
say("columns:", [c for c in res.columns][:14], "...")
worst_pooled = 0.0
worst_mean = 0.0
for _, row in res.iterrows():
    blk = oof[(oof["configuration"] == row["configuration"]) & (oof["scheme"] == "grouped")
              & (oof["model"] == row["model"])]
    m = met(blk["actual"].to_numpy(), blk["predicted"].to_numpy())
    for k in ("mae", "rmse"):
        worst_pooled = max(worst_pooled, abs(m[k] - float(row[f"pooled_{k}"])))
    if np.isfinite(row["pooled_r2"]):
        worst_pooled = max(worst_pooled, abs(m["r2"] - float(row["pooled_r2"])))
    if np.isfinite(row["pooled_spearman"]):
        worst_pooled = max(worst_pooled, abs(m["spearman"] - float(row["pooled_spearman"])))
check("30 pooled metric sets reproduce from OOF predictions", worst_pooled < 1e-9, f"max|d|={worst_pooled:.2e}")

H("D. Fold-level metrics recomputed from independently built folds")
foldres = pd.read_csv(M / "phase3_fold_results.csv")
g = foldres[foldres["scheme"] == "grouped"]
fold_map_by_config = {}
for cid, label, folds_list in (("a640", "A_primary_640", f640),
                               ("a628", "A_restricted_628", fa),
                               ("b628", "B_augmented_628", fa)):
    kk = build_configurations([cid])[cid]["keys"].to_numpy()
    mapping = {}
    for fid, (tr, te) in enumerate(folds_list, start=1):
        for i in te:
            mapping[kk[i]] = fid
    fold_map_by_config[label] = mapping
bad = 0
checked = 0
for (c, mdl), grp in g.groupby(["configuration", "model"]):
    blk = oof[(oof["configuration"] == c) & (oof["scheme"] == "grouped") & (oof["model"] == mdl)].copy()
    blk["fold"] = blk["key"].map(fold_map_by_config[c])
    if blk["fold"].isna().any():
        bad += 1
        continue
    for fold_id, fb2 in blk.groupby("fold"):
        saved = grp[grp["fold"] == fold_id]
        if saved.empty:
            bad += 1
            continue
        recomputed = met(fb2["actual"].to_numpy(), fb2["predicted"].to_numpy())["mae"]
        if abs(recomputed - float(saved["mae"].iloc[0])) > 1e-9:
            bad += 1
        checked += 1
check(f"all {checked} fold-level MAE values reproduce", bad == 0, f"{bad} mismatches")

H("E. The reference table quoted in the audit brief")
sel = pd.read_csv(M / "phase3_selected_models.csv")
say(sel.round(4).to_string(index=False))
quoted = [("A_primary_640", "svr_rbf", 2.0293, 2.6144, 0.1699, 0.4626),
          ("A_restricted_628", "extra_trees", 2.0888, 2.6540, 0.1837, 0.4966),
          ("B_augmented_628", "svr_rbf", 1.7248, 2.2556, 0.3992, 0.6258)]
for c, m_, mae, rmse, r2, rho in quoted:
    row = res[(res["configuration"] == c) & (res["model"] == m_)].iloc[0]
    hit_mean = all(abs(float(row[k]) - v) < 5e-4 for k, v in
                   (("mae_mean", mae), ("rmse_mean", rmse), ("r2_mean", r2), ("spearman_mean", rho)))
    hit_pool = all(abs(float(row[k]) - v) < 5e-4 for k, v in
                   (("pooled_mae", mae), ("pooled_rmse", rmse), ("pooled_r2", r2), ("pooled_spearman", rho)))
    say(f"  {c}/{m_}: mean-based match={hit_mean}, pooled-based match={hit_pool} "
        f"(stored mean {row['mae_mean']:.4f}/{row['rmse_mean']:.4f}/{row['r2_mean']:.4f}/{row['spearman_mean']:.4f}; "
        f"pooled {row['pooled_mae']:.4f}/{row['pooled_rmse']:.4f}/{row['pooled_r2']:.4f}/{row['pooled_spearman']:.4f})")
    check(f"quoted figures for {c}/{m_} match the stored results", hit_mean or hit_pool)

ridge_b = res[(res["configuration"] == "B_augmented_628") & (res["model"] == "ridge")].iloc[0]
svr_b = res[(res["configuration"] == "B_augmented_628") & (res["model"] == "svr_rbf")].iloc[0]
say(f"\n  B628 Ridge  MAE {ridge_b['mae_mean']:.4f} / R2 {ridge_b['r2_mean']:.4f}")
say(f"  B628 SVR    MAE {svr_b['mae_mean']:.4f} / R2 {svr_b['r2_mean']:.4f}")
check("SVR-RBF B is the best-performing model on grouped MAE", svr_b["mae_mean"] == res[res["configuration"] == "B_augmented_628"]["mae_mean"].min())
check("Ridge B is not the lowest-MAE model (it is the interpretable primary)",
      float(ridge_b["mae_mean"]) > float(svr_b["mae_mean"]))

H("F. Refit Ridge on B and compare with the stored OOF predictions")
params = json.loads((M / "phase3_best_parameters.json").read_text())
per_fold = params["grouped_best_parameters"]["best_params"]["B_augmented_628"]["ridge"]["per_fold"]
cfg = cfg_b628
X, y = cfg["X"], cfg["y"]
est = build_models()["ridge"].estimator
oof_ridge = (oof[(oof["configuration"] == "B_augmented_628") & (oof["scheme"] == "grouped")
                 & (oof["model"] == "ridge")].set_index("key")["predicted"])
mine = np.full(len(y), np.nan)
alphas = []
for i, (tr, te) in enumerate(fa):
    model = clone(est)
    if per_fold[i]:
        model.set_params(**per_fold[i])
    model.fit(X.iloc[tr], y[tr])
    mine[te] = model.predict(X.iloc[te])
    alphas.append(float(model.named_steps["model"].alpha))
d = np.max(np.abs(mine - oof_ridge.loc[cfg["keys"]].to_numpy()))
check("Ridge B OOF reproduces bit-for-bit from the registered pipeline", d < 1e-10, f"max|d|={d:.2e}")
check("alphas per fold match the stored parameters", alphas == [float(p["model__alpha"]) for p in per_fold], alphas)

H("G. Coefficient stability recomputed")
stab = pd.read_csv(M / "phase3_coefficient_stability.csv")
say("columns:", list(stab.columns))
coefs = {}
for i, (tr, _) in enumerate(fa):
    model = clone(est)
    if per_fold[i]:
        model.set_params(**per_fold[i])
    model.fit(X.iloc[tr], y[tr])
    coefs[f"fold_{i+1}"] = model.named_steps["model"].coef_.copy()
mat = pd.DataFrame(coefs, index=X.columns)
mine_mean = mat.mean(axis=1)
joined = stab.set_index("feature")["coef_mean"]
dd = (mine_mean - joined.reindex(mine_mean.index)).abs().max()
check("coefficient means reproduce", float(dd) < 1e-9, f"max|d|={dd:.2e}")
check("sign-consistent count reported matches recomputation",
      int(stab["sign_consistent"].sum()) == int(((mat > 0).sum(axis=1).isin([0, 5])).sum()),
      f"{int(stab['sign_consistent'].sum())} vs {int(((mat > 0).sum(axis=1).isin([0, 5])).sum())}")

H("H. Permutation importance recomputed for the selected model (SVR-RBF on B)")
imp = pd.read_csv(M / "phase3_permutation_importance.csv")
say("configs/models with importance:", imp.groupby(["configuration", "model"]).size().to_dict())
check("Ridge (primary interpretable model) has no permutation importance entry",
      not (imp["model"] == "ridge").any())
imp_label, imp_model = "B_augmented_628", "svr_rbf"
bimp = imp[(imp["configuration"] == imp_label) & (imp["model"] == imp_model)]
say(f"  rows for {imp_label}/{imp_model}: {len(bimp)} (features: {len(X.columns)})")
per_fold_svr = params["grouped_best_parameters"]["best_params"][imp_label]["svr_rbf"]["per_fold"]
est_svr = build_models()["svr_rbf"].estimator
vals = {}
for i, (tr, te) in enumerate(fa):
    model = clone(est_svr)
    if per_fold_svr[i]:
        model.set_params(**per_fold_svr[i])
    model.fit(X.iloc[tr], y[tr])
    r = permutation_importance(model, X.iloc[te], y[te], n_repeats=10, random_state=42,
                               scoring="neg_mean_absolute_error")
    vals[f"fold_{i+1}"] = r.importances_mean
recomputed = pd.DataFrame(vals, index=X.columns).mean(axis=1)
joined = bimp.set_index("feature")["importance_mean"]
dd = (recomputed - joined.reindex(recomputed.index)).abs().max()
check("permutation importance reproduces", float(dd) < 1e-8, f"max|d|={dd:.2e}")
say("  top 8 recomputed: " + ", ".join(f"{k}={v:.4f}" for k, v in recomputed.sort_values(ascending=False).head(8).items()))
say("  top 8 stored:     " + ", ".join(f"{k}={v:.4f}" for k, v in joined.sort_values(ascending=False).head(8).items()))

H("I. Phase 4 chain intact")
tail = pd.read_csv(P / "tail_oof_predictions.csv")
check("Phase 4 OOF still 5 arms x 628", len(tail) == 3140 and tail["arm"].nunique() == 5, len(tail))
check("Phase 4 gate table still 30 rows", len(pd.read_csv(M / "tail_gate_results.csv")) == 30)
check("Phase 3 OOF untouched (628 ridge/grouped/B rows)",
      int(((oof["configuration"] == "B_augmented_628") & (oof["scheme"] == "grouped")
           & (oof["model"] == "ridge")).sum()) == 628)

Path("_final_audit/model_audit_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("\nlog -> _final_audit/model_audit_log.txt")
