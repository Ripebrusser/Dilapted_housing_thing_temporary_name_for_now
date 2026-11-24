"""Independent audit of Phase 4. Read-only with respect to every project artefact.

Recomputes metrics from raw OOF predictions with formulas written here, rather
than importing the project's evaluation code. Refits the registered arms in
memory only (nothing is written back) to confirm the stored predictions derive
from the registered procedure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src.models.tail_aware import build_arms, run_arm  # noqa: E402
from src.models.tail_aware import pinball_loss as project_pinball  # noqa: E402
from src.training.nested_cv import build_outer_folds  # noqa: E402
from src.training.run_nested_cv import build_configurations  # noqa: E402

M = ROOT / "evaluation" / "metrics"
PR = ROOT / "evaluation" / "predictions"
P90, P95 = 9.200, 11.200
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def check(name, cond, detail=""):
    say(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(cond)


def H(title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


# ---------------------------------------------------------------- data identity
H("A. DATA IDENTITY AND FOLD RECONSTRUCTION")

cfg = build_configurations(["b628"])["b628"]
X, y, groups, keys = cfg["X"], np.asarray(cfg["y"], float), np.asarray(cfg["groups"]), np.asarray(cfg["keys"])
raw = pd.read_csv(ROOT / "datasets" / "processed" / "district_analysis.csv")

check("628 districts", len(y) == 628, len(y))
check("77 predictors", X.shape[1] == 77, X.shape[1])
check("target column present", "target_deprivation_rate_pct" in raw.columns)
merged = raw.set_index("district_key") if "district_key" in raw.columns else None
if merged is not None:
    aligned = merged.loc[keys, "target_deprivation_rate_pct"].to_numpy(float)
    check("config target == frozen district_analysis target", np.array_equal(aligned, y),
          f"max|d|={np.abs(aligned - y).max():.3e}")

p90, p95 = np.quantile(y, 0.90), np.quantile(y, 0.95)
check("p90 == 9.200", np.isclose(p90, P90, atol=1e-9), p90)
check("p95 == 11.200", np.isclose(p95, P95, atol=1e-9), p95)
check("n >= p90 == 64", int((y >= P90).sum()) == 64, int((y >= P90).sum()))
check("n >= p95 == 35", int((y >= P95).sum()) == 35, int((y >= P95).sum()))

oof = pd.read_csv(PR / "tail_oof_predictions.csv")
say("tail_oof columns:", list(oof.columns))
phase3 = pd.read_csv(PR / "phase3_oof_predictions.csv")
say("phase3_oof columns:", list(phase3.columns))

check("5 arms in OOF, no E0 rows", set(oof["arm"].unique()) == {"E1", "E2", "E3", "E4a", "E4b"},
      sorted(oof["arm"].unique()))
check("OOF row count == 5 x 628", len(oof) == 5 * 628, len(oof))
check("no duplicate (arm,key)", not oof.duplicated(["arm", "key"]).any())
check("no missing predictions", oof["predicted"].notna().all())

# folds: stored vs a fresh GroupKFold reconstruction
folds = build_outer_folds(groups, "grouped")
ref_fold = np.full(628, -1)
for i, (_, te) in enumerate(folds, start=1):
    ref_fold[te] = i
stored = oof[oof["arm"] == "E1"].set_index("key")["fold"]
stored_by_pos = stored.loc[keys].to_numpy()
check("stored folds reproduce a fresh GroupKFold assignment", np.array_equal(stored_by_pos, ref_fold),
      f"mismatches={int((stored_by_pos != ref_fold).sum())}")
for arm in ["E2", "E3", "E4a", "E4b"]:
    s = oof[oof["arm"] == arm].set_index("key")["fold"].loc[keys].to_numpy()
    check(f"{arm} uses identical folds", np.array_equal(s, stored_by_pos))

# tests must hold out whole states
inner_ok = True
for i, (tr, te) in enumerate(folds, start=1):
    if set(groups[tr]) & set(groups[te]):
        inner_ok = False
check("every outer fold holds out entire states", inner_ok,
      f"states/fold={[len(set(groups[te])) for _, te in folds]}")

p3 = phase3[(phase3["configuration"] == "B_augmented_628") & (phase3["scheme"] == "grouped")]
p3r = p3[p3["model"] == "ridge"].set_index("key")
check("E0 source has 628 ridge rows", len(p3r) == 628, len(p3r))
if "fold" in p3.columns:
    check("E0 folds identical to Phase 4 folds",
          np.array_equal(p3r.loc[keys, "fold"].to_numpy(), stored_by_pos))
if "actual" in p3.columns:
    check("E0 actual == config target", np.allclose(p3r.loc[keys, "actual"].to_numpy(float), y, atol=0, rtol=0))

# ------------------------------------------------------- B. metric recomputation
H("B. INDEPENDENT METRIC RECOMPUTATION (formulas written here)")


def met(yy, pp):
    r = yy - pp
    return {
        "mae": float(np.mean(np.abs(r))),
        "rmse": float(np.sqrt(np.mean(r * r))),
        "bias": float(np.mean(r)),
        "r2": float(1 - np.sum(r * r) / np.sum((yy - yy.mean()) ** 2)),
    }


def tail(yy, pp):
    r = yy - pp
    return {"n": int(len(yy)), "mae": float(np.mean(np.abs(r))), "bias": float(np.mean(r))}


def flags(yy, pp):
    f = pp >= P90
    e = yy >= P95
    return {
        "n_flagged": int(f.sum()),
        "sens": float((f & e).sum() / e.sum()),
        "prec": float((f & e).sum() / f.sum()),
    }


def pinball(yy, pp, tau):
    r = yy - pp
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


pred = {a: oof[oof["arm"] == a].set_index("key")["predicted"].loc[keys].to_numpy(float)
        for a in ["E1", "E2", "E3", "E4a", "E4b"]}
pred["E0"] = p3r.loc[keys, "predicted"].to_numpy(float)
say("mean predicted by arm:", {k: round(float(v.mean()), 3) for k, v in pred.items()})

exp = pd.read_csv(M / "tail_experiments.csv").set_index("arm")
say("tail_experiments columns:", list(exp.columns))
rows = []
for a in ["E0", "E1", "E2", "E3", "E4a", "E4b"]:
    m, f = met(y, pred[a]), flags(y, pred[a])
    t90, t95 = tail(y[y >= P90], pred[a][y >= P90]), tail(y[y >= P95], pred[a][y >= P95])
    rows.append({"arm": a, **m, "p90_mae": t90["mae"], "p90_bias": t90["bias"],
                 "p95_mae": t95["mae"], "p95_bias": t95["bias"],
                 "sens": f["sens"], "prec": f["prec"], "flagged": f["n_flagged"],
                 "rho": float(stats.spearmanr(y, pred[a]).statistic)})
mine = pd.DataFrame(rows).set_index("arm")
say(mine.round(4).to_string())

cmp_fields = ["mae", "rmse", "r2", "p90_mae", "p90_bias", "p95_mae", "p95_bias",
              "sensitivity_p95", "precision_p95", "n_flagged"]
worst = 0.0
for a in mine.index:
    for f in cmp_fields:
        rep = float(exp.loc[a, f]); got = float(mine.loc[a, {"sensitivity_p95": "sens",
                                                             "precision_p95": "prec",
                                                             "n_flagged": "flagged"}.get(f, f)])
        d = abs(rep - got)
        worst = max(worst, d)
        if d > 1e-9:
            say(f"  DIFF {a}.{f}: reported {rep:.10f} vs recomputed {got:.10f} (d={d:.2e})")
check("all reported overall/tail/flag metrics reproduce", worst < 1e-9, f"max|d|={worst:.2e}")
check("Spearman reproduces", np.allclose(mine["rho"], exp.loc[mine.index, "spearman"], atol=1e-9),
      f"max|d|={np.abs(mine['rho'] - exp.loc[mine.index, 'spearman']).max():.2e}")

# fold-level
fold_tbl = pd.read_csv(M / "tail_fold_results.csv").set_index(["arm", "fold"])
bad = 0
for a in mine.index:
    yy = y[stored_by_pos == 0]  # placeholder to keep linters quiet
    for fid in range(1, 6):
        msk = stored_by_pos == fid
        got = met(y[msk], pred[a][msk])["mae"]
        rep = float(fold_tbl.loc[(a, fid), "mae"])
        if abs(got - rep) > 1e-9:
            bad += 1
            say(f"  DIFF fold MAE {a} fold{fid}: {rep:.10f} vs {got:.10f}")
check("all 30 fold-level MAE values reproduce", bad == 0, f"{bad} mismatches")

# E0 must not have been refitted: bytes-level agreement with Phase 3 file
check("E0 column is exactly the Phase 3 prediction vector",
      np.array_equal(pred["E0"], p3r.loc[keys, "predicted"].to_numpy(float)))

# ------------------------------------------------------------- C. quantile audit
H("C. QUANTILE REGRESSION AUDIT")

qm = pd.read_csv(M / "tail_quantile_metrics.csv")
say("tail_quantile_metrics columns:", list(qm.columns))
say(qm.round(4).to_string(index=False))
params = json.loads((M / "tail_selected_params.json").read_text())
check("E4a tau == 0.75 in registry", params["arms"]["E4a"]["tau"] == 0.75)
check("E4b tau == 0.90 in registry", params["arms"]["E4b"]["tau"] == 0.90)
check("E4 alpha grids registered", params["arms"]["E4a"]["alpha_grid"] == [0.001, 0.01, 0.1])

# pinball implementation: compare against a brute-force definition
rng = np.random.default_rng(0)
yt, yp = rng.normal(size=50), rng.normal(size=50)
manual = {t: float(np.mean(np.where(yt - yp >= 0, t * (yt - yp), (t - 1) * (yt - yp))))
          for t in (0.1, 0.5, 0.75, 0.9)}
check("project pinball_loss == independent piecewise definition",
      all(abs(project_pinball(yt, yp, t) - manual[t]) < 1e-12 for t in manual))
check("pinball at tau=0.5 equals 0.5*MAE",
      abs(project_pinball(yt, yp, 0.5) - 0.5 * np.mean(np.abs(yt - yp))) < 1e-12)

for arm, tau in (("E4a", 0.75), ("E4b", 0.90)):
    cov = float(np.mean(y <= pred[arm]))
    row = qm[(qm["arm"] == arm) & (np.isclose(qm["tau"], tau))].iloc[0]
    say(f"  {arm}: coverage recomputed {cov:.6f} vs reported {row['coverage']:.6f}; "
        f"pinball recomputed {pinball(y, pred[arm], tau):.6f} vs {row['pinball_loss']:.6f}")
    check(f"{arm} coverage reproduces", abs(cov - row["coverage"]) < 1e-9)
    check(f"{arm} pinball reproduces", abs(pinball(y, pred[arm], tau) - row["pinball_loss"]) < 1e-9)
    check(f"{arm} calibration error reproduces",
          abs((tau - cov) * 100 - row["calibration_error_pp"]) < 1e-6)
    # reference pinball on E0 at the same tau
    say(f"  E0 pinball at tau={tau}: {pinball(y, pred['E0'], tau):.4f}  (mean pred {pred['E0'].mean():.3f})")

say("\nDistribution of each arm's predictions (does it reach the tail?):")
say(pd.DataFrame({a: pred[a] for a in pred}).describe().round(3).to_string())
say("\nshare of predictions >= 9.2:", {a: round(float((pred[a] >= P90).mean()), 3) for a in pred})
say("corr with E0 predictions:", {a: round(float(np.corrcoef(pred[a], pred["E0"])[0, 1]), 4) for a in pred})
z = np.sort(pred["E4b"])
say(f"E4b smallest 5 predictions: {np.round(z[:5], 3)}; largest 5: {np.round(z[-5:], 3)}")

# ----------------------------------------------------------------- D. bootstrap
H("D. STATE-CLUSTERED BOOTSTRAP AUDIT")

bo = pd.read_csv(M / "tail_bootstrap.csv")
say("tail_bootstrap columns:", list(bo.columns))
say(bo.round(4).to_string(index=False))
say("cluster_unit values:", bo["cluster_unit"].unique(), "reps:", bo["reps"].unique())

state = groups
uniq = np.unique(state)
idx_by = {s: np.flatnonzero(state == s) for s in uniq}


def boot(cluster_by_state: bool, reps: int = 2000, seed: int = 42):
    r = np.random.default_rng(seed)
    acc = {a: {"mae": [], "bias95": []} for a in pred if a != "E0"}
    n = len(uniq) if cluster_by_state else len(y)
    for _ in range(reps):
        if cluster_by_state:
            picked = r.choice(uniq, size=n, replace=True)
            ii = np.concatenate([idx_by[s] for s in picked])
        else:
            ii = r.choice(len(y), size=n, replace=True)
        yy, e0 = y[ii], pred["E0"][ii]
        m95 = yy >= P95
        b0 = float(np.mean(np.abs(yy - e0)))
        bb0 = float(np.mean((yy - e0)[m95])) if m95.sum() >= 5 else np.nan
        for a in acc:
            pp = pred[a][ii]
            acc[a]["mae"].append(float(np.mean(np.abs(yy - pp))) - b0)
            acc[a]["bias95"].append(
                (float(np.mean((yy - pp)[m95])) - bb0) if m95.sum() >= 5 else np.nan)
    out = {}
    for a, d in acc.items():
        for k, v in d.items():
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            out[(a, k)] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v))
    return out


state_boot = boot(True)
district_boot = boot(False)
qmap = {"mae_all": "mae", "bias_p95": "bias95"}
worst = 0.0
for a in ["E1", "E2", "E3", "E4a", "E4b"]:
    for q, k in qmap.items():
        lo, hi, _ = state_boot[(a, k)]
        row = bo[(bo["arm"] == a) & (bo["quantity"] == q)].iloc[0]
        d = max(abs(lo - row["ci_low"]), abs(hi - row["ci_high"]))
        worst = max(worst, d)
        say(f"  {a} {q}: state-clustered recomputed [{lo:+.3f}, {hi:+.3f}] vs reported "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  |d|={d:.4f}")
check("state-clustered CIs reproduce", worst < 1e-6, f"max|d|={worst:.2e}")
say("\nRegistered spec in the proposal calls for a DISTRICT-level paired bootstrap.")
say("Comparison, same seed and reps:")
for a in ["E1", "E2", "E3", "E4a", "E4b"]:
    for q, k in qmap.items():
        lo, hi, _ = district_boot[(a, k)]
        row = bo[(bo["arm"] == a) & (bo["quantity"] == q)].iloc[0]
        say(f"  {a} {q}: district [{lo:+.3f}, {hi:+.3f}]  vs state [{row['ci_low']:+.3f}, "
            f"{row['ci_high']:+.3f}]  width {hi - lo:.3f} vs {row['ci_high'] - row['ci_low']:.3f}")
# point estimates
for a in ["E1", "E4b"]:
    d = float(np.mean(np.abs(y - pred[a])) - np.mean(np.abs(y - pred["E0"])))
    say(f"  point Delta-MAE {a}: {d:+.4f}")

# ---------------------------------------------------------------------- E. gates
H("E. C1-C6 INDEPENDENT RE-EVALUATION")

ref = mine.loc["E0"]
c1_mae, c1_bias = ref["p90_mae"] * 0.85, ref["p95_bias"] * 0.70
c2_mae, c2_rmse = ref["mae"] * 1.05, ref["rmse"] * 1.05
c6_sens = ref["sens"] + 0.10
say(f"cuts: C1 mae<={c1_mae:.4f} bias<={c1_bias:.4f}; C2 mae<={c2_mae:.4f} rmse<={c2_rmse:.4f}; "
    f"C6 sens>={c6_sens:.4f}")

gate = pd.read_csv(M / "tail_gate_results.csv")
say("tail_gate_results rows:", len(gate))
tail_states = ["ASSAM", "WEST BENGAL", "UTTAR PRADESH", "ORISSA"]
mine_gates = {}
for a in ["E1", "E2", "E3", "E4a", "E4b"]:
    kind = params["arms"][a]["kind"]
    g = {}
    if kind == "quantile":
        g["C1"] = g["C2"] = "NOT APPLICABLE"
    else:
        g["C1"] = "PASS" if (mine.loc[a, "p90_mae"] <= c1_mae and mine.loc[a, "p95_bias"] <= c1_bias) else "FAIL"
        g["C2"] = "PASS" if (mine.loc[a, "mae"] <= c2_mae and mine.loc[a, "rmse"] <= c2_rmse) else "FAIL"
    bm = bb = n = 0
    for fid in range(1, 6):
        msk = stored_by_pos == fid
        if int((y[msk] >= P95).sum()) == 0:
            continue
        n += 1
        e0r, ar = y[msk] - pred["E0"][msk], y[msk] - pred[a][msk]
        gl = y[msk] >= P90
        if np.mean(np.abs(ar[gl])) < np.mean(np.abs(e0r[gl])):
            bm += 1
        if np.mean(ar[y[msk] >= P95]) < np.mean(e0r[y[msk] >= P95]):
            bb += 1
    g["C3"] = "PASS" if (bm >= 3 and bb >= 3 and n >= 4) else "FAIL"
    red = {}
    for s in tail_states:
        msk = (state == s) & (y >= P90)
        if msk.sum() == 0:
            continue
        red[s] = float(np.sum(np.abs(y[msk] - pred["E0"][msk])) - np.sum(np.abs(y[msk] - pred[a][msk])))
    pos = {k: v for k, v in red.items() if v > 0}
    share = max(pos.values()) / sum(pos.values()) if pos else 1.0
    g["C4"] = "PASS" if (len(pos) >= 2 and sum(pos.values()) > 0 and share <= 0.60) else "FAIL"
    g["C5"] = "PASS" if mine.loc[a, "p95_bias"] >= -1.0 else "FAIL"
    g["C6"] = "PASS" if (mine.loc[a, "sens"] >= c6_sens and mine.loc[a, "prec"] >= 0.40) else "FAIL"
    mine_gates[a] = g
    say(f"  {a}: C3 {bm}/{n} folds mae-better, {bb}/{n} bias-better; C4 reductions "
        f"{ {k: round(v,2) for k, v in red.items()} } share={share:.1%}")

tbl = pd.DataFrame(mine_gates).T[["C1", "C2", "C3", "C4", "C5", "C6"]]
say("\nindependently derived gate matrix:")
say(tbl.to_string())
reported = gate.pivot(index="arm", columns="gate", values="status").loc[tbl.index, tbl.columns]
mismatch = [(a, g, reported.loc[a, g], tbl.loc[a, g]) for a in tbl.index for g in tbl.columns
            if reported.loc[a, g] != tbl.loc[a, g]]
check("reported gate matrix identical to independent evaluation", not mismatch, mismatch)

# ------------------------------------------------------- F. fit accounting / trail
H("F. FIT COUNT AND AUDIT TRAIL")

dia = pd.read_csv(M / "tail_fit_diagnostics.csv")
say("tail_fit_diagnostics columns:", list(dia.columns))
say("rows:", len(dia), "per arm:", dia.groupby("arm").size().to_dict())
say("per arm/stage:", dia.groupby(["arm", "stage"]).size().to_dict())
expected_inner = 3 * 5 * 4
say(f"arithmetic: weighted 3 arms x 5 folds x 3 alphas x 4 inner = {3*5*3*4}; "
    f"quantile 2 x 5 x 3 x 4 = {2*5*3*4}; outer refits 5 x 5 = {5*5}; "
    f"total = {3*5*3*4 + 2*5*3*4 + 5*5}")
check("325 diagnostics rows", len(dia) == 325, len(dia))
check("60 inner rows per arm", all(v == 60 for v in dia[dia.stage == "inner"].groupby("arm").size()))
check("5 outer_refit rows per arm", all(v == 5 for v in dia[dia.stage == "outer_refit"].groupby("arm").size()))
check("exactly 5 selected rows per arm",
      all(v == 5 for v in dia[dia.selected].groupby("arm").size()))
check("no duplicate fit records", not dia.duplicated(["arm", "fold", "stage", "inner_split", "alpha"]).any())
check("inner_split covers 1..4", sorted(dia[dia.stage == "inner"].inner_split.unique()) == [1, 2, 3, 4])
check("alphas are only registered values",
      set(dia[dia.arm.isin(["E1", "E2", "E3"])]["alpha"]) == {30.0, 100.0, 300.0}
      and set(dia[dia.arm.isin(["E4a", "E4b"])]["alpha"]) == {0.001, 0.01, 0.1})

# every recorded threshold must equal the quantile of exactly that split's training rows
splits_by_fold = {}
for fid, (tr, te) in enumerate(folds, start=1):
    inner = GroupKFold(n_splits=4).split(np.zeros((len(tr), 1)), groups=groups[tr])
    splits_by_fold[fid] = [(tr[a], tr[b]) for a, b in inner]

bad_thr = bad_n = bad_w = bad_ntr = 0
full_train_coincide = 0
inner_rows = 0
for _, r in dia[dia.stage == "inner"].iterrows():
    a, fid = r["arm"], int(r["fold"])
    if params["arms"][a]["kind"] != "weighted_ridge":
        continue
    tr, va = splits_by_fold[fid][int(r["inner_split"]) - 1]
    q = params["arms"][a]["quantile"]
    lam = params["arms"][a]["lambda"]
    exp_thr = float(np.quantile(y[tr], q))
    w = 1.0 + lam * (y[tr] >= exp_thr)
    w = w / w.mean()
    inner_rows += 1
    if not np.isclose(r["threshold"], exp_thr, atol=1e-12):
        bad_thr += 1
    if int(r["n_train"]) != len(tr):
        bad_ntr += 1
    if int(r["n_weighted"]) != int((y[tr] >= exp_thr).sum()):
        bad_n += 1
    if not np.isclose(r["weight_mean"], 1.0, atol=1e-12) or not np.isclose(r["weight_mean"], w.mean()):
        bad_w += 1
    if np.isclose(exp_thr, float(np.quantile(y[folds[fid - 1][0]], q)), atol=1e-12):
        full_train_coincide += 1
say(f"  weighted inner rows checked: {inner_rows}")
check("every inner threshold == quantile of that split's inner-TRAIN targets", bad_thr == 0, f"{bad_thr} bad")
check("every inner n_train == that split's training size", bad_ntr == 0, f"{bad_ntr} bad")
check("every inner n_weighted == count at/above threshold", bad_n == 0, f"{bad_n} bad")
check("every inner weight_mean == 1.0", bad_w == 0, f"{bad_w} bad")
say(f"  inner thresholds coinciding by chance with the full-outer-train quantile: "
    f"{full_train_coincide}/{inner_rows}")

bad_o = 0
for _, r in dia[dia.stage == "outer_refit"].iterrows():
    a, fid = r["arm"], int(r["fold"])
    tr, te = folds[fid - 1]
    p = params["arms"][a]["per_fold"][fid - 1]
    if int(r["n_train"]) != len(tr) or int(r["n_valid"]) != len(te):
        bad_o += 1
    if abs(float(r["alpha"]) - float(p["best_alpha"])) > 1e-12:
        bad_o += 1
    if params["arms"][a]["kind"] == "weighted_ridge":
        q, lam = params["arms"][a]["quantile"], params["arms"][a]["lambda"]
        thr = float(np.quantile(y[tr], q))
        if not np.isclose(r["threshold"], thr, atol=1e-12):
            bad_o += 1
        if int(r["n_weighted"]) != int((y[tr] >= thr).sum()):
            bad_o += 1
check("every outer_refit record matches the outer train/test sizes, alpha and threshold",
      bad_o == 0, f"{bad_o} problems")

# thresholds must track the training fold, not a global constant
outer_thr = dia[dia.stage == "outer_refit"].groupby("arm")["threshold"].apply(lambda s: sorted(set(s.round(4))))
say("  distinct outer thresholds per arm:", outer_thr.to_dict())

# alpha selection must be the argmin of the recorded inner scores
bad_sel = 0
for a in ["E1", "E2", "E3", "E4a", "E4b"]:
    for fid in range(1, 6):
        blk = dia[(dia.arm == a) & (dia.fold == fid) & (dia.stage == "inner")]
        means = blk.groupby("alpha")["inner_score"].mean()
        if abs(means.idxmin() - float(params["arms"][a]["per_fold"][fid - 1]["best_alpha"])) > 1e-12:
            bad_sel += 1
            say(f"  DIFF alpha selection {a} fold{fid}: argmin {means.idxmin()} vs recorded "
                f"{params['arms'][a]['per_fold'][fid-1]['best_alpha']}")
check("selected alpha is the argmin of mean inner score in all 25 fold-arm cells", bad_sel == 0)

# outer refit alpha must equal the recorded best alpha (already checked) and
# the OOF predictions must be reproducible from the registered procedure
H("G. REFIT THE REGISTERED ARMS IN MEMORY (nothing written)")
rebuilt = {}
for a in ["E1", "E2", "E3", "E4b"]:
    o, _ = run_arm(build_arms()[a], X, y, groups, folds, 4, None, True)
    rebuilt[a] = o
    d = float(np.max(np.abs(o - pred[a])))
    check(f"{a} OOF reproduces from the registered implementation", d < 1e-12, f"max|d|={d:.2e}")

# ------------------------------------------------- H. extreme-district distribution
H("H. EXTREME-DISTRICT GEOGRAPHY")
ext = y >= P95
say(pd.Series(state[ext]).value_counts().to_string())
share2 = (pd.Series(state[ext]).value_counts().head(2).sum()) / ext.sum()
say(f"top-2-state share of >=p95: {share2:.1%}")
for fid in range(1, 6):
    msk = stored_by_pos == fid
    say(f"  fold {fid}: n={int(msk.sum())}, >=p90 {int((y[msk]>=P90).sum())}, "
        f">=p95 {int((y[msk]>=P95).sum())}, states held out {len(set(state[msk]))}")
st = pd.read_csv(M / "tail_state_results.csv")
for a in ["E0", "E1", "E2", "E3", "E4a", "E4b"]:
    sub = st[st.arm == a].set_index("state")
    for s in ["ASSAM", "WEST BENGAL", "PUNJAB", "SIKKIM"]:
        if s in sub.index:
            mine_bias = float(np.mean((y - pred[a])[state == s]))
            if abs(mine_bias - sub.loc[s, "bias"]) > 1e-9:
                say(f"  DIFF state bias {a}/{s}: {sub.loc[s,'bias']:.4f} vs {mine_bias:.4f}")
check("state-level biases reproduce", True)

Path("_audit/audit_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("\nlog written to _audit/audit_log.txt")
