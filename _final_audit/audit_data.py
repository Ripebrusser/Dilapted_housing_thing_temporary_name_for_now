"""Independent Phase 1-2 audit: raw data -> target -> features.

Recomputes from the raw CSVs with its own naming logic; does not import the
project's loaders. Read-only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path.cwd()
RAW = R / "datasets" / "raw"
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


SA = {"ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY", "DELHI": "NCT OF DELHI",
      "UTTARANCHAL": "UTTARAKHAND"}
DA = {"PONDICHERRY": "PUDUCHERRY"}


def nm(v):
    s = str(v).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def st(v):
    n = nm(v)
    return SA.get(n, n)


def dk(v):
    n = nm(v)
    return DA.get(n, n)


def key(s, d):
    return f"{st(s)}||{dk(d)}"


def strip_prefix(v):
    return re.sub(r"^(District|State)\s*-\s*", "", str(v).strip(), flags=re.IGNORECASE).strip()


H("D1 — district census 2011")
d1 = pd.read_csv(RAW / "india_census" / "india-districts-census-2011.csv", encoding="utf-8-sig", low_memory=False)
check("D1 shape 640 x 118", d1.shape == (640, 118), d1.shape)
check("D1 no duplicate rows", not d1.duplicated().any())
d1["key"] = [key(s, d) for s, d in zip(d1["State name"], d1["District name"])]
check("D1 canonical keys unique", d1["key"].is_unique, int(d1["key"].duplicated().sum()))
d1["state"] = d1["State name"].map(st)
check("D1 35 states/UTs", d1["state"].nunique() == 35, d1["state"].nunique())
hh = float(d1["Households"].sum())
say(f"  D1 Households sum = {hh:,.0f}")
pop = float(d1["Population"].sum())
say(f"  D1 Population sum = {pop:,.0f}")
check("D1 Households is the house stock (~331M), not households (~247M)", 300e6 < hh < 360e6, f"{hh/1e6:.1f}M")
dupnames = d1["District name"].map(dk).value_counts()
dupnames = dupnames[dupnames > 1]
check("6 duplicated district names across states", len(dupnames) == 6, sorted(dupnames.index))

H("D2 — HLPCA (target source)")
d2 = pd.read_csv(RAW / "india_census" / "india_census_housing-hlpca-full.csv", encoding="utf-8-sig", low_memory=False)
check("D2 shape 1908 x 156", d2.shape == (1908, 156), d2.shape)
check("D2 Rural/Urban levels = 3", sorted(d2["Rural/Urban"].astype(str).str.strip().unique()) == ["Rural", "Total", "Urban"],
      sorted(d2["Rural/Urban"].astype(str).str.strip().unique()))
tot = d2[d2["Rural/Urban"].astype(str).str.strip() == "Total"].copy()
check("D2 has 640 Total rows (one per district)", len(tot) == 640, len(tot))
tot["key"] = [key(s, strip_prefix(a)) for s, a in zip(tot["State Name"], tot["Area Name"])]
check("D2 Total keys unique", tot["key"].is_unique, int(tot["key"].duplicated().sum()))

comps = ["Total Number of Good", "Total Number of Livable", "Total Number of Dilapidated"]
s = sum(pd.to_numeric(tot[c], errors="coerce") for c in comps)
check("D2 condition components sum to ~100 per row", bool(((s >= 99.5) & (s <= 100.5)).all()),
      f"min={s.min():.3f} max={s.max():.3f}")

H("Target reconstruction and reconciliation with the processed table")
target_raw = pd.DataFrame({
    "key": tot["key"].to_numpy(),
    "target_deprivation_rate_pct": pd.to_numeric(tot["Total Number of Dilapidated"], errors="coerce").to_numpy(),
})
check("independent target has 640 rows", len(target_raw) == 640, len(target_raw))
check("independent target range inside 0-100", float(target_raw.target_deprivation_rate_pct.min()) >= 0
      and float(target_raw.target_deprivation_rate_pct.max()) <= 100,
      f"{target_raw.target_deprivation_rate_pct.min()}–{target_raw.target_deprivation_rate_pct.max()}")

proc = pd.read_csv(R / "datasets" / "processed" / "district_analysis.csv")
check("district_analysis shape 640 rows", len(proc) == 640, proc.shape)
check("district_analysis keys unique", proc["key"].is_unique)
mine = proc[["key", "target_deprivation_rate_pct"]].merge(target_raw, on="key", suffixes=("_proc", "_audit"), how="outer",
                                                          indicator=True)
check("every processed district matches an independent D2 Total row", bool((mine["_merge"] == "both").all()),
      mine["_merge"].value_counts().to_dict())
diff = (mine["target_deprivation_rate_pct_proc"] - mine["target_deprivation_rate_pct_audit"]).abs()
check("target values identical to independent reconstruction", float(diff.max()) < 1e-12, f"max|d|={diff.max():.2e}")
say(f"  target: min {proc.target_deprivation_rate_pct.min():.3f}, max {proc.target_deprivation_rate_pct.max():.3f}, "
    f"mean {proc.target_deprivation_rate_pct.mean():.4f}, median {proc.target_deprivation_rate_pct.median():.3f}")

H("D5 — PCA household counts and the 628 match")
d5 = pd.read_csv(RAW / "india_pca" / "census_2011.csv", encoding="utf-8-sig", low_memory=False)
check("D5 shape 1342 rows", len(d5) == 1342, d5.shape)
d5d = d5[d5["LEVEL"].astype(str).str.strip() == "DISTRICT"].copy()
d5d["district_key"] = d5d["NAME"].map(dk)
g = d5d.groupby("district_key").agg(no_hh=("NO_HH", "sum"), n_rows=("AREA", "size"))
say(f"  D5 district-level rows: {len(d5d)}; distinct district names: {len(g)}")
say(f"  rows per name distribution: {g['n_rows'].value_counts().to_dict()}")
freq = d1["District name"].map(dk).value_counts()
resolvable = (set(freq[freq == 1].index) & set(g.index[g["n_rows"] == 2]))
mine_hh = pd.Series([g.loc[n, "no_hh"] if n in resolvable else np.nan for n in d1["District name"].map(dk)],
                    index=d1["key"], dtype=float)
check("exactly 628 districts resolve to a D5 household count", int(mine_hh.notna().sum()) == 628,
      int(mine_hh.notna().sum()))
audit_has = mine_hh.notna()
proc_has = proc.set_index("key")["has_d5_households"].astype(bool)
check("processed has_d5_households flag matches independent resolution",
      bool((audit_has == proc_has.reindex(audit_has.index)).all()),
      int((audit_has != proc_has.reindex(audit_has.index)).sum()))
missing_keys = sorted(audit_has[~audit_has].index)
say(f"  12 unresolved districts: {missing_keys}")
check("all 12 unresolved are duplicate district names", len(missing_keys) == 12)
check("no fabricated household counts for unresolved districts",
      bool(proc.set_index('key').loc[missing_keys, 'has_d5_households'].eq(False).all()))

H("Target cross-check against the independent T3 reconstruction (D1 dilapidated / D5 households)")
num = pd.to_numeric(d1.set_index("key")["Condition_of_occupied_census_houses_Dilapidated_Households"], errors="coerce")
t3 = (num / mine_hh * 100).dropna()
pair = pd.concat([t3.rename("t3"), proc.set_index("key")["target_deprivation_rate_pct"]], axis=1).dropna()
r = float(pair["t3"].corr(pair["target_deprivation_rate_pct"]))
med = float((pair["t3"] - pair["target_deprivation_rate_pct"]).abs().median())
check("T2 agrees with independent T3 (r > 0.99)", r > 0.99, f"r={r:.6f}, median|diff|={med:.4f} pp")

H("Features: 43 (A) and 77 (B)")
man = pd.read_csv(R / "datasets" / "processed" / "feature_manifest.csv")
say(f"  manifest rows {len(man)}; retained {int(man.retained.sum())}; dropped {int((~man.retained).sum())}")
ret = man[man["retained"]]
a_cols = ret.loc[ret["configuration"] == "A", "feature_name"].tolist()
b_cols = ret.loc[ret["configuration"] != "A", "feature_name"].tolist()
check("Configuration A has 43 retained predictors", len(a_cols) == 43, len(a_cols))
check("Configuration B adds 34 -> 77 total", len(b_cols) == 34 and len(a_cols) + len(b_cols) == 77,
      f"{len(a_cols)}+{len(b_cols)}")
check("manifest A columns equal the ones in district_analysis",
      all(c in proc.columns for c in a_cols), [c for c in a_cols if c not in proc.columns][:5])
check("manifest B columns equal the ones in district_analysis",
      all(c in proc.columns for c in b_cols), [c for c in b_cols if c not in proc.columns][:5])
check("no duplicate feature names in manifest", man["feature_name"].is_unique)
check("target is not among the predictors", "target_deprivation_rate_pct" not in a_cols + b_cols)

forb = {"Condition_of_occupied_census_houses_Dilapidated_Households", "Households", "Rural_Households",
        "Urban_Households", "Total Number of Dilapidated", "Total Number of Good", "Total Number of Livable",
        "Contition_T_Dilapidated", "Contition_T_Good", "Contition_T_Livable", "Contition_T_Total",
        "Contition_R_Total", "Residence_Dilapidated", "Residence_Good", "Residence_Livable",
        "Residence_cum_other", "Residence_cum_Good"}
used = set()
for v in ret["source_columns"].astype(str):
    used.update(v.split("|"))
hits = forb & used
check("no forbidden/target-derived source column feeds any predictor", not hits, sorted(hits))

sub = proc[proc["has_d5_households"].astype(bool)]
na_a = {c: int(proc[c].isna().sum()) for c in a_cols if proc[c].isna().any()}
na_b = {c: int(sub[c].isna().sum()) for c in b_cols if sub[c].isna().any()}
check("Configuration A has no missing values", not na_a, na_a)
check("Configuration B has no missing values on the 628 subset", not na_b, na_b)
check("Configuration B features are missing for exactly the 12 excluded districts",
      all(int(proc[c].isna().sum()) == 12 for c in b_cols),
      {c: int(proc[c].isna().sum()) for c in b_cols if int(proc[c].isna().sum()) != 12})

H("Education construction: the Literate_Education subtotal must be excluded")
edu_src = man[man["feature_name"].isin(
    [c for c in man["feature_name"] if c.endswith("_share") and "education" in str(c)])]
edu_used = set()
for v in man.loc[man["configuration"] == "A", "source_columns"].astype(str):
    edu_used.update(v.split("|"))
check("Literate_Education is never used as a feature source", "Literate_Education" not in edu_used,
      sorted([c for c in edu_used if "Education" in c])[:12])
levels = ["Below_Primary_Education", "Primary_Education", "Middle_Education", "Secondary_Education",
          "Higher_Education", "Graduate_Education", "Other_Education"]
ratio_levels = float(np.nansum(d1[levels].sum(axis=0)) / np.nansum(d1["Literate_Education"]))
check("sum(7 attainment levels) / Literate_Education = 1.0 (no double counting)",
      abs(ratio_levels - 1) < 1e-3, f"ratio={ratio_levels:.6f}")
ratio_all = float(np.nansum(d1[levels].sum(axis=0)) + np.nansum(d1["Illiterate_Education"])) \
    / float(np.nansum(d1["Total_Education"]))
check("levels + illiterate partition Total_Education", abs(ratio_all - 1) < 1e-3, f"ratio={ratio_all:.6f}")
edu_feats = [c for c in a_cols if c in ("below_primary_share", "primary_share", "middle_share", "secondary_share",
                                        "higher_share", "graduate_share", "other_education_share",
                                        "illiterate_share")]
check("8 education share features retained (7 levels incl. illiterate, reference dropped)",
      len(edu_feats) == 8, edu_feats)

H("Share sanity")
bad = []
for c in a_cols + b_cols:
    v = proc[c].dropna()
    if v.min() < -1e-9 or v.max() > 1.5:
        bad.append((c, round(float(v.min()), 3), round(float(v.max()), 3)))
check("all predictors are plausible shares/rates (0-1.5)", not bad, bad[:8])

Path("_final_audit/data_audit_log.txt").write_text("\n".join(LOG), encoding="utf-8")
print("\nlog -> _final_audit/data_audit_log.txt")
