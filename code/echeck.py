import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""Dump structure of the two newly arrived exports and run the HCV sum check."""
import pandas as pd, numpy as np, sys

F1 = os.path.join(C.DATA, "hcv_band_2023.csv")
F2 = os.path.join(C.DATA, "agg3c_2023.csv")

out = []
def W(s=""):
    out.append(str(s))

def dump(tag, path):
    W("=" * 70)
    W("FILE %s  %s" % (tag, path))
    try:
        d = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        W("READ FAIL: %r" % (e,)); return None
    W("shape: %s" % (d.shape,))
    W("columns: %s" % list(d.columns))
    for c in ["measure_name", "metric_name", "sex_name", "location_name"]:
        if c in d.columns:
            v = d[c].unique()
            W("%-16s n=%d  %s" % (c, len(v), list(v)[:6]))
    if "year" in d.columns:
        W("years: %d..%d (n=%d)" % (d.year.min(), d.year.max(), d.year.nunique()))
    if "cause_id" in d.columns:
        pairs = d[["cause_id", "cause_name"]].drop_duplicates().sort_values("cause_id")
        W("causes:")
        for _, r in pairs.iterrows():
            W("   %s -> %s" % (r.cause_id, r.cause_name))
    if "age_id" in d.columns:
        ap = d[["age_id", "age_name"]].drop_duplicates().sort_values("age_id")
        W("ages (n=%d): %s" % (len(ap), ", ".join("%s=%s" % (r.age_id, r.age_name) for _, r in ap.iterrows())))
    # hygiene
    key = [c for c in ["location_name", "sex_name", "age_name", "cause_name", "metric_name", "year"] if c in d.columns]
    W("dup rows: %d" % d.duplicated(subset=key).sum())
    W("NaN in val/upper/lower: %d / %d / %d" % (d.val.isna().sum(), d.upper.isna().sum(), d.lower.isna().sum()))
    W("val<0: %d   upper<val: %d   lower>val: %d" % ((d.val < 0).sum(), (d.upper < d.val).sum(), (d.lower > d.val).sum()))
    z = ((d.lower == 0) & (d.val > 0)).sum()
    W("lower==0 & val>0: %d" % z)
    W("val==0 rows: %d" % (d.val == 0).sum())
    if "upper" in d.columns:
        m = (d.val > 0) & (d.lower > 0) & (d.upper > d.lower)
        sig = np.log(d.loc[m, "upper"] / d.loc[m, "lower"]) / 3.92
        W("sigma: n=%d  min=%.3f  med=%.3f  max=%.3f" % (len(sig), sig.min(), sig.median(), sig.max()))
    W("rows per year (first 5): %s" % d.groupby("year").size().head().to_dict())
    return d

d1 = dump("E1", F1)
d2 = dump("E2", F2)

# sum check: 19 bands of one file vs All ages rows of the other, by cause/year/metric
def prep(d):
    if d is None: return None
    k = d.copy()
    k["metric_name"] = k["metric_name"].astype(str)
    return k

d1, d2 = prep(d1), prep(d2)
if d1 is not None and d2 is not None:
    W("=" * 70)
    W("SUM CHECK")
    for (na, a), (nb, b) in ((("E1", d1), ("E2", d2)), (("E2", d2), ("E1", d1))):
        if "age_id" not in a.columns or "age_id" not in b.columns:
            continue
        bands = a[a.age_id.astype(int).isin([1, 23] + list(range(4, 24)))].copy()
        allag = b[b.age_id.astype(int).isin([22])].copy()
        if len(bands) == 0 or len(allag) == 0:
            continue
        num = bands[bands.metric_name.isin(["Number", "数量"])]
        num_all = allag[allag.metric_name.isin(["Number", "数量"])]
        if len(num) == 0 or len(num_all) == 0:
            continue
        g = num.groupby(["cause_name", "year"]).val.sum().rename("bands")
        h = num_all.groupby(["cause_name", "year"]).val.sum().rename("all")
        j = pd.concat([g, h], axis=1).dropna()
        if len(j) == 0:
            continue
        j["diff"] = j.bands - j["all"]
        W("[%s bands  vs  %s All ages]  pairs=%d  max|diff|=%.6f  mean|diff|=%.6f"
          % (na, nb, len(j), j["diff"].abs().max(), j["diff"].abs().mean()))
        W(j.head(8).to_string())

W("=" * 70)
W("DONE")
open(os.path.join(C.OUT, "echeck_out.txt"), "w", encoding="utf-8").write("\n".join(out))
print("written")
