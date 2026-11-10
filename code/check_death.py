import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""check_death.py —— GBD 导出文件体检（编码 utf-8-sig，输出全部写入 txt，避免控制台乱码）。"""
import sys
import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.DATA, "deaths_en_1980_2023.csv")
OUT = os.path.join(C.OUT, "check_out.txt")

L = []
def p(s=""):
    L.append(str(s))

d = pd.read_csv(PATH, encoding="utf-8-sig")
p("=== FILE ===")
p(PATH)
p("shape: %s   ncol=%d" % (d.shape, d.shape[1]))
p("columns: %s" % d.columns.tolist())
p()

p("=== DIMENSIONS ===")
p("measure: %s" % list(zip(d.measure_id.unique(), d.measure_name.unique())))
p("metric : %s" % list(zip(d.metric_id.unique(), d.metric_name.unique())))
p("sex    : %s" % list(zip(d.sex_id.unique(), d.sex_name.unique())))
p("location: n=%d  %s" % (d.location_id.nunique(), d.location_name.unique()[:5]))
p("year   : %d..%d  n=%d  %s" % (d.year.min(), d.year.max(), d.year.nunique(),
                                 sorted(d.year.unique())[:3] + ["..."] + sorted(d.year.unique())[-3:]))
p("population_group: %s" % d.population_group_name.unique().tolist())
p()

p("=== CAUSES (英文 name 核对) ===")
for _, r in d.groupby(["cause_id", "cause_name"]).size().reset_index(name="rows").iterrows():
    p("  id=%-6s name=%-60s rows=%d" % (r.cause_id, r.cause_name, r.rows))
p()

p("=== AGE LABELS ===")
for m in d.metric_name.unique():
    sub = d[d.metric_name == m]
    p("  [%s] n_age=%d" % (m, sub.age_name.nunique()))
    for _, r in sub.groupby(["age_id", "age_name"]).size().reset_index(name="n").sort_values("age_id").iterrows():
        p("     id=%-4s %-24s rows=%d" % (r.age_id, r.age_name, r.n))
p()

p("=== ROW MATH ===")
per_year = d.groupby("year").size()
p("rows per year: unique=%s  min=%d  max=%d" % (sorted(per_year.unique()), per_year.min(), per_year.max()))
for c, g in d.groupby("cause_name"):
    p("  cause %-55s rows=%d  per-year=%s" % (c, len(g), sorted(g.groupby("year").size().unique())))
p()

p("=== DUPLICATES ===")
key = ["location_id", "sex_id", "age_id", "cause_id", "metric_id", "year", "measure_id"]
dup = d[d.duplicated(subset=key, keep=False)]
p("duplicate rows on full key: %d" % len(dup))
if len(dup):
    p(dup.sort_values(key).head(12).to_string())
p()

p("=== MISSING ===")
p("NaN per column: %s" % {c: int(n) for c, n in d.isna().sum().items() if n > 0})
p("rows with any NaN: %d" % d.isna().any(axis=1).sum())
p("empty-string cells: %d" % (d.astype(str) == "").sum().sum())
p()

p("=== VALUE SANITY ===")
p("val==0 rows: %d" % (d.val == 0).sum())
p("val<0 rows: %d" % (d.val < 0).sum())
p("upper<val: %d ; lower>val: %d" % ((d.upper < d.val).sum(), (d.lower > d.val).sum()))
p("lower==0 & val>0: %d" % ((d.lower == 0) & (d.val > 0)).sum())
pos = d[d.val > 0]
sig = (np.log(pos.upper) - np.log(pos.lower)) / 3.92
p("sigma ln(U/L)/3.92: min %.3f  median %.3f  max %.3f" % (sig.min(), sig.median(), sig.max()))
p()

p("=== HEAD 3 LINES (raw) ===")
with open(PATH, encoding="utf-8-sig") as f:
    for i, line in enumerate(f):
        if i > 3:
            break
        p(line.rstrip()[:300])

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("written:", OUT)
