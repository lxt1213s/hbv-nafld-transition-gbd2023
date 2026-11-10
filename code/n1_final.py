import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""n1_final.py —— N1 定稿：选定假设 = 年龄档等相关 rho=0.30 + 跨年共用随机数。
输出层级 UI、官方宽度比、变化量 UI，并给出 rho=0 与 rho=1 作为上下界。
"""
import numpy as np
import pandas as pd

PATH = os.path.join(C.DATA, "deaths_en_1980_2023.csv")
OUT = os.path.join(C.OUT, "n1_final_out.txt")
NDRAW = 2000
BANDS = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 30, 31, 32, 235]
SHOW = [1990, 2015, 2020, 2023]
rng = np.random.default_rng(20261110)

d = pd.read_csv(PATH, encoding="utf-8-sig")
n = d[(d.metric_id == 1) & (d.age_id.isin(BANDS))].copy()
n["sigma"] = (np.log(n.upper) - np.log(n.lower)) / 3.92
causes = sorted(n.cause_id.unique())
years = sorted(n.year.unique())
val, sig, bands = {}, {}, {}
for c in causes:
    b = [x for x in BANDS if ((n.cause_id == c) & (n.age_id == x)).any()]
    bands[c] = b
    for y in years:
        s = n[(n.cause_id == c) & (n.year == y)].set_index("age_id").reindex(b)
        val[(c, y)] = s.val.values.astype(float)
        sig[(c, y)] = s.sigma.values.astype(float)
All = d[(d.metric_id == 1) & (d.age_id == 22)].set_index(["cause_id", "year"])
off = {c: {y: (All.loc[(c, y), "lower"], All.loc[(c, y), "upper"]) for y in years} for c in causes}

L = []
def p(s=""):
    L.append(str(s))

def draw_totals(rho, year_corr=True):
    out = {}
    for c in causes:
        A = len(bands[c]); Y = len(years)
        z0 = rng.standard_normal((NDRAW, Y, 1))
        eps = rng.standard_normal((NDRAW, Y, A))
        Z = np.sqrt(rho) * z0 + np.sqrt(1 - rho) * eps
        if year_corr:
            Z[:] = Z[:, [0], :]
        for yi, y in enumerate(years):
            out[(c, y)] = (val[(c, y)] * np.exp(sig[(c, y)] * Z[:, yi, :])).sum(axis=1)
    return out

def q(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

for tag, rho in (("选定 rho=0.30 / year=共用", 0.30),
                 ("下界 rho=0（年龄独立）/ year=共用", 0.0),
                 ("上界 rho=1（年龄全共用）/ year=共用", 1.0)):
    T = draw_totals(rho)
    p("=== %s ===" % tag)
    rs = []
    for c in causes:
        name = "HBV" if c == 1026 else "NAFLD"
        for y in SHOW:
            lo, hi = q(T[(c, y)])
            olo, ohi = off[c][y]
            rs.append((hi - lo) / (ohi - olo))
            p("  %-5s %d  point %10.0f  MC %10.0f-%10.0f  official %10.0f-%10.0f  ratio %.2f"
              % (name, y, val[(c, y)].sum(), lo, hi, olo, ohi, (hi - lo) / (ohi - olo)))
        ch = 100 * (T[(c, 2023)] / T[(c, 2020)] - 1)
        yr = 100 * ((T[(c, 2023)] / T[(c, 2015)]) ** (1 / 8) - 1)
        p("        2020->2023 %+.1f%%  approx95UI %+.1f to %+.1f"
          % (100 * (val[(c, 2023)].sum() / val[(c, 2020)].sum() - 1), *q(ch)))
        p("        2015->2023 %+.2f%%/yr approx95UI %+.2f to %+.2f"
          % (100 * ((val[(c, 2023)].sum() / val[(c, 2015)].sum()) ** (1 / 8) - 1), *q(yr)))
    p("  median width ratio %.2f (%.2f-%.2f)" % (float(np.median(rs)), min(rs), max(rs)))
    p()

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("written")
