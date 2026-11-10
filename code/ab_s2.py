import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""Table S2: NAFLD per-band fitted slope and single-band marginal effect (four windows)."""
import numpy as np, pandas as pd

DALY = os.path.join(C.DATA, "daly_band_2023.csv")
WPP = os.path.join(C.DATA, "WPP2024_pop_GBDagebands_1990_2035.csv")
OUT = os.path.join(C.OUT, "ab_s2_out.txt")

BANDS = ["5岁以下", "5-14岁", "15-19岁", "20-24岁", "25-29岁", "30-34岁", "35-39岁",
         "40-44岁", "45-49岁", "50-54岁", "55-59岁", "60-64岁", "65-69岁", "70-74岁",
         "75-79岁", "80-84岁", "85-89岁", "90-94岁", ">95岁"]

d = pd.read_csv(DALY, encoding="utf-8-sig")
rate = d[d.metric_name.astype(str).str.contains("率")]
num = d[d.metric_name.astype(str).str.contains("数量")]


def series(df, cause):
    s = df[(df.cause_id == cause) & (df.age_name.isin(BANDS))]
    return s.pivot_table(index="year", columns="age_name", values="val", aggfunc="sum").reindex(columns=BANDS)


R_H, R_N = series(rate, 1026), series(rate, 1028)
N_H = series(num, 1026)
pop23 = N_H.loc[2023] / R_H.loc[2023] * 1e5
r23H, r23N = R_H.loc[2023].copy(), R_N.loc[2023].copy()

w = pd.read_csv(WPP, encoding="utf-8-sig")
wp = w[w.gbd_name == "Global"].pivot_table(index="year", columns="age_band", values="population", aggfunc="sum").reindex(columns=BANDS)
popsc = {y: pop23 * (wp.loc[y] / wp.loc[2023]) for y in range(2023, 2036)}
popsc[2023] = pop23.copy()


def fits(R, y0, y1):
    sub = R.loc[y0:y1].astype(float)
    yrs = sub.index.values.astype(float)
    agg = sub.sum(axis=1)
    b_agg = np.polyfit(yrs, np.log(agg.values), 1)[0]
    bh = {}
    for b in BANDS:
        v = sub[b].values.astype(float)
        if (v <= 1e-12).all():
            bh[b] = 0.0
            continue
        if (v <= 1e-12).any():
            bh[b] = b_agg
            continue
        bh[b] = np.polyfit(yrs, np.log(v), 1)[0]
    return bh


WINDOWS = {"1990–2019": (1990, 2019), "2000–2019": (2000, 2019),
           "1990–2023": (1990, 2023), "2010–2023": (2010, 2023)}
FITN = {k: fits(R_N, *v) for k, v in WINDOWS.items()}


def ldt(y, rH, rN):
    H = float((rH * popsc[y] / 1e5).sum())
    N = float((rN * popsc[y] / 1e5).sum())
    return N / (H + N)


L_A = ldt(2035, r23H, r23N)
L = []
L.append("=== Table S2a: NAFLD fitted slope by band (log %/yr) ===")
L.append("%-9s %9s %9s %9s %9s" % ("band", "1990-2019", "2000-2019", "1990-2023", "2010-2023"))
for b in BANDS:
    L.append("%-9s %9.4f %9.4f %9.4f %9.4f" % (
        b, 100 * FITN["1990–2019"][b], 100 * FITN["2000–2019"][b],
        100 * FITN["1990–2023"][b], 100 * FITN["2010–2023"][b]))
L.append("")
L.append("=== Table S2b: NAFLD single-band marginal effect on 2035 LDTI (pp) ===")
L.append("%-9s %9s %9s %9s %9s" % ("band", "1990-2019", "2000-2019", "1990-2023", "2010-2023"))
tot = {k: 0.0 for k in WINDOWS}
for b in BANDS:
    rv = []
    for win in WINDOWS:
        bN = FITN[win][b]
        rN = r23N.copy()
        rN[b] = r23N[b] * np.exp(bN * 12)
        m = (ldt(2035, r23H, rN) - L_A) * 100
        rv.append(m)
        tot[win] += m
    L.append("%-9s %9.4f %9.4f %9.4f %9.4f" % (b, *rv))
L.append("%-9s %9.4f %9.4f %9.4f %9.4f" % (
    "sum", tot["1990–2019"], tot["2000–2019"], tot["1990–2023"], tot["2010–2023"]))
L.append("")
L.append("A (all frozen) 2035 LDTI = %.4f%%" % (100 * L_A))
neg2000 = 0.0
neg = []
for b in BANDS:
    bN = FITN["2000–2019"][b]
    rN = r23N.copy()
    rN[b] = r23N[b] * np.exp(bN * 12)
    m = (ldt(2035, r23H, rN) - L_A) * 100
    if m < 0:
        neg2000 += m
        neg.append((b, m))
L.append("2000-2019 negative band sum = %.4f pp" % neg2000)
for b, m in neg:
    L.append("   %-9s %+.4f" % (b, m))
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("ok")
