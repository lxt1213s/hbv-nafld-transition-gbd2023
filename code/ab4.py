import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""AB4: full 2x2 scenario contrast for ALL FOUR windows.
Cells: A (both off), HB_on (HBV trend, NAFLD frozen), NA_on (NAFLD trend, HBV frozen), B (both on).
Reports LDTI2035, delta-pp, single-cell marginals and two-cell-averaged marginals.
"""
import numpy as np, pandas as pd

DALY = os.path.join(C.DATA, "daly_band_2023.csv")
DEATH = os.path.join(C.DATA, "deaths_zh_1980_2023.csv")
WPP = os.path.join(C.DATA, "WPP2024_pop_GBDagebands_1990_2035.csv")
OUT = os.path.join(C.OUT, "ab4_out.txt")

BANDS = ["5岁以下","5-14岁","15-19岁","20-24岁","25-29岁","30-34岁","35-39岁",
         "40-44岁","45-49岁","50-54岁","55-59岁","60-64岁","65-69岁","70-74岁",
         "75-79岁","80-84岁","85-89岁","90-94岁",">95岁"]
HB, NA = 1026, 1028
L = []
def say(s=""):
    L.append(str(s))

d = pd.read_csv(DALY, encoding="utf-8-sig")
rate = d[d.metric_name.astype(str).str.contains("率")].copy()
num  = d[d.metric_name.astype(str).str.contains("数量")].copy()
def series(df, cause):
    s = df[(df.cause_id == cause) & (df.age_name.isin(BANDS))]
    return s.pivot_table(index="year", columns="age_name", values="val", aggfunc="sum").reindex(columns=BANDS)
R_H, R_N = series(rate, HB), series(rate, NA)
N_H, N_N = series(num, HB), series(num, NA)
pop23 = N_H.loc[2023] / R_H.loc[2023] * 1e5
H_obs, N_obs = N_H.loc[2023].sum(), N_N.loc[2023].sum()
L_obs = N_obs/(H_obs+N_obs)

w = pd.read_csv(WPP, encoding="utf-8-sig")
wp = w[w.gbd_name == "Global"].pivot_table(index="year", columns="age_band",
                                           values="population", aggfunc="sum").reindex(columns=BANDS)
YEARS = list(range(2024, 2036))
popsc = {y: pop23 * (wp.loc[y]/wp.loc[2023]) for y in YEARS}
popsc[2023] = pop23.copy()

WINDOWS = {"w1990_2019": (1990, 2019), "w2000_2019": (2000, 2019),
           "w1990_2023": (1990, 2023), "w2010_2023": (2010, 2023)}
def fits(R, y0, y1):
    bh, se = {}, {}
    sub = R.loc[y0:y1].astype(float); yrs = sub.index.values.astype(float)
    sxx = ((yrs-yrs.mean())**2).sum()
    agg = sub.sum(axis=1)
    b_agg = np.polyfit(yrs, np.log(agg.values), 1)[0]
    for b in BANDS:
        v = sub[b].values.astype(float)
        if (v <= 1e-12).all():
            bh[b], se[b] = 0.0, 0.0; continue
        if (v <= 1e-12).any():
            bh[b], se[b] = b_agg, 0.0; continue
        sl, ic = np.polyfit(yrs, np.log(v), 1)
        res = np.log(v)-(ic+sl*yrs); s2 = (res**2).sum()/(len(yrs)-2)
        bh[b], se[b] = sl, np.sqrt(s2/sxx)
    return bh, se
FIT = {k: fits(R_H, *v) for k, v in WINDOWS.items()}
FITN = {k: fits(R_N, *v) for k, v in WINDOWS.items()}

r23H, r23N = R_H.loc[2023].copy(), R_N.loc[2023].copy()
def ldt(y, rH, rN):
    Hh = float((rH*popsc[y]/1e5).sum()); Nn = float((rN*popsc[y]/1e5).sum())
    return Nn/(Hh+Nn), Hh, Nn

def cell(kind, win):
    bH = FIT[win][0] if win else None
    bN = FITN[win][0] if win else None
    if kind == "A":
        rH, rN = r23H.copy(), r23N.copy()
    elif kind == "hb_on":
        rH = pd.Series({b: r23H[b]*np.exp(bH[b]*(2035-2023)) for b in BANDS})
        rN = r23N.copy()
    elif kind == "na_on":
        rH = r23H.copy()
        rN = pd.Series({b: r23N[b]*np.exp(bN[b]*(2035-2023)) for b in BANDS})
    elif kind == "B":
        rH = pd.Series({b: r23H[b]*np.exp(bH[b]*(2035-2023)) for b in BANDS})
        rN = pd.Series({b: r23N[b]*np.exp(bN[b]*(2035-2023)) for b in BANDS})
    l = ldt(2035, rH, rN)[0]
    return l, (l - L_obs)*100

say("=== AB4  2x2 scenario contrast, ALL FOUR windows (LDTI 2035, delta pp vs 2023=%.4f) ===" % (L_obs*100))
for k in WINDOWS:
    cells = {}
    for label, kind in [("A  (HBV off, NAFLD off)", "A"), ("HB_on (HBV on , NAFLD off)", "hb_on"),
                        ("NA_on (HBV off, NAFLD on )", "na_on"), ("B  (HBV on , NAFLD on )", "B")]:
        cells[label] = cell(kind, k)
    A  = cells["A  (HBV off, NAFLD off)"][1]
    HBon_naOff = cells["HB_on (HBV on , NAFLD off)"][1]
    NAon_hbOff = cells["NA_on (HBV off, NAFLD on )"][1]
    B  = cells["B  (HBV on , NAFLD on )"][1]
    # two-cell-averaged marginals
    hb_marg = 0.5*((HBon_naOff - A) + (B - NAon_hbOff))
    na_marg = 0.5*((NAon_hbOff - A) + (B - HBon_naOff))
    inter = B - HBon_naOff - NAon_hbOff + A
    say("")
    say("window %s" % k)
    say("  %-30s  LDTI2035   d_pp" % "cell")
    for label in ("A  (HBV off, NAFLD off)", "HB_on (HBV on , NAFLD off)",
                  "NA_on (HBV off, NAFLD on )", "B  (HBV on , NAFLD on )"):
        say("  %-30s  %.4f   %+7.2f" % (label, cells[label][0]*100, cells[label][1]))
    say("  interaction = %+.2f pp" % inter)
    say("  single-cell marginals:  HBV = %+.2f pp ; NAFLD = %+.2f pp" % (HBon_naOff-A, NAon_hbOff-A))
    say("  two-cell-averaged:      HBV = %+.2f pp ; NAFLD = %+.2f pp" % (hb_marg, na_marg))

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("ok")
