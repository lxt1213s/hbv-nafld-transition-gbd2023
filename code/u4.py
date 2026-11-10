import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""U1-U4: hygiene, sum check, rho scan on DALYs, ASR-based LDTI, three-cause decomposition."""
import pandas as pd, numpy as np

BAND = os.path.join(C.DATA, "daly_band_2023.csv")   # HBV+NAFLD, 19 bands
HCVB = os.path.join(C.DATA, "hcv_band_2023.csv")  # HCV, 19 bands
AGG  = os.path.join(C.DATA, "agg3c_2023.csv")  # 3 causes, All ages + ASR

AGES = [1, 23, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
HB, HC, NA = 1026, 1027, 1028
Y0, Y1 = 1990, 2023
rng = np.random.default_rng(20261110)
NSIM = 2000
out = []
def W(s=""): out.append(str(s))

b = pd.read_csv(BAND, encoding="utf-8-sig")
h = pd.read_csv(HCVB, encoding="utf-8-sig")
a = pd.read_csv(AGG, encoding="utf-8-sig")
for d in (b, h, a):
    d["metric_name"] = d["metric_name"].replace({"数量": "Number", "率": "Rate"})
    d["age_id"] = d["age_id"].astype(int)
    d["cause_id"] = d["cause_id"].astype(int)

# ---------- U1 summary ----------
W("=== U1 hygiene (already dumped; key figures) ===")
W("307-row file (3 causes, All ages + ASR): rows=%d  per-year=%d" % (len(a), len(a) // a.year.nunique()))
W("1292-row file (HCV bands): rows=%d  per-year=%d  ages=%d" % (len(h), len(h) // h.year.nunique(), h.age_id.nunique()))
W("band file HBV+NAFLD: rows=%d  per-year=%d  ages=%d" % (len(b), len(b) // b.year.nunique(), b.age_id.nunique()))
for nm, d in (("307", a), ("1292", h), ("bands", b)):
    key = [c for c in ["age_id", "cause_id", "metric_name", "year"] if c in d.columns]
    W("  %-5s dups=%d  NaN=%d  lower0&val>0=%d" % (nm, d.duplicated(subset=key).sum(),
      int(d[["val", "upper", "lower"]].isna().sum().sum()), int(((d.lower == 0) & (d.val > 0)).sum())))

# ---------- U3 sum check ----------
W("\n=== U3 sum check: HCV 19 bands vs HCV All ages ===")
hb = h[h.age_id.isin(AGES) & (h.metric_name == "Number")].groupby("year").val.sum()
ha = a[(a.cause_id == HC) & (a.age_id == 22) & (a.metric_name == "Number")].set_index("year").val
j = pd.concat([hb.rename("bands"), ha.rename("all")], axis=1).dropna()
j["diff"] = j.bands - j["all"]
j["pct"] = 100 * j["diff"] / j["all"]
W("pairs=%d  max|diff|=%.3f  max|pct|=%.4f%%" % (len(j), j["diff"].abs().max(), j["pct"].abs().max()))
W(j.loc[[1990, 2000, 2010, 2021, 2023]].to_string())

# ---------- U4a ASR-based LDTI ----------
W("\n=== U4a ASR-based LDTI (Rate / Age-standardized) ===")
asr = a[(a.age_id == 27) & (a.metric_name == "Rate")].pivot_table(index="year", columns="cause_id", values="val")
asr.columns = ["HBV", "HCV", "NAFLD"]
asr["LDTI2_ASR"] = asr.NAFLD / (asr.HBV + asr.NAFLD)
asr["share3_ASR"] = asr.NAFLD / (asr.HBV + asr.HCV + asr.NAFLD)
W(asr.loc[[1990, 2015, 2020, 2021, 2023]].round(4).to_string())
W("delta LDTI_ASR 1990->2023 = %+.4f ; rel = %+.1f%%" %
  (asr.LDTI2_ASR[2023] - asr.LDTI2_ASR[1990], 100 * (asr.LDTI2_ASR[2023] / asr.LDTI2_ASR[1990] - 1)))

W("\n=== U4a-bis three-cause share from All-ages counts ===")
cnt = a[(a.age_id == 22) & (a.metric_name == "Number")].pivot_table(index="year", columns="cause_id", values="val")
cnt.columns = ["HBV", "HCV", "NAFLD"]
cnt["share3"] = cnt.NAFLD / (cnt.HBV + cnt.HCV + cnt.NAFLD)
cnt["share2"] = cnt.NAFLD / (cnt.HBV + cnt.NAFLD)
W(cnt.loc[[1990, 2015, 2020, 2021, 2023]].assign(share3=lambda x: (x.NAFLD/(x.HBV+x.HCV+x.NAFLD)).round(4), share2=lambda x: (x.NAFLD/(x.HBV+x.NAFLD)).round(4)).to_string())

# ---------- helper: band table ----------
def bands(cause_df, cause_id):
    d = cause_df[(cause_df.cause_id == cause_id) & (cause_df.age_id.isin(AGES)) & (cause_df.metric_name == "Number")]
    v = d.pivot_table(index="age_id", columns="year", values="val").reindex(AGES)
    up = d.pivot_table(index="age_id", columns="year", values="upper").reindex(AGES)
    lo = d.pivot_table(index="age_id", columns="year", values="lower").reindex(AGES)
    sig = (np.log(up / lo) / 3.92).clip(0, None)
    return v, sig

b["cause_id"] = b["cause_id"].astype(int)
h["cause_id"] = h["cause_id"].astype(int)
HV, HS = bands(b, HB)
NV, NS = bands(b, NA)
CV, CS = bands(h, HC)

def official_width(cid, yr):
    r = a[(a.cause_id == cid) & (a.age_id == 22) & (a.metric_name == "Number") & (a.year == yr)]
    return float(r.upper.iloc[0] - r.lower.iloc[0]), float(r.val.iloc[0])

# ---------- U2 rho scan on DALYs ----------
W("\n=== U2 rho scan on DALYs (official All-ages widths from the 3-cause export) ===")
CHECK = [(HB, HV, HS, "HBV"), (NA, NV, NS, "NAFLD")]
YRS = [1990, 2015, 2020, 2023]

def sim_width(V, S, rho, yr):
    base = V[yr].values
    sd = S[yr].values
    ok = ~np.isnan(base)
    base, sd = base[ok], np.nan_to_num(sd[ok])
    z0 = rng.standard_normal(NSIM)
    eps = rng.standard_normal((NSIM, len(base)))
    z = np.sqrt(rho) * z0[:, None] + np.sqrt(1 - rho) * eps
    draws = base[None, :] * np.exp(sd[None, :] * z)
    tot = draws.sum(axis=1)
    return np.percentile(tot, 97.5) - np.percentile(tot, 2.5)

rows = []
for rho in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.0]:
    ratios = []
    for cid, V, S, nm in CHECK:
        for yr in YRS:
            ow, ov = official_width(cid, yr)
            mw = sim_width(V, S, rho, yr)
            if ow > 0:
                ratios.append(mw / ow)
    rows.append((rho, np.median(ratios), np.min(ratios), np.max(ratios)))
    W("rho=%.2f  median ratio=%.3f  range %.3f-%.3f" % (rho, np.median(ratios), np.min(ratios), np.max(ratios)))

sel = min(rows, key=lambda r: abs(np.median([r[1]]) - 1.0))[0]
W("closest-to-1 rho on this grid = %.2f (median ratio %.3f)" % (sel, [r[1] for r in rows if r[0] == sel][0]))

W("\nper-checkpoint detail at rho=0.30 and rho=0.15:")
for rho in (0.15, 0.30):
    W("  rho=%.2f" % rho)
    for cid, V, S, nm in CHECK:
        for yr in YRS:
            ow, ov = official_width(cid, yr)
            mw = sim_width(V, S, rho, yr)
            W("    %-6s %d  official=%10.0f  mc=%10.0f  ratio=%.3f" % (nm, yr, ow, mw, mw / ow))

# ---------- U4b three-cause decomposition ----------
W("\n=== U4b three-cause decomposition (sensitivity; directions only) ===")
def crude(V):
    return {y: np.nansum(V[y].values) for y in V.columns}
# population share from HBV implied population
def implied_pop(V, rate_df, yr):
    return np.nansum((V[yr] / rate_df[yr] * 1e5).values)

HVr = b[(b.cause_id == HB) & (b.age_id.isin(AGES)) & (b.metric_name == "Rate")].pivot_table(index="age_id", columns="year", values="val").reindex(AGES)
NVr = b[(b.cause_id == NA) & (b.age_id.isin(AGES)) & (b.metric_name == "Rate")].pivot_table(index="age_id", columns="year", values="val").reindex(AGES)
CVr = h[(h.cause_id == HC) & (h.age_id.isin(AGES)) & (h.metric_name == "Rate")].pivot_table(index="age_id", columns="year", values="val").reindex(AGES)

# population is ONE vector; derive it from HBV, whose 19 bands all have positive rates
pop = {}
for y in (Y0, Y1):
    v = HV[y].values.astype(float)
    r = HVr[y].values.astype(float)
    ok = (~np.isnan(v)) & (~np.isnan(r)) & (r > 0)
    assert ok.sum() == 19, "HBV band coverage incomplete at %d" % y
    pop[y] = v[ok] / r[ok] * 1e5
W("\nimplied population from HBV: %.1fM (1990) -> %.1fM (2023)" % (pop[Y0].sum() / 1e6, pop[Y1].sum() / 1e6))
for nm, R in (("HCV", CVr), ("NAFLD", NVr)):
    chk = []
    for y in (Y0, Y1):
        v = (CV[y].values if nm == "HCV" else NV[y].values).astype(float)
        r = R[y].values.astype(float)
        m = (~np.isnan(v)) & (~np.isnan(r)) & (r > 0)
        chk.append(np.nanmax(np.abs(v[m] / r[m] * 1e5 / pop[y][m] - 1)))
    W("  implied-population consistency %s: max rel diff %.2e" % (nm, max(chk)))

def parts(V, R, y0, y1):
    s0 = pop[y0] / pop[y0].sum(); s1 = pop[y1] / pop[y1].sum()
    r0 = V[y0].values / pop[y0] * 1e5; r1 = V[y1].values / pop[y1] * 1e5
    rate = np.nansum((s0 + s1) / 2 * (r1 - r0))
    stru = np.nansum((r0 + r1) / 2 * (s1 - s0))
    return rate, stru

from math import factorial
def shapley_frac(players, f, n):
    idx = list(range(n)); tot = 0.0
    for i in idx:
        for k in range(n):
            rest = [x for x in idx if x != i]
            for S in __import__("itertools").combinations(rest, k):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                tot += w * (f({**{j: 0 for j in idx}, **{j: players[j] for j in S}, i: players[i]})
                            - f({**{j: 0 for j in idx}, **{j: players[j] for j in S}}))
    return tot

def make_L(h0, c0, n0):
    def L(p):
        hh = h0 + p[0] + p[1]; cc = c0 + p[2] + p[3]; nn = n0 + p[4] + p[5]
        return nn / (hh + cc + nn)
    return L

rh, sh = parts(HV, HVr, Y0, Y1)
rc, sc = parts(CV, CVr, Y0, Y1)
rn, sn = parts(NV, NVr, Y0, Y1)
# crude rate = total counts / total population (NOT the sum of age-specific rates)
def crude_rate(V, y):
    return 1e5 * np.nansum(V[y].values.astype(float)) / pop[y].sum()
h0, c0, n0 = crude_rate(HV, Y0), crude_rate(CV, Y0), crude_rate(NV, Y0)
L = make_L(h0, c0, n0)
players = [rh, sh, rc, sc, rn, sn]
names = ["HBV rate", "HBV age", "HCV rate", "HCV age", "NAFLD rate", "NAFLD age"]
vals = [shapley_frac(players, L, 6) for _ in range(1)]
# shapley_frac returns sum over i only for i=0; do it per player instead
def shap_i(i):
    return shapley_frac_sep(i)
def shapley_frac_sep(i):
    n = 6; tot = 0.0
    base = {j: 0.0 for j in range(n)}
    for k in range(n):
        for S in __import__("itertools").combinations([j for j in range(n) if j != i], k):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            a = dict(base); b = dict(base)
            for j in S: a[j] = players[j]; b[j] = players[j]
            b[i] = players[i]
            tot += w * (L(b) - L(a))
    return tot
phi = [shap_i(i) for i in range(6)]
L0 = L({j: 0.0 for j in range(6)})
L1 = L({j: players[j] for j in range(6)})
W("L0=%.5f  L1=%.5f  sum(phi)=%.5f  resid=%.3e" % (L0, L1, sum(phi), sum(phi) - (L1 - L0)))
for nm, v in zip(names, phi):
    W("  %-11s %+8.4f pp  (%5.1f%%)" % (nm, 100 * v, 100 * v / (L1 - L0)))
W("HBV rate + HCV rate + NAFLD rate = %+.4f pp" % (100 * (phi[0] + phi[2] + phi[4])))
W("all ageing = %+.4f pp" % (100 * (phi[1] + phi[3] + phi[5])))

W("\n=== crude rates (per 100k, 19 bands) ===")
for nm, V, P in (("HBV", HV, HVr), ("HCV", CV, CVr), ("NAFLD", NV, NVr)):
    r0 = crude_rate(V, Y0); r1 = crude_rate(V, Y1)
    W("  %-6s %.1f -> %.1f  (%+.1f%%)" % (nm, r0, r1, 100 * (r1 / r0 - 1)))

open(os.path.join(C.OUT, "u4_out.txt"), "w", encoding="utf-8").write("\n".join(out))
print("written")
