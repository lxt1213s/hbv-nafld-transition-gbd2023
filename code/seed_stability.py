"""Seed stability of the calibrated interval width (Supplementary Table S6).

For each of twenty random seeds, the Monte Carlo simulation of the
age-specific DALY counts is repeated (2000 draws per seed, rho = 0.30,
independent draws across years) and the width of the simulated 95%
interval of the age-aggregated total is divided by the width of the
published GBD all-ages interval.  Eight checkpoints are evaluated
(HBV and NAFLD, 1990, 2015, 2020 and 2023) and the median of the eight
ratios is reported per seed.  The analysis seed is 20261110.

Inputs  (data/, see README for the GBD Results Tool parameters):
    daly_band_2023.csv   DALY counts and bounds by cause, age band and year
    agg3c_2023.csv       all-ages DALY counts and bounds (age_id 22 = All ages)
Output:
    output/seed_stability_out.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BAND = os.path.join(C.DATA, "daly_band_2023.csv")
AGG = os.path.join(C.DATA, "agg3c_2023.csv")
DEST = os.path.join(C.OUT, "seed_stability_out.txt")

AGES = [1, 23, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
HB, NA = 1026, 1028
YRS = [1990, 2015, 2020, 2023]
SEEDS = [7, 101, 202, 303, 404, 505, 606, 707, 808, 909,
         11, 22, 33, 44, 55, 66, 77, 88, 99, 20261110]
RHO = 0.30
NSIM = 2000

for _p, _what in ((BAND, "band-level DALYs"), (AGG, "all-ages DALYs")):
    if not os.path.exists(_p):
        raise SystemExit(
            "missing input: %s\n"
            "  %s must be downloaded from the GBD Results Tool with the\n"
            "  parameters listed in README section 1 and placed in data/.\n"
            "  GBD exports are not redistributed with this repository." % (_p, _what))

b = pd.read_csv(BAND, encoding="utf-8-sig")
a = pd.read_csv(AGG, encoding="utf-8-sig")
for df in (b, a):
    df["metric_name"] = df["metric_name"].replace({"数量": "Number", "率": "Rate"})
    df["age_id"] = df["age_id"].astype(int)
    df["cause_id"] = df["cause_id"].astype(int)


def bands(cid):
    d = b[(b.cause_id == cid) & (b.age_id.isin(AGES)) & (b.metric_name == "Number")]
    v = d.pivot_table(index="age_id", columns="year", values="val").reindex(AGES)
    up = d.pivot_table(index="age_id", columns="year", values="upper").reindex(AGES)
    lo = d.pivot_table(index="age_id", columns="year", values="lower").reindex(AGES)
    sig = (np.log(up / lo) / 3.92).clip(0, None)
    return v, sig


def official_width(cid, yr):
    r = a[(a.cause_id == cid) & (a.age_id == 22) & (a.metric_name == "Number") & (a.year == yr)]
    return float(r.upper.iloc[0] - r.lower.iloc[0])


HV, HS = bands(HB)
NV, NS = bands(NA)
CHECK = [("HBV", y, HV, HS, HB) for y in YRS] + [("NAFLD", y, NV, NS, NA) for y in YRS]

lines = []


def w(s=""):
    lines.append(str(s))


w("=== Seed stability of the eight width ratios (rho = %.2f, %d draws, %d seeds) ==="
  % (RHO, NSIM, len(SEEDS)))
med_each = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    ratios = []
    for name, yr, V, S, cid in CHECK:
        base = V[yr].values.astype(float)
        sd = np.nan_to_num(S[yr].values.astype(float))
        ok = np.isfinite(base)
        base, sd = base[ok], sd[ok]
        z0 = rng.standard_normal(NSIM)
        eps = rng.standard_normal((NSIM, len(base)))
        z = np.sqrt(RHO) * z0[:, None] + np.sqrt(1 - RHO) * eps
        tot = (base[None, :] * np.exp(sd[None, :] * z)).sum(1)
        ratios.append((np.percentile(tot, 97.5) - np.percentile(tot, 2.5)) / official_width(cid, yr))
    med = float(np.median(ratios))
    med_each.append(med)
    w("seed %-9d median %.4f  min %.4f  max %.4f" % (seed, med, np.min(ratios), np.max(ratios)))

med_each = np.array(med_each)
w("seeds=%d  median-of-medians %.4f  min %.4f  max %.4f  sd %.4f"
  % (len(SEEDS), np.median(med_each), med_each.min(), med_each.max(), med_each.std(ddof=1)))
w("seeds above 1.00: %d/%d" % (int((med_each > 1).sum()), len(med_each)))

with open(DEST, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nwritten:", DEST)
