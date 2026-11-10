import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
"""GBD 2021 vs GBD 2023 双轮 LDTI 分解对比（全球）。"""
import itertools
from math import factorial

import numpy as np
import pandas as pd

F21 = os.path.join(C.DATA, "gbd2021_global_master_1990_2021.csv")
F23 = os.path.join(C.DATA, "daly_band_2023.csv")
OUTF = os.path.join(C.OUT, "dual_round_out.txt")

HB21 = "Total burden related to hepatitis B"
NA21 = "Total burden related to Non-alcoholic fatty liver disease (NAFLD)"
HB23 = "乙肝相关疾病负担"
NA23 = "NASH相关疾病负担"

AGES21 = (["<5 years", "5-14 years", "15-19 years"]
          + [f"{a}-{a+4} years" for a in range(20, 95, 5)] + ["95+ years"])
AGES23 = (["5岁以下", "5-14岁", "15-19岁"]
          + [f"{a}-{a+4}岁" for a in range(20, 95, 5)] + [">95岁"])


def shapley(n):
    allidx = list(range(n))
    for i in allidx:
        others = [j for j in allidx if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                yield i, set(S), factorial(k) * factorial(n - k - 1) / factorial(n)


def load21(y0, y1):
    d = pd.read_csv(F21, encoding="utf-8-sig")
    d = d[d.measure_name.str.startswith("DALYs")]

    def ser(c, m, y):
        s = d[(d.cause_name == c) & (d.metric_name == m) & (d.year == y)
              & (d.age_name.isin(AGES21))]
        assert s.age_name.is_unique
        return s.set_index("age_name").val.reindex(AGES21).astype(float).values
    N = {(c, y): ser(c, "Number", y) for c in (HB21, NA21) for y in (y0, y1)}
    R = {(c, y): ser(c, "Rate", y) for c in (HB21, NA21) for y in (y0, y1)}
    for c in (HB21, NA21):
        for y in (y0, y1):
            allages = d[(d.cause_name == c) & (d.metric_name == "Number")
                        & (d.year == y) & (d.age_name == "All ages")].val.iloc[0]
            assert abs(N[(c, y)].sum() - allages) < 1
    return N, R, AGES21, y0, y1


def load23(y0, y1):
    d = pd.read_csv(F23, encoding="utf-8")
    def ser(c, m, y):
        s = d[(d.cause_name == c) & (d.metric_name == m) & (d.year == y)
              & (d.age_name.isin(AGES23))]
        assert s.age_name.is_unique and len(s) == len(AGES23), (c, m, y, len(s))
        v = s.set_index("age_name").val.reindex(AGES23).astype(float).values
        assert not np.isnan(v).any(), (c, m, y)
        return v
    N = {(c, y): ser(c, "数量", y) for c in (HB23, NA23) for y in (y0, y1)}
    R = {(c, y): ser(c, "率", y) for c in (HB23, NA23) for y in (y0, y1)}
    return N, R, AGES23, y0, y1


def run(N, R, ages, y0, y1, hb, na, out):
    pop = {}
    for y in (y0, y1):
        p = []
        for i in range(len(ages)):
            cand = [N[(c, y)][i] / R[(c, y)][i] * 1e5
                    for c in (hb, na) if R[(c, y)][i] > 0]
            p.append(float(np.mean(cand)) if cand else 0.0)
        pop[y] = np.array(p)
    P = {y: pop[y].sum() for y in (y0, y1)}
    s = {y: pop[y] / P[y] for y in (y0, y1)}
    rb = {(c, y): float((R[(c, y)] * s[y]).sum()) for c in (hb, na) for y in (y0, y1)}
    comp = {}
    for c in (hb, na):
        comp[(c, "rate")] = float(((s[y0] + s[y1]) / 2 * (R[(c, y1)] - R[(c, y0)])).sum())
        comp[(c, "age")] = float(((R[(c, y0)] + R[(c, y1)]) / 2 * (s[y1] - s[y0])).sum())
        assert abs(comp[(c, "rate")] + comp[(c, "age")] - (rb[(c, y1)] - rb[(c, y0)])) < 1e-6
    h0, n0, h1, n1 = rb[(hb, y0)], rb[(na, y0)], rb[(hb, y1)], rb[(na, y1)]
    L = lambda h, n: n / (h + n)
    L0, L1 = L(h0, n0), L(h1, n1)
    players = [("HBV rate", hb, "rate"), ("HBV age", hb, "age"),
               ("NAFLD rate", na, "rate"), ("NAFLD age", na, "age")]
    delta = {p[0]: comp[(p[1], p[2])] for p in players}

    def val(idx):
        h, n = h0, n0
        for i in idx:
            if players[i][1] == hb:
                h += delta[players[i][0]]
            else:
                n += delta[players[i][0]]
        return L(h, n)
    phi = {p[0]: 0.0 for p in players}
    for i, S, w in shapley(4):
        phi[players[i][0]] += w * (val(S | {i}) - val(S))

    resid = sum(phi.values()) - (L1 - L0)
    assert abs(resid) < 1e-9, "four-component sum off by %+.3e (proportion scale)" % resid

    out.append("\n===== %s -> %s =====" % (y0, y1))
    out.append("population (implied): %.1fM -> %.1fM" % (P[y0] / 1e6, P[y1] / 1e6))
    out.append("HBV  DALYs %.3fM -> %.3fM ; crude rate %.2f -> %.2f (%+.1f%%)"
               % (N[(hb, y0)].sum() / 1e6, N[(hb, y1)].sum() / 1e6, h0, h1, 100 * (h1 - h0) / h0))
    out.append("NASH DALYs %.3fM -> %.3fM ; crude rate %.2f -> %.2f (%+.1f%%)"
               % (N[(na, y0)].sum() / 1e6, N[(na, y1)].sum() / 1e6, n0, n1, 100 * (n1 - n0) / n0))
    out.append("LDTI %.5f -> %.5f  (delta %+.5f, rel %+.1f%%)" % (L0, L1, L1 - L0, 100 * (L1 / L0 - 1)))
    out.append("   residual (proportion scale) %+.3e" % resid)
    for nm, v in phi.items():
        out.append("   %-11s %+7.3f pp  (%+7.1f%%)" % (nm, v * 100, 100 * v / (L1 - L0)))
    rate = phi["HBV rate"] + phi["NAFLD rate"]
    age = phi["HBV age"] + phi["NAFLD age"]
    out.append("   rate total  %+7.3f pp (%+.1f%%)" % (rate * 100, 100 * rate / (L1 - L0)))
    out.append("   age  total  %+7.3f pp (%+.1f%%)" % (age * 100, 100 * age / (L1 - L0)))
    out.append("   rate effect: HBV %+.2f | NAFLD %+.2f per100k ; structure: HBV %+.2f | NAFLD %+.2f"
               % (comp[(hb, "rate")], comp[(na, "rate")], comp[(hb, "age")], comp[(na, "age")]))
    return dict(L0=L0, L1=L1, phi=phi, h0=h0, h1=h1, n0=n0, n1=n1, P0=P[y0], P1=P[y1])


out = []
REC = []


def rec(round_, period, r):
    row = dict(round=round_, period=period,
               LDTI_start=r["L0"], LDTI_end=r["L1"],
               LDTI_delta=r["L1"] - r["L0"],
               LDTI_rel_pct=100 * (r["L1"] / r["L0"] - 1),
               HBV_rate_pp=100 * r["phi"]["HBV rate"],
               HBV_age_pp=100 * r["phi"]["HBV age"],
               NAFLD_rate_pp=100 * r["phi"]["NAFLD rate"],
               NAFLD_age_pp=100 * r["phi"]["NAFLD age"],
               rate_total_pp=100 * (r["phi"]["HBV rate"] + r["phi"]["NAFLD rate"]),
               age_total_pp=100 * (r["phi"]["HBV age"] + r["phi"]["NAFLD age"]),
               HBV_crude0=r["h0"], HBV_crude1=r["h1"],
               NAFLD_crude0=r["n0"], NAFLD_crude1=r["n1"],
               pop0_M=r["P0"] / 1e6, pop1_M=r["P1"] / 1e6)
    REC.append(row)


N, R, A, a0, a1 = load21(1990, 2021)
r21 = run(N, R, A, 1990, 2021, HB21, NA21, out)
rec("GBD2021", "1990-2021", r21)
N, R, A, a0, a1 = load23(1990, 2021)
r2321 = run(N, R, A, 1990, 2021, HB23, NA23, out)
rec("GBD2023", "1990-2021", r2321)
N, R, A, a0, a1 = load23(1990, 2023)
r2323 = run(N, R, A, 1990, 2023, HB23, NA23, out)
rec("GBD2023", "1990-2023", r2323)
pd.DataFrame(REC).round(5).to_csv(os.path.join(C.DERIVED, "dual_round_results.csv"),
                                  index=False, encoding="utf-8-sig")

out.append("\n===== 轮次一致性（同 1990->2021，两轮对比） =====")
out.append("LDTI 2021: GBD2021 round %.4f | GBD2023 round %.4f | diff %+.4f" %
           (r21["L1"], r2321["L1"], r2321["L1"] - r21["L1"]))
for k in ["HBV rate", "HBV age", "NAFLD rate", "NAFLD age"]:
    out.append("   %-11s %+7.3f pp  vs  %+7.3f pp  (diff %+.3f)" %
               (k, r21["phi"][k] * 100, r2321["phi"][k] * 100, (r2321["phi"][k] - r21["phi"][k]) * 100))
out.append("HBV crude rate 1990: %.2f vs %.2f ; 2021: %.2f vs %.2f" %
           (r21["h0"], r2321["h0"], r21["h1"], r2321["h1"]))
out.append("NASH crude rate 1990: %.2f vs %.2f ; 2021: %.2f vs %.2f" %
           (r21["n0"], r2321["n0"], r21["n1"], r2321["n1"]))

open(OUTF, "w", encoding="utf-8").write("\n".join(out))
print("ok")
