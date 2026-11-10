import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

import pandas as pd, numpy as np, io

AL = os.path.join(C.OUT, "wpp_aligned.csv")
CSV_OUT = os.path.join(C.DATA, "WPP2024_pop_GBDagebands_1990_2035.csv")
DALY = os.path.join(C.DATA, "daly_band_2023.csv")

g = pd.read_csv(AL, encoding="utf-8-sig")
g = g.rename(columns={"Time": "year", "age": "age_band", "PopTotal": "population"})
g["gbd_name"] = "Global"
g = g[["gbd_name", "year", "age_band", "population"]].sort_values(["year", "age_band"])
g.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

# reconciliation: GBD-implied population (DALYs Number / Rate * 1e5) vs WPP 2024 medium, Global 1990-2023
d = pd.read_csv(DALY, encoding="utf-8-sig")
d = d[(d.location_name == "全球") & (d.cause_name.str.contains("乙肝"))]
d = d[(d.metric_name == "数量") | (d.metric_name == "率")]
pos = d.pivot_table(index=["age_name", "year"], columns="metric_name", values="val").reset_index()
pos = pos[(pos["率"] > 0) & (pos["数量"] > 0)]
pos["pop"] = pos["数量"] / pos["率"] * 1e5
gbdpop = pos.groupby("year")["pop"].sum()
wpppop = g.groupby("year")["population"].sum()
yrs = [y for y in range(1990, 2024) if y in gbdpop.index and y in wpppop.index]
rel = pd.Series({y: 100 * (gbdpop[y] - wpppop[y]) / wpppop[y] for y in yrs})

L = []
L.append("rows=%d  years=%d..%d  bands=%d" % (len(g), g.year.min(), g.year.max(), g.age_band.nunique()))
L.append("")
L.append("Global population reconciliation (GBD implied vs WPP2024 medium), billions")
L.append("%6s %10s %10s %8s" % ("year", "GBD", "WPP", "diff%"))
for y in [1990, 2000, 2010, 2021, 2023]:
    if y in rel.index:
        L.append("%6d %10.3f %10.3f %8.2f" % (y, gbdpop[y] / 1e9, wpppop[y] / 1e9, rel[y]))
L.append("")
L.append("max |diff| = %.2f%%   median |diff| = %.2f%%   (gate: >2%% triggers calibration)"
         % (rel.abs().max(), rel.abs().median()))
L.append("2035 WPP projection = %.3f bn (GBD stops at 2023)" % (wpppop.get(2035, float("nan")) / 1e9))
io.open(os.path.join(C.OUT, "wpp_recon.txt"), "w", encoding="utf-8").write("\n".join(L))
print("ok")
