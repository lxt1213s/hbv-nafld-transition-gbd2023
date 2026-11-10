import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

import pandas as pd, numpy as np

P = os.path.join(C.DATA, "WPP2024_pop5.gz")
OUT = os.path.join(C.OUT, "wpp_aligned.csv")

# map WPP 5-year age groups to the 19 GBD age bands (Global only)
ACTS = [("0-4", "5岁以下"), ("5-9", "5-14岁"), ("10-14", "5-14岁")]
for a in range(15, 95, 5):
    ACTS.append(("%d-%d" % (a, a + 4), "%d-%d岁" % (a, a + 4)))
ACTS.append(("95-99", ">95岁"))
ACTS.append(("100+", ">95岁"))
MAP = dict(ACTS)

df = pd.read_csv(P, compression="gzip", encoding="utf-8-sig",
                 usecols=["Location", "LocTypeName", "Time", "AgeGrp", "PopTotal"],
                 low_memory=False)
df = df[(df["Time"] >= 1990) & (df["Time"] <= 2035)]
df = df[df["LocTypeName"] == "World"].copy()  # Global only
df["age"] = df["AgeGrp"].map(MAP)
assert df["age"].notna().all(), df.loc[df["age"].isna(), "AgeGrp"].unique()
df["PopTotal"] = pd.to_numeric(df["PopTotal"], errors="coerce") * 1000.0

g = (df.groupby(["Time", "age"], dropna=False)["PopTotal"].sum().reset_index())
order = ["5岁以下", "5-14岁"] + ["%d-%d岁" % (a, a + 4) for a in range(15, 95, 5)] + [">95岁"]
g["ord"] = g["age"].map({a: i for i, a in enumerate(order)})
g = g.sort_values(["Time", "ord"])
g.to_csv(OUT, index=False, encoding="utf-8-sig")
print("ok")
