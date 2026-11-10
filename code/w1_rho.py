import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
BAND = os.path.join(C.DATA, "daly_band_2023.csv")
AGG  = os.path.join(C.DATA, "agg3c_2023.csv")
AGES = [1,23,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235]
HB, NA = 1026, 1028
YRS = [1990,2015,2020,2023]
NSIM = 2000
rng = np.random.default_rng(20261110)
b = pd.read_csv(BAND, encoding="utf-8-sig"); a = pd.read_csv(AGG, encoding="utf-8-sig")
b["metric_name"] = b["metric_name"].replace({"数量":"Number","率":"Rate"})
a["metric_name"] = a["metric_name"].replace({"数量":"Number","率":"Rate"})
b["age_id"]=b["age_id"].astype(int); a["age_id"]=a["age_id"].astype(int)
b["cause_id"]=b["cause_id"].astype(int); a["cause_id"]=a["cause_id"].astype(int)
def bands(cid):
    d = b[(b.cause_id==cid)&(b.age_id.isin(AGES))&(b.metric_name=="Number")]
    v = d.pivot_table(index="age_id",columns="year",values="val").reindex(AGES)
    up = d.pivot_table(index="age_id",columns="year",values="upper").reindex(AGES)
    lo = d.pivot_table(index="age_id",columns="year",values="lower").reindex(AGES)
    sig = (np.log(up/lo)/3.92).clip(0,None)
    return v, sig
HV,HS = bands(HB); NV,NS = bands(NA)
def offw(cid,yr):
    r = a[(a.cause_id==cid)&(a.age_id==22)&(a.metric_name=="Number")&(a.year==yr)]
    return float(r.upper.iloc[0]-r.lower.iloc[0])
def simw(V,S,rho,yr):
    base=V[yr].values; sd=S[yr].values
    ok=~np.isnan(base); base=base[ok]; sd=np.nan_to_num(sd[ok])
    z0=rng.standard_normal(NSIM); eps=rng.standard_normal((NSIM,len(base)))
    z=np.sqrt(rho)*z0[:,None]+np.sqrt(1-rho)*eps
    tot=(base[None,:]*np.exp(sd[None,:]*z)).sum(1)
    return np.percentile(tot,97.5)-np.percentile(tot,2.5)
out=[]
def W(s=""): out.append(str(s))
W("=== rho=0.30 八检查点宽度比（本次重算）===")
allr=[]
for cid,V,S,nm in ((HB,HV,HS,"HBV"),(NA,NV,NS,"NAFLD")):
    for yr in YRS:
        mw=simw(V,S,0.30,yr); ow=offw(cid,yr); r=mw/ow
        allr.append((nm,yr,r)); W("  %-6s %d ratio=%.3f"%(nm,yr,r))
vals=sorted(x[2] for x in allr)
W("sorted: "+", ".join("%.3f"%v for v in vals))
W("median=%.3f  range %.3f-%.3f"%(np.median(vals),min(vals),max(vals)))
W("\n=== 分病因最优 rho（网格 0.00-0.50 步 0.02）===")
for cid,V,S,nm in ((HB,HV,HS,"HBV"),(NA,NV,NS,"NAFLD")):
    best=None
    for rho in np.arange(0,0.52,0.02):
        rs=[simw(V,S,rho,yr)/offw(cid,yr) for yr in YRS]
        med=float(np.median(rs))
        if best is None or abs(med-1)<abs(best[1]-1): best=(round(float(rho),2),med)
    W("  %-6s optimal rho=%.2f (median ratio %.3f)"%(nm,best[0],best[1]))
    for rho in (0.18,0.20,0.24,0.28,0.30,0.34):
        rs=[simw(V,S,rho,yr)/offw(cid,yr) for yr in YRS]
        W("     rho=%.2f  ratios %s  median %.3f"%(rho,", ".join("%.3f"%round(x,3) for x in rs),float(np.median(rs))))
open(os.path.join(C.OUT, "w1_rho_out.txt"),"w",encoding="utf-8").write("\n".join(out))
print("w1 done")
