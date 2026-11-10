import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
# AG4: single script, single seed, 10,000 draws, rho=0.30, both year assumptions.
import itertools
from math import factorial
import numpy as np, pandas as pd
BAND = os.path.join(C.DATA, "daly_band_2023.csv")
OUT = os.path.join(C.OUT, "w23_mc10k_out.txt")
NDRAW = 10000
HB, NA = 1026, 1028
Y0, Y1 = 1990, 2023
AGES = [1,23,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235]
rng = np.random.default_rng(20261110)
d = pd.read_csv(BAND, encoding="utf-8-sig")
d["metric_name"]=d["metric_name"].replace({"数量":"Number","率":"Rate"})
d["age_id"]=d["age_id"].astype(int); d["cause_id"]=d["cause_id"].astype(int)
num=d[d.metric_id==1]; rat=d[d.metric_id==3]
years=sorted(num.year.unique()); ages=sorted(num.age_id.unique()); assert len(ages)==19
def cube(df):
    out={}
    for c in (HB,NA):
        for y in years:
            s=df[(df.cause_id==c)&(df.year==y)].set_index("age_id").reindex(AGES)
            assert s.val.notna().all(),(c,y)
            out[(c,y)]=s.val.values.astype(float)
    return out
N,R=cube(num),cube(rat)
SIG={}
for c in (HB,NA):
    for y in years:
        s=num[(num.cause_id==c)&(num.year==y)].set_index("age_id").reindex(AGES)
        v,lo,hi=s.val.values.astype(float),s.lower.values.astype(float),s.upper.values.astype(float)
        sg=np.zeros(19); ok=(v>0)&(lo>0)&(hi>0); sg[ok]=(np.log(hi[ok])-np.log(lo[ok]))/3.92
        SIG[(c,y)]=sg
POP={}
for y in (Y0,Y1):
    p=[]
    for i in range(19):
        cand=[N[(c,y)][i]/R[(c,y)][i]*1e5 for c in (HB,NA) if R[(c,y)][i]>0]
        p.append(float(np.mean(cand)) if cand else 0.0)
    POP[y]=np.array(p)
P={y:POP[y].sum() for y in (Y0,Y1)}; S={y:POP[y]/P[y] for y in (Y0,Y1)}
PL=[("HBV_rate",HB,"rate"),("HBV_age",HB,"age"),("NAFLD_rate",NA,"rate"),("NAFLD_age",NA,"age")]
def shapley_w(n):
    for i in range(n):
        others=[j for j in range(n) if j!=i]
        for k in range(n):
            for sub in itertools.combinations(others,k):
                yield i,set(sub),factorial(k)*factorial(n-k-1)/factorial(n)
def decomp(R0h,R1h,R0n,R1n):
    def rb(Rv,y): return float((Rv*S[y]).sum())
    h0,h1=rb(R0h,Y0),rb(R1h,Y1); n0,n1=rb(R0n,Y0),rb(R1n,Y1)
    delta={"HBV_rate":float(((S[Y0]+S[Y1])/2*(R1h-R0h)).sum()),
           "HBV_age":float(((R0h+R1h)/2*(S[Y1]-S[Y0])).sum()),
           "NAFLD_rate":float(((S[Y0]+S[Y1])/2*(R1n-R0n)).sum()),
           "NAFLD_age":float(((R0n+R1n)/2*(S[Y1]-S[Y0])).sum())}
    L=lambda h,n:n/(h+n); L0,L1=L(h0,n0),L(h1,n1)
    def val(idx):
        h,n=h0,n0
        for i in idx:
            nm,cause,kind=PL[i]
            if cause==HB: h+=delta[nm]
            else: n+=delta[nm]
        return L(h,n)
    phi={p[0]:0.0 for p in PL}
    for i,sub,w in shapley_w(4):
        phi[PL[i][0]]+=w*(val(sub|{i})-val(sub))
    return L0,L1,np.array([phi[k] for k,_,_ in PL])
L0,L1,phi=decomp(R[(HB,Y0)],R[(HB,Y1)],R[(NA,Y0)],R[(NA,Y1)])
A,Yn=19,len(years); i0,i1=years.index(Y0),years.index(Y1)
def run(rho,shared):
    Z={}
    for c in (HB,NA):
        z0=rng.standard_normal((NDRAW,Yn,1)); eps=rng.standard_normal((NDRAW,Yn,A))
        zz=np.sqrt(rho)*z0+np.sqrt(1-rho)*eps
        if shared: zz[:]=zz[:, [0], :]
        Z[c]=zz
    mc=np.zeros((NDRAW,4)); mcL=np.zeros((NDRAW,2))
    for k in range(NDRAW):
        rh={Y0:N[(HB,Y0)]*np.exp(SIG[(HB,Y0)]*Z[HB][k,i0,:])/POP[Y0]*1e5,
            Y1:N[(HB,Y1)]*np.exp(SIG[(HB,Y1)]*Z[HB][k,i1,:])/POP[Y1]*1e5}
        rn={Y0:N[(NA,Y0)]*np.exp(SIG[(NA,Y0)]*Z[NA][k,i0,:])/POP[Y0]*1e5,
            Y1:N[(NA,Y1)]*np.exp(SIG[(NA,Y1)]*Z[NA][k,i1,:])/POP[Y1]*1e5}
        a_,b_,ph=decomp(rh[Y0],rh[Y1],rn[Y0],rn[Y1]); mcL[k]=(a_,b_); mc[k]=ph
    return mc,mcL
def q(x): return float(np.percentile(x,2.5)),float(np.percentile(x,97.5))
out=[]
def W(s=""): out.append(str(s))
NMS=["HBV rate","HBV age","NAFLD rate","NAFLD age"]
W("AG4: NDRAW=%d  seed=20261110  rho=0.30" % NDRAW)
W("point LDTI %.5f -> %.5f  dL %+.4f pp"%(L0,L1,100*(L1-L0)))
W("point components: "+", ".join("%s %+.4f"%(n,100*phi[i]) for i,n in enumerate(NMS)))
for shared in (True,False):
    mc,mcL=run(0.30,shared)
    tag="year=shared" if shared else "year=indep"
    W("\n=== rho=0.30  %s ==="%tag)
    for i,n in enumerate(NMS):
        lo,hi=q(mc[:,i]); m=float(np.median(mc[:,i]))
        W("  %-12s point %+7.4f  UI [%+7.4f, %+7.4f]  %s"%(n,100*phi[i],100*lo,100*hi,"跨零" if lo<0<hi else "不跨零"))
    dLm=mcL[:,1]-mcL[:,0]; lo,hi=q(dLm)
    W("  dLDTI(pp)  point %+7.4f  UI [%+7.4f, %+7.4f]"%(100*(L1-L0),100*lo,100*hi))
    rel=mcL[:,1]/mcL[:,0]-1; rlo,rhi=q(rel)
    W("  LDTI rel   point %+7.1f%%  UI [%+7.1f, %+7.1f]%%"%(100*(L1/L0-1),100*rlo,100*rhi))
open(OUT,"w",encoding="utf-8").write("\n".join(out)); print("ag4 done")
