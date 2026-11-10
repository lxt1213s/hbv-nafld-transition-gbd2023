import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
# AF5: s sensitivity re-applied on the YLL-split baseline so s=1 == main value.
import numpy as np, pandas as pd
DALY = os.path.join(C.DATA, "daly_band_2023.csv")
DEATH = os.path.join(C.DATA, "deaths_zh_1980_2023.csv")
WPP = os.path.join(C.DATA, "WPP2024_pop_GBDagebands_1990_2035.csv")
YLL = os.path.join(C.DATA, "hbv_yll_yld_2015_2023.csv")
BANDS = ["5岁以下","5-14岁","15-19岁","20-24岁","25-29岁","30-34岁","35-39岁","40-44岁","45-49岁","50-54岁","55-59岁","60-64岁","65-69岁","70-74岁","75-79岁","80-84岁","85-89岁","90-94岁",">95岁"]
AGEID = [1,23,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235]
A2B = dict(zip(AGEID, BANDS)); HB, NA = 1026, 1028
L=[]
def say(s=""): L.append(str(s))
d = pd.read_csv(DALY, encoding="utf-8-sig")
rate = d[d.metric_name.astype(str).str.contains("率")]; num = d[d.metric_name.astype(str).str.contains("数量")]
def series(df,cause):
    s = df[(df.cause_id==cause)&(df.age_name.isin(BANDS))]
    return s.pivot_table(index="year",columns="age_name",values="val",aggfunc="sum").reindex(columns=BANDS)
R_H,R_N = series(rate,HB),series(rate,NA); N_H,N_N = series(num,HB),series(num,NA)
pop23 = N_H.loc[2023]/R_H.loc[2023]*1e5
H_obs,N_obs = N_H.loc[2023].sum(),N_N.loc[2023].sum(); L_obs=N_obs/(H_obs+N_obs)
r23H,r23N = R_H.loc[2023].copy(),R_N.loc[2023].copy()
w = pd.read_csv(WPP, encoding="utf-8-sig")
wp = w[w.gbd_name=="Global"].pivot_table(index="year",columns="age_band",values="population",aggfunc="sum").reindex(columns=BANDS)
YEARS=list(range(2023,2036)); popsc={y:pop23*(wp.loc[y]/wp.loc[2023]) for y in YEARS}; popsc[2023]=pop23.copy()
dd = pd.read_csv(DEATH, encoding="utf-8-sig")
dm = dd[dd.metric_name.astype(str).str.contains("率")]; dn = dd[dd.metric_name.astype(str).str.contains("数量")]
def dseries(df,cause):
    s = df[(df.cause_id==cause)&(df.age_name.isin(BANDS))]
    return s.pivot_table(index="year",columns="age_name",values="val",aggfunc="sum").reindex(columns=BANDS)
DR_H=dseries(dm,HB); DN_H=dseries(dn,HB)
base2015=DN_H.loc[2015].sum(); target=0.35*base2015
D2030=float((DR_H.loc[2023]*popsc[2030]/1e5).sum()); fC=(target/D2030)**(1/7)
yl = pd.read_csv(YLL, encoding="utf-8-sig"); yl=yl[yl.year==2023].assign(band=lambda z:z.age_id.map(A2B)); yl=yl[yl.band.isin(BANDS)]
piv = yl.pivot_table(index="band",columns="measure_name",values="val",aggfunc="sum")
ya = (piv["YLLs (Years of Life Lost)"]/(piv["YLLs (Years of Life Lost)"]+piv["YLDs (Years Lived with Disability)"])).reindex(BANDS)
def fits(R,y0,y1):
    bh={}; sub=R.loc[y0:y1].astype(float); yrs=sub.index.values.astype(float)
    agg=sub.sum(axis=1); b_agg=np.polyfit(yrs,np.log(agg.values),1)[0]
    for b in BANDS:
        v=sub[b].values.astype(float)
        if (v<=1e-12).all(): bh[b]=0.0; continue
        if (v<=1e-12).any(): bh[b]=b_agg; continue
        bh[b]=np.polyfit(yrs,np.log(v),1)[0]
    return bh
WINDOWS={"1990–2019":(1990,2019),"2000–2019":(2000,2019),"1990–2023":(1990,2023),"2010–2023":(2010,2023)}
FITN={k:fits(R_N,*v) for k,v in WINDOWS.items()}
def ldt(y,rH,rN):
    H=float((rH*popsc[y]/1e5).sum()); N=float((rN*popsc[y]/1e5).sum()); return N/(H+N),H,N
def C_of_s(s):
    out={}
    for win in WINDOWS:
        bN=FITN[win]; seg={}
        for y in YEARS:
            t=min(y-2023,7)
            fac=ya*fC**(t*s)+(1-ya)
            rH=pd.Series({b:r23H[b]*fac[b] for b in BANDS})
            rN=pd.Series({b:r23N[b]*np.exp(bN[b]*(y-2023)) for b in BANDS})
            l,_,_=ldt(y,rH,rN); seg[y]=l
        out[win]=seg
    return out
say("=== fC=%.5f  target=%.0f  base2015=%.0f  D2030=%.0f ===" % (fC,target,base2015,D2030))
say("=== death check (fC^t on death rate) ===")
for y in (2030,2031,2035):
    t=min(y-2023,7); deaths=float((DR_H.loc[2023]*(fC**t)*popsc[y]/1e5).sum())
    say("  C %d HBV deaths = %.0f" % (y,deaths))
say("=== AF5: s sensitivity on YLL-split baseline (s=1 == main) ===")
C_main = C_of_s(1.0)
main_vals = {w:100*C_main[w][2035] for w in WINDOWS}
for w in WINDOWS: say("  s=1.0  C %-11s %.3f%%" % (w, main_vals[w]))
say("  main range %.2f–%.2f" % (min(main_vals.values()),max(main_vals.values())))
for s in (0.9,0.8):
    C = C_of_s(s); vals=[100*C[w][2035] for w in WINDOWS]
    say("  s=%.1f  C %-11s %.3f / %.3f / %.3f / %.3f -> range %.2f–%.2f"
        % (s,list(WINDOWS)[0],vals[0],vals[1],vals[2],vals[3],min(vals),max(vals)))
say("")
say("=== uniform-baseline scan for s_equiv ===")
for tw in ["2000–2019","1990–2019"]:
    target_l = main_vals[tw]; bN=FITN[tw]; tN=12
    best=None
    for s in np.arange(0.90,1.001,0.0025):
        rH=pd.Series({b:r23H[b]*(fC**(7*s)) for b in BANDS})
        rN=pd.Series({b:r23N[b]*np.exp(bN[b]*tN) for b in BANDS})
        l,_,_=ldt(2035,rH,rN); v=100*l
        if best is None or abs(v-target_l)<abs(best[0]-target_l): best=(v,s)
    say("  main C (YLL split, %s) = %.3f%% ; closest uniform s=%.3f gives %.3f%%" % (tw,target_l,best[1],best[0]))
open(os.path.join(C.OUT, "af5_out.txt"),"w",encoding="utf-8").write("\n".join(L))
print("ok")
