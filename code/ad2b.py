import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
rate = d[d.metric_name.astype(str).str.contains("率")]
num  = d[d.metric_name.astype(str).str.contains("数量")]
def series(df, cause):
    s = df[(df.cause_id==cause) & (df.age_name.isin(BANDS))]
    return s.pivot_table(index="year", columns="age_name", values="val", aggfunc="sum").reindex(columns=BANDS)
R_H, R_N = series(rate,HB), series(rate,NA); N_H, N_N = series(num,HB), series(num,NA)
pop23 = N_H.loc[2023]/R_H.loc[2023]*1e5
H_obs, N_obs = N_H.loc[2023].sum(), N_N.loc[2023].sum(); L_obs=N_obs/(H_obs+N_obs)
r23H, r23N = R_H.loc[2023].copy(), R_N.loc[2023].copy()
w = pd.read_csv(WPP, encoding="utf-8-sig")
wp = w[w.gbd_name=="Global"].pivot_table(index="year", columns="age_band", values="population", aggfunc="sum").reindex(columns=BANDS)
YEARS=list(range(2023,2036)); popsc={y: pop23*(wp.loc[y]/wp.loc[2023]) for y in YEARS}; popsc[2023]=pop23.copy()
dd = pd.read_csv(DEATH, encoding="utf-8-sig")
dm = dd[dd.metric_name.astype(str).str.contains("率")]; dn = dd[dd.metric_name.astype(str).str.contains("数量")]
def dseries(df, cause):
    s = df[(df.cause_id==cause)&(df.age_name.isin(BANDS))]
    return s.pivot_table(index="year", columns="age_name", values="val", aggfunc="sum").reindex(columns=BANDS)
DN_H=dseries(dn,HB); DR_H=dseries(dm,HB)
base2015=DN_H.loc[2015].sum(); target=0.35*base2015
D2030=float((DR_H.loc[2023]*popsc[2030]/1e5).sum()); fC=(target/D2030)**(1/7)
yl = pd.read_csv(YLL, encoding="utf-8-sig"); yl=yl[yl.year==2023].assign(band=lambda z: z.age_id.map(A2B)); yl=yl[yl.band.isin(BANDS)]
piv = yl.pivot_table(index="band", columns="measure_name", values="val", aggfunc="sum")
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
FITH={k:fits(R_H,*v) for k,v in WINDOWS.items()}; FITN={k:fits(R_N,*v) for k,v in WINDOWS.items()}
def ldt(y,rH,rN):
    H=float((rH*popsc[y]/1e5).sum()); N=float((rN*popsc[y]/1e5).sum()); return N/(H+N), H, N
# corrected death check: mortality factor is fC^t (WHO target), NOT the DALY factor
say("=== corrected check: C deaths use fC^t on the death rate ===")
for y in (2029,2030,2031,2035):
    t=min(y-2023,7); deaths=float((DR_H.loc[2023]*(fC**t)*popsc[y]/1e5).sum())
    say("  C %d HBV deaths = %.0f (factor %.4f)" % (y, deaths, fC**t))
say("  target = %.0f" % target)
seriesA={}; seriesB={}; seriesC={}
for y in YEARS:
    l,_,_=ldt(y,r23H,r23N); seriesA[y]=l
for win in WINDOWS:
    bH,bN=FITH[win],FITN[win]; seg={}
    for y in YEARS:
        t=y-2023
        rH=pd.Series({b:r23H[b]*np.exp(bH[b]*t) for b in BANDS}); rN=pd.Series({b:r23N[b]*np.exp(bN[b]*t) for b in BANDS})
        l,_,_=ldt(y,rH,rN); seg[y]=l
    seriesB[win]=seg
    segc={}
    for y in YEARS:
        t=min(y-2023,7); fac=ya*fC**t+(1-ya)
        rH=pd.Series({b:r23H[b]*fac[b] for b in BANDS}); rN=pd.Series({b:r23N[b]*np.exp(bN[b]*(y-2023)) for b in BANDS})
        l,_,_=ldt(y,rH,rN); segc[y]=l
    seriesC[win]=segc
say("")
say("=== 2035 LDTI by scenario ===")
say("A            %.3f%%" % (100*seriesA[2035]))
for win in WINDOWS: say("B %-11s %.3f%%" % (win, 100*seriesB[win][2035]))
for win in WINDOWS: say("C %-11s %.3f%%" % (win, 100*seriesC[win][2035]))
bmin=min(100*seriesB[w][2035] for w in WINDOWS); bmax=max(100*seriesB[w][2035] for w in WINDOWS)
cmin=min(100*seriesC[w][2035] for w in WINDOWS); cmax=max(100*seriesC[w][2035] for w in WINDOWS)
say("B range %.2f–%.2f ; C range %.2f–%.2f" % (bmin,bmax,cmin,cmax))
fig, ax = plt.subplots(figsize=(7.4,4.6), dpi=300)
xs=list(YEARS)
ax.plot(xs,[100*seriesA[y] for y in xs], color="#808080", lw=1.8, label="A  age-specific rates frozen at 2023")
bvals=[[100*seriesB[w][y] for y in xs] for w in WINDOWS]
ax.fill_between(xs,[min(v[i] for v in bvals) for i in range(len(xs))],[max(v[i] for v in bvals) for i in range(len(xs))], color="#1f77b4", alpha=0.18, lw=0)
ax.plot(xs,[100*seriesB["2000–2019"][y] for y in xs], color="#1f77b4", lw=1.8, ls="-", label="B  fitted rate trends, four windows (band)")
ax.plot(xs,[100*seriesB["1990–2019"][y] for y in xs], color="#1f77b4", lw=1.4, ls="--")
cvals=[[100*seriesC[w][y] for y in xs] for w in WINDOWS]
ax.fill_between(xs,[min(v[i] for v in cvals) for i in range(len(xs))],[max(v[i] for v in cvals) for i in range(len(xs))], color="#2ca02c", alpha=0.16, lw=0)
ax.plot(xs,[100*seriesC["2000–2019"][y] for y in xs], color="#2ca02c", lw=1.8, ls="-", label="C  WHO hepatitis mortality target met for HBV by 2030, then held")
ax.plot([2023],[100*L_obs], marker="D", ms=7, color="black", zorder=5, label="Observed, 2023 (14.2%)")
ax.annotate("14.2%", (2023,100*L_obs), textcoords="offset points", xytext=(-6,-14), ha="right", fontsize=8)
ax.annotate("%.1f%%" % (100*seriesA[2035]), (2035,100*seriesA[2035]), textcoords="offset points", xytext=(4,0), va="center", fontsize=8)
ax.annotate("%.1f–%.1f%%" % (bmin,bmax), (2035,max(bvals)[-1]), textcoords="offset points", xytext=(4,0), va="center", fontsize=8, color="#1f77b4")
ax.annotate("%.1f–%.1f%%" % (cmin,cmax), (2035,max(cvals)[-1]), textcoords="offset points", xytext=(4,0), va="center", fontsize=8, color="#2ca02c")
ax.set_xlim(2022.6, 2037.6); ax.set_ylim(10, 40)
ax.set_xlabel("Year"); ax.set_ylabel("Liver disease transition index (%)")
ax.set_title("Projected composition of combined HBV and NAFLD DALYs, 2023–2035", fontsize=10)
ax.legend(loc="upper left", fontsize=7.2, frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e8e8e8", lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(C.DERIVED, "Fig_scenarios_2035.png"), dpi=300)
say("")
say("figure saved: derived/Fig_scenarios_2035.png")
# ---- write derived scenario tables (manuscript Table S) ----
def row(tag, rH, rN):
    Ld,H,N = ldt(2035, rH, rN)
    return [tag, 100*Ld, 100*(Ld-L_obs), H/1e6, N/1e6]
rows = [row("A", r23H, r23N)]
for win in ["1990–2019","1990–2023","2010–2023","2000–2019"]:
    bH,bN = FITH[win], FITN[win]
    rH = pd.Series({b: r23H[b]*np.exp(bH[b]*12) for b in BANDS})
    rN = pd.Series({b: r23N[b]*np.exp(bN[b]*12) for b in BANDS})
    rows.append(row("B_"+win.replace("–","_"), rH, rN))
bN2 = FITN["2000–2019"]
rN2 = pd.Series({b: r23N[b]*np.exp(bN2[b]*12) for b in BANDS})
def c_row(tag, s, split):
    if split:
        rH = pd.Series({b: r23H[b]*(ya[b]*fC**(7*s)+(1-ya[b])) for b in BANDS})
    else:
        rH = pd.Series({b: r23H[b]*(fC**(7*s)) for b in BANDS})
    return row(tag, rH, rN2)
rows.append(c_row("C_s08", 0.8, True))
rows.append(c_row("C_s09", 0.9, True))
rows.append(c_row("C_yllsplit", 1.0, True))
rows.append(c_row("C_uniform", 1.0, False))
pd.DataFrame(rows, columns=["scenario","LDTI_pct","delta_pp","HBV_DALYs_M","NAFLD_DALYs_M"]).to_csv(
    os.path.join(C.DERIVED, "scenario_results.csv"), index=False)
sr = [{"year":y, "A":100*seriesA[y],
       "B_1990_2019":100*seriesB["1990–2019"][y], "B_2000_2019":100*seriesB["2000–2019"][y],
       "B_1990_2023":100*seriesB["1990–2023"][y], "B_2010_2023":100*seriesB["2010–2023"][y],
       "C_1990_2019":100*seriesC["1990–2019"][y], "C_2000_2019":100*seriesC["2000–2019"][y],
       "C_1990_2023":100*seriesC["1990–2023"][y], "C_2010_2023":100*seriesC["2010–2023"][y]} for y in YEARS]
pd.DataFrame(sr).to_csv(os.path.join(C.DERIVED, "scenario_series.csv"), index=False)
open(os.path.join(C.OUT, "ad2b_out.txt"),"w",encoding="utf-8").write("\n".join(L))
print("ok")
