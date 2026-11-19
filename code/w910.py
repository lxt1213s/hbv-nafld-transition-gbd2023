# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
BAND = r"C:\Users\18328\AppData\Local\Temp\IHME-GBD_2023_DATA-577b8141-1.csv"
AGG  = r"C:\Users\18328\AppData\Local\Temp\IHME-GBD_2023_DATA-d90e847e-1 (1).csv"
WPP  = r"D:\论文\第一篇\WPP2024_pop_GBDagebands_1990_2035.csv"
HB,NA,HC=1026,1028,1027
AGES=[1,23,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235]
b=pd.read_csv(BAND,encoding="utf-8-sig"); a=pd.read_csv(AGG,encoding="utf-8-sig"); w=pd.read_csv(WPP,encoding="utf-8-sig")
b["metric_name"]=b["metric_name"].replace({"数量":"Number","率":"Rate"})
a["metric_name"]=a["metric_name"].replace({"数量":"Number","率":"Rate"})
b["age_id"]=b["age_id"].astype(int); a["age_id"]=a["age_id"].astype(int)
b["cause_id"]=b["cause_id"].astype(int); a["cause_id"]=a["cause_id"].astype(int)
out=[]
def W(s=""): out.append(str(s))
asr = a[(a.age_id==27)&(a.metric_name=="Rate")].pivot_table(index="year",columns="cause_id",values="val")
asr.columns=["HBV","HCV","NAFLD"]
W("=== W10 年龄标化 DALY 率（per 100k, GBD 2023）===")
for nm in ("HBV","HCV","NAFLD"):
    v0=float(asr.loc[1990,nm]); v1=float(asr.loc[2023,nm])
    W("  %-6s %.2f -> %.2f  (%+.1f%%)"%(nm,v0,v1,100*(v1/v0-1)))
name_map = b[["age_id","age_name"]].drop_duplicates().set_index("age_id")["age_name"].to_dict()
bnum = b[(b.cause_id==HB)&(b.age_id.isin(AGES))&(b.metric_name=="Number")].pivot_table(index="age_id",columns="year",values="val").reindex(AGES)
brat = b[(b.cause_id==HB)&(b.age_id.isin(AGES))&(b.metric_name=="Rate")].pivot_table(index="age_id",columns="year",values="val").reindex(AGES)
W("\n=== W9 分年龄档对账：GBD 反推人口 vs WPP 2024（单位：人）===")
for yr in (1990,2023):
    gpop = bnum[yr].values/brat[yr].values*1e5
    W("--- %d ---"%yr)
    gtot=wtot=0.0
    for i,aid in enumerate(AGES):
        lab=name_map[aid]
        wr = w[(w.gbd_name=="Global")&(w.year==yr)&(w.age_band==lab)]
        if len(wr)==0:
            W("  %-9s GBD %12.0f  WPP [no match]"%(lab,gpop[i])); continue
        wpop=float(wr.population.iloc[0])
        gtot+=gpop[i]; wtot+=wpop
        W("  %-9s GBD %12.0f  WPP %12.0f  diff %+.2f%%"%(lab,gpop[i],wpop,100*(wpop/gpop[i]-1)))
    W("  TOTAL GBD %.0f  WPP %.0f  diff %+.2f%%"%(gtot,wtot,100*(wtot/gtot-1)))
open(r"D:\论文\第一篇\_tmp\w910_out.txt","w",encoding="utf-8").write("\n".join(out))
print("w910 done")
