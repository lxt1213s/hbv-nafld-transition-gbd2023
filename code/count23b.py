import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

import itertools, math
import numpy as np
import pandas as pd

p = os.path.join(C.DATA, "daly_band_2023.csv")
d = pd.read_csv(p, encoding="utf-8-sig")
num = d[d.metric_name=="数量"]; rate = d[d.metric_name=="率"]
ages = sorted(num.age_id.unique())
def cell(df,c,y):
    return df[(df.cause_id==c)&(df.year==y)].set_index("age_id").loc[ages].val.values
# population from HBV (NAFLD has zero rates under 15 -> 0/0)
def popvec(y):
    return cell(num,1026,y)/cell(rate,1026,y)*1e5
def shap3(D0,D1,P0,P1,w0,w1,r0,r1):
    def f(m):
        P = P1 if m[0] else P0; w = w1 if m[1] else w0; r = r1 if m[2] else r0
        return P*np.sum(w*r)/1e5
    n=3; phi=[0.0]*n
    for k in range(n):
        others=[j for j in range(n) if j!=k]
        for size in range(n):
            wgt = math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            for S in itertools.combinations(others,size):
                m0=[j in S for j in range(n)]; m1=m0.copy(); m1[k]=True
                phi[k]+= wgt*(f(m1)-f(m0))
    return np.array(phi)
out=[]
for cause,name in [(1026,"HBV"),(1028,"NAFLD")]:
    for y0,y1 in [(1990,2023),(1990,2021)]:
        p0,p1 = popvec(y0), popvec(y1)
        P0,P1 = p0.sum(), p1.sum()
        r0,r1 = cell(rate,cause,y0), cell(rate,cause,y1)
        w0,w1 = p0/P0, p1/P1
        D0 = P0*np.sum(w0*r0)/1e5; D1 = P1*np.sum(w1*r1)/1e5
        ph = shap3(D0,D1,P0,P1,w0,w1,r0,r1); dD=D1-D0
        out.append("%s %d->%d  D %.3fM -> %.3fM (%+.1f%%)" % (name,y0,y1,D0/1e6,D1/1e6,100*(D1/D0-1)))
        out.append("   pop %+.3fM  age %+.3fM  rate %+.3fM  |  sum %+.4fM vs %+.4fM" % (
            ph[0]/1e6, ph[1]/1e6, ph[2]/1e6, ph.sum()/1e6, dD/1e6))
        out.append("   demographic (pop+age) = %+.3fM = %.1f%% of the increase" % ((ph[0]+ph[1])/1e6, 100*(ph[0]+ph[1])/dD))
open(os.path.join(C.OUT, "count2023.txt"),"w",encoding="utf-8").write("\n".join(out))
print("ok")
