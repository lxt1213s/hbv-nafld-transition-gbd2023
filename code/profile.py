import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import pandas as pd, numpy as np

f = os.path.join(C.DATA, "daly_band_2023.csv")
d = pd.read_csv(f, encoding="utf-8-sig")
AGES = [1,23,8,9,10,11,12,13,14,15,16,17,18,19,20,30,31,32,235]
LAB = {1:'5岁以下',23:'5-14岁',8:'15-19岁',9:'20-24岁',10:'25-29岁',11:'30-34岁',12:'35-39岁',13:'40-44岁',14:'45-49岁',15:'50-54岁',16:'55-59岁',17:'60-64岁',18:'65-69岁',19:'70-74岁',20:'75-79岁',30:'80-84岁',31:'85-89岁',32:'90-94岁',235:'>95岁'}
CAU = {1026:'HBV', 1028:'NAFLD'}

def vec(cid, y, mid):
    return d[(d.cause_id==cid) & (d.year==y) & (d.metric_id==mid)].set_index('age_id').val.reindex(AGES).to_numpy(dtype=float)

# population from HBV (all bands rate>0); NAFLD shares the same population
pH0 = vec(1026, 1990, 1) / vec(1026, 1990, 3) * 1e5
pH1 = vec(1026, 2023, 1) / vec(1026, 2023, 3) * 1e5
s0 = pH0 / pH0.sum(); s1 = pH1 / pH1.sum()

out = []
for cid, cn in CAU.items():
    r0 = vec(cid, 1990, 3); r1 = vec(cid, 2023, 3)
    rt = ((s0 + s1) / 2 * (r1 - r0)).sum()
    at = ((r0 + r1) / 2 * (s1 - s0)).sum()
    rb0 = (r0 * s0).sum(); rb1 = (r1 * s1).sum()
    out.append("%s: crude %.2f->%.2f (d=%.2f)  rate_term=%.2f  age_term=%.2f" % (cn, rb0, rb1, rb1-rb0, rt, at))
for cid, cn in CAU.items():
    r0 = vec(cid, 1990, 3); r1 = vec(cid, 2023, 3)
    i0 = int(np.argmax(r0)); i1 = int(np.argmax(r1))
    out.append("%s peak: 1990 %s=%.1f, 2023 %s=%.1f" % (cn, LAB[AGES[i0]], r0[i0], LAB[AGES[i1]], r1[i1]))
for cid, cn in [(1028, 'NAFLD')]:
    r0 = vec(cid, 1990, 3); r1 = vec(cid, 2023, 3)
    out.append("%s 55-59: %.1f -> %.1f ; 95+: %.1f -> %.1f" % (cn, r0[AGES.index(16)], r1[AGES.index(16)], r0[AGES.index(235)], r1[AGES.index(235)]))

open(os.path.join(C.OUT, "profile_out.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
