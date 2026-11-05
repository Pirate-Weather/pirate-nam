"""Compare cycled WOOF-NAM runs with the Fortran run on the same inputs (test 4), the Fortran-analysis hybrid and a GFS start."""
import os
import sys, pandas as pd
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
W=os.environ["PIRATE_DATA"]+"/woof/scored/"; C=os.environ["PIRATE_DATA"]+"/compare/"
def w(r,c='rmse'): d=pd.read_csv(W+r+'/series.csv'); return d.set_index(['fhr','field'])[c]
def f(p,c='rmse'): d=pd.read_csv(p); return d[d.source=='mine'].set_index(['fhr','field'])[c]
def table(c):
    t=pd.DataFrame({r:w(r,c) for r in sys.argv[1:]}); t['anl+woof']=w('ours_nodfi',c); t['gfs']=w('gfs',c); t['fort4']=f(C+'var/ens_radar_rap/series.csv',c)
    return t[t.index.get_level_values(1)!='PRMSL']
F=['HGT500','HGT250','RH700','TMP2m','DPT2m','TMP850','WIND10m','WIND850','WIND250','REFC','APCP','MSLET']
for c in ('rmse','corr','bias'):
    t=table(c); t=t[t.index.get_level_values(0).isin([0,6,12])&t.index.get_level_values(1).isin(F)]; print(c); print(t.round(3).to_string())
t=table('rmse')
for r in sys.argv[1:]:
    for b in ('fort4','anl+woof','gfs'):
        x=t[[r,b]].dropna(); print(r,'vs',b,int((x[r]<x[b]).sum()),len(x),'median',round((x[r]/x[b]).median(),3))
