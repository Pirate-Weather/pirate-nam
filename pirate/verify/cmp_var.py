"""Compare one test against the archive, next to the previous run. Usage: cmp_var.py NEW_CSV PREV_CSV [LABEL_PREV]"""
import sys, pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
n = pd.read_csv(sys.argv[1]); o = pd.read_csv(sys.argv[2]); lab = sys.argv[3] if len(sys.argv) > 3 else 'prev'
m = n[n.source == 'mine'].merge(o[o.source == 'mine'], on=['fhr', 'field'], suffixes=('_new', '_prev'))
m = m[m.rmse_prev > 0]
for h in (0, 6, 12):
    print('--- f', h); print(m[m.fhr == h].set_index('field')[['rmse_new', 'rmse_prev', 'bias_new', 'bias_prev', 'corr_new', 'corr_prev']].round(3).to_string())
print('closer than', lab, 'by hour'); print(m.assign(b=m.rmse_new < m.rmse_prev).groupby('fhr').b.agg(['sum', 'count']).T.to_string())
print('median rmse ratio'); print(m.assign(r=m.rmse_new / m.rmse_prev).groupby('fhr').r.median().round(3).to_string())
g = n[n.source == 'gfs'][['fhr', 'field', 'rmse']].rename(columns={'rmse': 'g'}).merge(n[n.source == 'mine'][['fhr', 'field', 'rmse']], on=['fhr', 'field'])
print('closer than GFS', int((g.rmse < g.g).sum()), 'of', len(g))
