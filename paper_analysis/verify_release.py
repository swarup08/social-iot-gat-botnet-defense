"""Validate released observations and derived summaries; no training is run."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import stats

root=Path(__file__).resolve().parent
data=root/'data'
names=['degree-centrality','eigenscore','betweenness-centrality','highest-p_uv',
       'GAT top-k','GAT threshold','greedy simulation-guided heuristic','random']
raw=pd.read_csv(data/'harness_per_graph.csv')
main=raw[raw.method.isin(names)]
assert len(main)==320
assert not main.duplicated(['graph_seed','method']).any()
assert set(main.graph_seed)==set(range(40))
assert main.groupby('graph_seed').size().eq(8).all()
assert np.isfinite(main[['containment_ratio','frozen_f1','time_s']]).all().all()
summary=pd.read_csv(root/'main_checked.csv').set_index('method')
for name in names:
    rows=main[main.method==name]
    for col,out in [('containment_ratio','cr'),('frozen_f1','f1'),('time_s','time')]:
        assert np.isclose(rows[col].mean(),summary.loc[name,out],rtol=0,atol=1e-12)

paired=pd.read_csv(root/'paired_checked.csv')
assert len(paired)==14
for row in paired.itertuples():
    w=main.pivot(index='graph_seed',columns='method',values=row.metric)
    d=w[row.method]-w['degree-centrality']
    assert np.isclose(d.mean(),row.diff,atol=1e-12,rtol=0)
    assert np.isclose(stats.ttest_rel(w[row.method],w['degree-centrality']).pvalue,row.p,atol=1e-12,rtol=0)
    half=stats.t.ppf(.975,39)*d.std(ddof=1)/np.sqrt(40)
    assert np.allclose([d.mean()-half,d.mean()+half],[row.lo,row.hi],atol=1e-12,rtol=0)
order=np.argsort(paired.p.to_numpy());running=0
for rank,idx in enumerate(order):
    running=max(running,(14-rank)*paired.iloc[idx].p)
    assert np.isclose(min(1,running),paired.iloc[idx].holm,atol=1e-12,rtol=0)

repeat=pd.read_csv(data/'reward_ablation_per_graph.csv')
joined=main.merge(repeat,on=['graph_seed','method'],suffixes=('_main','_repeat'),validate='one_to_one')
assert len(joined)==240
for c in ['containment_ratio','frozen_f1']:
    assert np.allclose(joined[c+'_main'],joined[c+'_repeat'],atol=1e-12,rtol=0)
budget=joined.groupby('method').removal_fraction.agg(['mean','std'])
assert np.isclose(budget.loc['GAT top-k','mean']*100,55.6117,atol=.0001)
stress=pd.read_csv(data/'stress_tests_per_graph.csv')
stress=stress[stress.method.isin(names)]
assert stress.groupby(['condition','method']).size().eq(15).all()
assert stress.condition.nunique()==6 and stress.method.nunique()==6
cor=pd.read_csv(data/'diagnostic_correlation_per_graph.csv')
assert len(cor)==120 and cor.groupby('metric').size().eq(40).all()
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.csv')}
manifest=root/'INPUT_SHA256.json'
if manifest.exists(): assert json.loads(manifest.read_text())==hashes
else: manifest.write_text(json.dumps(hashes,indent=2))
out={'main_graphs':40,'main_methods':8,'main_selected_rows':320,'paired_tests':14,
     'stress_conditions':6,'graphs_per_stress_condition':15,'stress_selected_rows':len(stress),
     'matching_repeated_baseline_rows':240,'correlation_rows':120,
     'scope':'Recalculation from released records; original training not rerun',
     'budget_means':budget['mean'].to_dict(),
     'correlation_means':cor.groupby('metric').correlation_value.mean().to_dict()}
(root/'VERIFICATION.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
