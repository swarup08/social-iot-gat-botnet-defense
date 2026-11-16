from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source';DATA=ROOT/'data'
raw=pd.read_csv(DATA/'harness_per_graph.csv')
raw=raw[raw.method!='RL'].copy()
assert len(raw)==320 and not raw.duplicated(['graph_seed','method']).any()
summary=raw.groupby('method').agg(cr=('containment_ratio','mean'),cr_sd=('containment_ratio','std'),f1=('frozen_f1','mean'),f1_sd=('frozen_f1','std'),time=('time_s','mean')).sort_values('cr')
names={'degree-centrality':'Degree','eigenscore':'Eigenscore','betweenness-centrality':'Betweenness','GAT top-k':'GAT top-k','highest-p_uv':r'Highest $p_{uv}$','GAT threshold':'GAT threshold','greedy simulation-guided heuristic':'Greedy search','random':'Random'}
texnames={**names,'GAT top-k':r'GAT top-$k$','highest-p_uv':r'Highest $p_{uv}$'}
colors={k:('#2f7b85' if k.startswith('GAT') else '#b08635' if k.startswith('greedy') else '#425775') for k in names}
plt.rcParams.update({'font.family':['Arial','Liberation Sans','Nimbus Sans','DejaVu Sans'],'font.size':9,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'mathtext.fontset':'dejavusans','axes.spines.top':False,'axes.spines.right':False,'text.color':'#263342','axes.labelcolor':'#263342','xtick.color':'#455466','ytick.color':'#455466','pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):
    fig.savefig(SRC/(name+'.pdf'));fig.savefig(SRC/(name+'.png'),dpi=180);plt.close(fig)
def table(name,header,body,cols):
    (SRC/(name+'.tex')).write_text('\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n',encoding='utf-8')
# achieved removal (%): measured from the budget records for six rules; eigenscore and greedy search
# remove round(0.5*891)=446 edges by construction (prune_highest_score / 446 greedy steps, no early stop)
bud=pd.read_csv(DATA/'reward_ablation_per_graph.csv');bud=bud[~bud.method.str.startswith('RL')]
removed=(100*bud.groupby('method').removal_fraction.mean()).to_dict()
removed['eigenscore']=removed['greedy simulation-guided heuristic']=100*round(0.5*891)/891
table('table_main',r'Method & Containment ratio & Frozen F1 & Time (s) & Removed (\%)',[f'{texnames[m]} & ${r.cr:.4f} \\pm {r.cr_sd:.4f}$ & ${r.f1:.3f} \\pm {r.f1_sd:.3f}$ & {r.time:.4g} & {removed[m]:.1f} \\\\' for m,r in summary.iterrows()],'lcccc')
methods=[m for m in summary.index if m!='degree-centrality']
tests=[]
for metric in ['containment_ratio','frozen_f1']:
    wide=raw.pivot(index='graph_seed',columns='method',values=metric)
    for m in methods:
        d=wide[m]-wide['degree-centrality'];lo,hi=stats.t.interval(.95,39,loc=d.mean(),scale=stats.sem(d))
        tests.append({'metric':metric,'method':m,'diff':d.mean(),'lo':lo,'hi':hi,'p':stats.ttest_1samp(d,0).pvalue})
ps=np.array([t['p'] for t in tests]);order=np.argsort(ps);adj=np.empty(len(ps));adj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*np.arange(len(ps),0,-1)))
for t,a in zip(tests,adj):t['holm']=a
def ptex(p):
    if p>=.01:return f'{p:.3f}'
    e=int(np.floor(np.log10(p)));return f'{p/10**e:.2f}\\times10^{{{e}}}'
for metric,name in [('containment_ratio','table_containment_tests'),('frozen_f1','table_f1_tests')]:
    rows=[f"{texnames[t['method']]} & ${t['diff']:.4f}$ & $[{t['lo']:.4f}, {t['hi']:.4f}]$ & ${ptex(t['holm'])}$ \\\\" for t in tests if t['metric']==metric]
    table(name,r'Method & Difference & 95\% CI & Holm $p$',rows,'lccc')
fig,axes=plt.subplots(1,2,figsize=(7.0,3.3),sharey=True)
for ax,metric,title in zip(axes,['containment_ratio','frozen_f1'],['Containment ratio','Frozen-classifier F1']):
    for y,m in enumerate(methods):
        t=next(t for t in tests if t['metric']==metric and t['method']==m)
        ax.errorbar(t['diff'],y,xerr=[[t['diff']-t['lo']],[t['hi']-t['diff']]],fmt='o',color=colors[m],capsize=3,ms=4,lw=1.3)
    ax.axvline(0,color='#64748b',ls='--',lw=1);ax.set_title(title,fontsize=10);ax.set_xlabel('Difference from degree');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(len(methods)),[names[m] for m in methods]);axes[0].invert_yaxis();fig.tight_layout();save(fig,'paired_comparisons')
frontier=[]
for m,r in summary.iterrows():
    dom=(summary.cr<=r.cr)&(summary.f1>=r.f1)&((summary.cr<r.cr)|(summary.f1>r.f1))
    if not dom.any():frontier.append(m)
fig,ax=plt.subplots(figsize=(7.0,3.3));f=summary.loc[frontier].sort_values('cr');ax.plot(f.cr,f.f1,color='#9ab6ba',lw=1.5,zorder=1)
offsets={'degree-centrality':(-12,-18),'eigenscore':(-12,10),'betweenness-centrality':(9,-15),'GAT top-k':(-5,13),'GAT threshold':(8,0),'highest-p_uv':(10,-2),'greedy simulation-guided heuristic':(8,4),'random':(-47,-16)}
for m,r in summary.iterrows():
    ax.scatter(r.cr,r.f1,s=55,color=colors[m],edgecolor='white',linewidth=.8,zorder=3,marker='D' if m in frontier else 'o');ax.annotate(names[m],(r.cr,r.f1),xytext=offsets[m],textcoords='offset points',fontsize=9)
ax.set_xlim(.013,.125);ax.set_ylim(.375,.592);ax.set_xlabel('Mean containment ratio (lower is better)');ax.set_ylabel('Mean frozen-classifier F1 (higher is better)');ax.grid(alpha=.14);ax.set_axisbelow(True);fig.tight_layout();save(fig,'containment_utility_frontier')
stress=pd.read_csv(DATA/'stress_tests_per_graph.csv');stress=stress[stress.method!='RL'].copy()
conditions=['size_small (n=150, m=3)','size_large (n=600, m=3)','density_sparse (n=300, m=2)','density_dense (n=300, m=5)','aggressive_moderate (bias=-1.0, ~2x p_uv)','aggressive_high (bias=0.0, ~3.5x p_uv)']
labels=['150 nodes','600 nodes','Sparse (m = 2)','Dense (m = 5)','Infection bias = −1','Infection bias = 0']
sm=['degree-centrality','betweenness-centrality','highest-p_uv','GAT top-k','GAT threshold','random']
grid=stress.groupby(['condition','method']).containment_ratio.mean().unstack().loc[conditions,sm]
fig,ax=plt.subplots(figsize=(7.0,3.3));im=ax.imshow(grid.to_numpy(),cmap='YlGnBu',vmin=0,vmax=.75,aspect='auto')
ax.set_xticks(range(6),['Degree','Between.','Highest\n$p_{uv}$','GAT\ntop-k','GAT\nthresh.','Random'],fontsize=9);ax.set_yticks(range(6),labels,fontsize=9);ax.tick_params(length=0)
for i in range(6):
    for j in range(6):
        v=grid.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.40 else '#172b40',fontsize=9)
c=fig.colorbar(im,ax=ax,fraction=.03,pad=.025);c.set_label('Mean containment ratio',fontsize=9);c.ax.tick_params(labelsize=9)
for s in ax.spines.values():s.set_visible(False)
fig.tight_layout();save(fig,'stress_summary')
summary.to_csv(ROOT/'main_checked.csv');pd.DataFrame(tests).to_csv(ROOT/'paired_checked.csv',index=False)
stress.groupby(['condition','method']).agg(n=('graph_seed','size'),cr=('containment_ratio','mean'),sd=('containment_ratio','std'),f1=('frozen_f1','mean'),removed=('removal_fraction','mean')).to_csv(ROOT/'stress_checked.csv')
(ROOT/'verification.json').write_text(json.dumps({'main_rows':len(raw),'graphs':40,'methods':8,'frontier':frontier,'paired_family':14,'statistical_scope':'exploratory degree-referenced comparisons of two outcomes','training_rerun':False},indent=2))
print(pd.DataFrame(tests).to_string(index=False))
