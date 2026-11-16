"""Draw the study protocol (print-safe, text >= 9 pt at final size). No experiment outcomes are simulated."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
root=Path(__file__).resolve().parent/'source'
plt.rcParams.update({'font.family':['Arial','Liberation Sans','Nimbus Sans','DejaVu Sans'],'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(6.8,3.05))   # about the page text-block width (7 in): printed at ~1:1
ax.set(xlim=(0,10),ylim=(0,4.4));ax.axis('off')
ink='#111111';dark='#1b2a3a';blue='#0b3d6e';green='#0b5d4f'
def box(x,y,w,h,title,lines,face,edge,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.08',facecolor=face,edgecolor=edge,lw=1.6))
    ax.text(x+.12,y+h-.27,title,fontsize=9.5,weight='bold',color=color,va='center')
    ax.text(x+.12,y+h-.55,lines,fontsize=9,color=ink,va='top',linespacing=1.35)
box(.05,1.35,3.05,2.95,'ORIGINAL GRAPHS','40 BA graphs\n300 nodes, 891 edges\nDerived node attributes\nLabels: 15 cascades\nOne GAT per graph','#dce6f2',blue,blue)
box(3.55,1.35,2.75,2.95,'PRUNING','4 structural rules\n2 GAT attention rules\nRandom removal\nGreedy cascade search\nNominal removal: 50%','#dce6f2',blue,blue)
box(6.8,2.85,3.15,1.45,'CONTAINMENT','15 final rollouts\nFive initial conditions','#d5ebe3',green,green)
box(6.8,1.35,3.15,1.3,'CLASSIFIER UTILITY','Frozen GAT, original labels\nF1 on held-out nodes','#d5ebe3',green,green)
def arrow(x0,y0,x1,y1): ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops={'arrowstyle':'-|>','color':ink,'lw':1.6,'mutation_scale':12})
arrow(3.17,2.8,3.47,2.8);arrow(6.37,3.1,6.72,3.5);arrow(6.37,2.5,6.72,2.0)
ax.add_patch(FancyBboxPatch((.05,.1),9.9,.88,boxstyle='round,pad=.05,rounding_size=.06',facecolor='#e6e6e6',edgecolor=dark,lw=1.5))
ax.text(5.0,.74,'PAIRED COMPARISON PER GRAPH',ha='center',va='center',fontsize=9.5,weight='bold',color=ink)
ax.text(5.0,.34,'40 pairs | 14 Holm-adjusted tests | budgets and timings reported',ha='center',va='center',fontsize=9,color=ink)
fig.tight_layout(pad=.1)
fig.savefig(root/'evaluation_protocol.pdf',bbox_inches='tight');fig.savefig(root/'evaluation_protocol.png',dpi=160,bbox_inches='tight')
