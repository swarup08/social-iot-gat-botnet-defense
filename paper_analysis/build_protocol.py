"""Draw the study protocol. No experiment outcomes are simulated."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root=Path(__file__).resolve().parent/'source'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9.3,4.0))
ax.set(xlim=(0,10),ylim=(0,4.25));ax.axis('off')
navy='#263e57';teal='#237d83';muted='#4f5f6d'
def box(x,y,w,h,title,lines,color=navy):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.08',facecolor='#f5f8fa',edgecolor='#bdccd6',lw=1))
    ax.text(x+.15,y+h-.28,title,fontsize=13,weight='bold',color=color,va='center')
    ax.text(x+.15,y+h-.58,lines,fontsize=11.5,color=muted,va='top',linespacing=1.5)
def arrow(x0,y0,x1,y1):
    ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops={'arrowstyle':'-|>','color':teal,'lw':1.7,'mutation_scale':13})
box(.1,1.30,2.7,2.55,'ORIGINAL GRAPHS','40 BA instances\n300 nodes, 891 edges\nTopology-derived attributes\nLabels from 15 cascades\nOne GAT trained per graph')
box(3.35,1.30,2.55,2.55,'PRUNING','4 structural / risk rules\n2 GAT attention rules\nRandom removal\nGreedy cascade search\nNominal removal: 50%')
box(6.55,2.69,3.2,1.16,'CONTAINMENT','15 final rollouts per condition\nFive initial-node conditions',teal)
box(6.55,1.30,3.2,1.16,'CLASSIFIER UTILITY','Frozen GAT; original labels\nF1 on held-out test nodes',teal)
arrow(2.92,2.5,3.23,2.5);arrow(6.00,2.64,6.43,3.22);arrow(6.00,2.47,6.43,1.85)
ax.add_patch(FancyBboxPatch((.1,.16),9.65,.74,boxstyle='round,pad=.07,rounding_size=.06',facecolor='#e8f2f2',edgecolor='none'))
ax.text(4.92,.68,'PAIRED COMPARISON AT THE GRAPH LEVEL',ha='center',va='center',fontsize=12.5,weight='bold',color=navy)
ax.text(4.92,.34,'40 paired outcomes  |  14 Holm-adjusted tests  |  Achieved budgets and timing scopes reported',ha='center',va='center',fontsize=10.5,color=muted)
fig.tight_layout(pad=.2)
fig.savefig(root/'evaluation_protocol.pdf',bbox_inches='tight')
fig.savefig(root/'evaluation_protocol.png',dpi=220,bbox_inches='tight')
