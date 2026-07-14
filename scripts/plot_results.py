import json, os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path.home()/'.sciforge/default_workspace/provenance/mplconfig')
import matplotlib.pyplot as plt, pandas as pd
from common import ROOT,PAPER_ARI,build_or_load_processed,ensure_dirs
ensure_dirs(); rows=[]
for p in sorted((ROOT/'metrics').glob('*.json')):
    if p.name in ['verified_metrics.json','baseline_summary.json','mcfst_summary.json','audit.json']: continue
    m=json.loads(p.read_text())
    if isinstance(m,dict) and 'ari' in m: rows.append({'method':m.get('method'), 'seed':m.get('seed'), 'ari':m['ari']})
df=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(7,4));
for i,(method,sub) in enumerate(df.groupby('method')):
    ax.scatter([i]*len(sub),sub.ari,s=50,label=method); ax.plot([i-0.15,i+0.15],[sub.ari.mean(),sub.ari.mean()],color='black')
ax.axhline(PAPER_ARI,color='red',ls='--',label='paper MCFST 0.693'); ax.set_xticks(range(len(df.method.unique()))); ax.set_xticklabels(list(df.method.unique())); ax.set_ylabel('ARI'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'plots/ari_comparison.png',dpi=180); plt.close(fig)
data=build_or_load_processed(); pos=data['positions']
for predfile in sorted((ROOT/'predictions').glob('*.csv')):
    pred=pd.read_csv(predfile); d=pos.merge(pred[['ID','label','prediction']],on='ID')
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    for ax,col,title in [(axs[0],'label','Manual 20 labels'),(axs[1],'prediction',predfile.stem)]:
        sc=ax.scatter(d.pxl_col,-d.pxl_row,c=d[col],s=7,cmap='tab20',linewidths=0); ax.set_title(title); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(ROOT/f'plots/{predfile.stem}_spatial.png',dpi=180); plt.close(fig)
print('plots written')
