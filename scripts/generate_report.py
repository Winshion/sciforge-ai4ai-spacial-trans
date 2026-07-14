import json, numpy as np, statistics, subprocess
from common import ROOT,PAPER_ARI,latest_commit
items=[]
for p in sorted((ROOT/'metrics').glob('*.json')):
    if p.name in ['verified_metrics.json','baseline_summary.json','mcfst_summary.json','audit.json']: continue
    m=json.loads(p.read_text())
    if 'ari' in m: items.append(m)
mc=[m for m in items if m.get('method')=='MCFST']; bs=[m for m in items if m.get('method')!='MCFST']; ar=np.array([m['ari'] for m in mc],float); audit=json.loads((ROOT/'metrics/audit.json').read_text()) if (ROOT/'metrics/audit.json').exists() else {}; prov=json.loads((ROOT/'provenance/provenance.json').read_text()) if (ROOT/'provenance/provenance.json').exists() else {}
if len(mc)==0: verdict='failed'
elif audit.get('pass') and abs(float(ar.mean())-PAPER_ARI)<=0.10 and abs(float(ar.max())-PAPER_ARI)<=0.10: verdict='reproduced'
else: verdict='partially reproduced'
lines=['# Final Report','','## Paper-reported metric',f'MCFST ARI = {PAPER_ARI:.3f} on Section 2.3 / Fig. 3A human breast cancer, 20 annotated subregions.','','## Best reproduced run']
if len(mc):
    best=max(mc,key=lambda m:m['ari']); lines.append(f"Seed {best['seed']}, epochs {best['epochs']}, ARI {best['ari']:.6f}.")
else: lines.append('No MCFST run completed.')
lines += ['','## Full-run mean +/- std']
if len(ar): lines += [f"Seeds: {[m['seed'] for m in mc]}", f"All ARIs: {[round(float(x),6) for x in ar]}", f"Mean +/- std: {ar.mean():.6f} +/- {(ar.std(ddof=1) if len(ar)>1 else 0):.6f}", f"Median: {float(np.median(ar)):.6f}", f"Min/max: {ar.min():.6f} / {ar.max():.6f}"]
else: lines.append('No completed MCFST runs.')
lines += ['','## Selected subset result','No subset selected; all completed fixed-seed runs are reported.','','## Baseline results']
for b in bs: lines.append(f"- {b.get('method')} seed {b.get('seed')}: ARI {b['ari']:.6f}, predicted clusters {b.get('n_clusters_predicted')}.")
if not bs: lines.append('No baselines completed.')
lines += ['','## Instability analysis']
lines.append('MCFST run variance is reported above; the verdict is based on the full distribution, not the best seed alone.' if len(ar)>1 else 'Instability cannot be estimated from fewer than two completed MCFST runs.')
lines += ['','## Blockers','- Original `.git_repro` and contract files disappeared during execution; no destructive reset was performed. A new `.git_repro` was initialized only if needed for final checkpointing.','- Previous local preprocessed arrays were unavailable after that workspace change; this reproduction rebuilt features/graphs from official raw H5 and metadata, so it is not bit-identical to the paper preprocessing.','- Official `main_v4.py` is CUDA-assumptive; CPU wrapper used official model architecture with rebuilt graph views.','','## Reviewer verdict',f"Audit pass: {audit.get('pass')}; failures: {audit.get('failures')}.",'','## Latest Git commit',latest_commit(),'','## Final verdict',verdict]
(ROOT/'reports/final_report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines[-6:]))
