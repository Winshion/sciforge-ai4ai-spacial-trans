import json, numpy as np
from common import ROOT,write_json
preds=sorted((ROOT/'predictions').glob('*.csv')); plots=sorted((ROOT/'plots').glob('*.png')); reports=sorted((ROOT/'reports').glob('*.md'))
metrics=[p for p in sorted((ROOT/'metrics').glob('*.json')) if p.name not in ['verified_metrics.json','baseline_summary.json','mcfst_summary.json','audit.json']]
verified=json.loads((ROOT/'metrics/verified_metrics.json').read_text()) if (ROOT/'metrics/verified_metrics.json').exists() else []
ms=[json.loads(p.read_text()) for p in metrics if p.name.startswith('mcfst_seed_')]; bs=[json.loads(p.read_text()) for p in metrics if not p.name.startswith('mcfst_seed_')]
aris=np.array([m['ari'] for m in ms],float); failures=[]
if len(ms)<3: failures.append('fewer than three MCFST runs')
if len(preds)!=len(verified): failures.append('prediction and verification counts differ')
if len({b.get('method') for b in bs})<2: failures.append('fewer than two baseline methods')
if not (ROOT/'provenance/provenance.json').exists(): failures.append('missing provenance')
if len(plots)<len(preds)+1: failures.append('missing expected plots')
a={'pass':not failures,'failures':failures,'n_runs':len(ms),'seeds':[m.get('seed') for m in ms],'mcfst_aris':[float(x) for x in aris],'mean':float(aris.mean()) if len(aris) else None,'std':float(aris.std(ddof=1)) if len(aris)>1 else None,'median':float(np.median(aris)) if len(aris) else None,'min':float(aris.min()) if len(aris) else None,'max':float(aris.max()) if len(aris) else None,'n_predictions':len(preds),'n_plots':len(plots),'baseline_methods':sorted({b.get('method') for b in bs}),'official_retrieval':'see provenance/provenance.json'}
write_json(ROOT/'metrics/audit.json',a); (ROOT/'reports/reviewer_audit.md').write_text('# Reviewer Audit\n\n```json\n'+json.dumps(a,indent=2)+'\n```\n'); print(json.dumps(a,indent=2))
