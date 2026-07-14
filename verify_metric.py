import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT=Path('/Users/winshion/.sciforge/default_workspace')
p=argparse.ArgumentParser(); p.add_argument('files',nargs='*'); a=p.parse_args(); paths=[Path(x) for x in a.files] if a.files else sorted((ROOT/'predictions').glob('*.csv'))
res=[]
for path in paths:
    path=path if path.is_absolute() else ROOT/path; df=pd.read_csv(path); res.append({'prediction_file':str(path.relative_to(ROOT)),'n_rows':len(df),'n_labels':int(df.label.nunique()),'n_predictions':int(df.prediction.nunique()),'ari':float(adjusted_rand_score(df.label,df.prediction))})
(ROOT/'metrics/verified_metrics.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n'); print(json.dumps(res,indent=2,sort_keys=True))
