from pathlib import Path
import json, platform, subprocess
from common import ROOT, OFFICIAL, WSSENV, sha256, write_json, ensure_dirs, build_or_load_processed
ensure_dirs(); data=build_or_load_processed()
def run(cmd):
    r=subprocess.run(cmd,text=True,capture_output=True); return {'cmd':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
files=[ROOT/'reprod.pdf', OFFICIAL/'README.md', OFFICIAL/'data/v1_filtered_feature_bc_matrix.h5', OFFICIAL/'data/metadata.tsv', OFFICIAL/'data/spatial/tissue_positions_list.csv', ROOT/'data/processed/features_pca.npy', ROOT/'data/processed/labels.npy']
prov={'official_code_url':'https://github.com/dw666666/MCFST','official_head':run(['git','-C',str(OFFICIAL),'rev-parse','HEAD']),'retrieval_attempts':['sandbox git ls-remote failed via 127.0.0.1:7897 proxy refusal','escalated git ls-remote succeeded: 4fb850fb5e2c91b89cdfc1b7f9ce184b01842261','git clone succeeded into provenance/official_MCFST_remote','previous .git_repro/local preprocessed snapshot disappeared during run; raw official data path used'], 'dataset':'official repository 10x Visium H5 plus metadata.tsv, 3798 spots, 20 fine_annot_type labels', 'processed_features':'CP10K normalization, log1p, top 3000 variance genes, z-score, PCA 1000 components', 'environment':{'python':run([str(WSSENV/'bin/python'),'-V']),'pip_freeze':run([str(WSSENV/'bin/python'),'-m','pip','freeze']),'platform':platform.platform()}, 'checksums_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files if p.exists()}}
write_json(ROOT/'provenance/provenance.json', prov); (ROOT/'provenance/pip_freeze.txt').write_text(prov['environment']['pip_freeze']['stdout']); print(json.dumps({'wrote':'provenance/provenance.json','n_spots':len(data['labels'])},indent=2))
