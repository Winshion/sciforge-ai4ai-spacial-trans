from __future__ import annotations
import hashlib, json, os, random, subprocess
from pathlib import Path
import h5py, networkx as nx, numpy as np, pandas as pd, torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

ROOT=Path.home()/'.sciforge/default_workspace'
OFFICIAL=ROOT/'provenance/official_MCFST_remote'
WSSENV=Path.home()/'wssenv'
N_CLUSTERS=20
PAPER_ARI=0.693

def ensure_dirs():
    for d in ['provenance','code','data','scripts','runs','predictions','metrics','plots','reports']:
        (ROOT/d).mkdir(exist_ok=True)
    (ROOT/'data/processed').mkdir(parents=True, exist_ok=True)
    (ROOT/'provenance/mplconfig').mkdir(parents=True, exist_ok=True)

def set_seed(seed:int):
    os.environ['PYTHONHASHSEED']=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path:Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def load_raw_counts():
    p=OFFICIAL/'data/v1_filtered_feature_bc_matrix.h5'
    with h5py.File(p,'r') as f:
        m=f['matrix']; shape=tuple(m['shape'][()])
        x=sparse.csc_matrix((m['data'][()], m['indices'][()], m['indptr'][()]), shape=shape).T.tocsr()
        barcodes=np.array([b.decode() for b in m['barcodes'][()]])
    return x, barcodes

def build_or_load_processed(n_features=1000, k=10):
    ensure_dirs(); feat_p=ROOT/'data/processed/features_pca.npy'; lab_p=ROOT/'data/processed/labels.npy'; pos_p=ROOT/'data/processed/positions.csv'; meta_p=ROOT/'data/processed/metadata.csv'
    if feat_p.exists() and lab_p.exists() and pos_p.exists():
        return {'features':np.load(feat_p), 'labels':np.load(lab_p), 'metadata':pd.read_csv(meta_p), 'positions':pd.read_csv(pos_p)}
    counts, barcodes=load_raw_counts(); meta=pd.read_csv(OFFICIAL/'data/metadata.tsv', sep='\t')
    order=pd.DataFrame({'ID':barcodes, 'row':np.arange(len(barcodes))}).merge(meta, on='ID', how='inner')
    counts=counts[order['row'].to_numpy()]
    labels=pd.Categorical(order['fine_annot_type']).codes.astype(int)
    totals=np.asarray(counts.sum(axis=1)).ravel(); totals[totals==0]=1
    norm=counts.multiply(10000/totals[:,None]).tocsr(); norm.data=np.log1p(norm.data)
    means=np.asarray(norm.mean(axis=0)).ravel(); sq=norm.copy(); sq.data **=2; vars_=np.asarray(sq.mean(axis=0)).ravel()-means**2
    idx=np.argsort(vars_)[-3000:]
    dense=norm[:,idx].toarray().astype('float32')
    dense=(dense-dense.mean(0))/(dense.std(0)+1e-6)
    features=PCA(n_components=n_features, random_state=0).fit_transform(dense).astype('float32')
    pos_all=pd.read_csv(OFFICIAL/'data/spatial/tissue_positions_list.csv', header=None, names=['ID','in_tissue','array_row','array_col','pxl_row','pxl_col'])
    pos=order[['ID']].merge(pos_all,on='ID',how='left')
    np.save(feat_p,features); np.save(lab_p,labels); order.drop(columns=['row']).to_csv(meta_p,index=False); pos.to_csv(pos_p,index=False)
    return {'features':features,'labels':labels,'metadata':order.drop(columns=['row']),'positions':pos}

def make_graphs(features, positions):
    g1=nx.from_scipy_sparse_array(kneighbors_graph(positions[['array_row','array_col']].to_numpy(), 6, mode='connectivity', include_self=False).maximum(kneighbors_graph(positions[['array_row','array_col']].to_numpy(), 6, mode='connectivity', include_self=False).T))
    g2=nx.from_scipy_sparse_array(kneighbors_graph(features, 10, mode='connectivity', include_self=False).maximum(kneighbors_graph(features, 10, mode='connectivity', include_self=False).T))
    g3=nx.from_scipy_sparse_array(kneighbors_graph(positions[['pxl_row','pxl_col']].to_numpy(), 8, mode='connectivity', include_self=False).maximum(kneighbors_graph(positions[['pxl_row','pxl_col']].to_numpy(), 8, mode='connectivity', include_self=False).T))
    g4=nx.compose(g1,g2)
    return [g1,g2,g3,g4]

def mi_neighbors(features, k=3):
    mat=kneighbors_graph(features, k+1, mode='connectivity', include_self=True)
    arr=[]
    for i in range(mat.shape[0]):
        inds=mat[i].indices[:k]
        if len(inds)<k: inds=np.pad(inds,(0,k-len(inds)),constant_values=i)
        arr.append(inds)
    return np.asarray(arr,dtype=np.int64)

def latest_commit():
    r=subprocess.run(['git','--git-dir',str(ROOT/'.git_repro'),'--work-tree',str(ROOT),'rev-parse','HEAD'],text=True,capture_output=True)
    return r.stdout.strip() if r.returncode==0 else 'unavailable'
