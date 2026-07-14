import argparse,json,networkx as nx,numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from common import ROOT,N_CLUSTERS,build_or_load_processed,ensure_dirs,set_seed,write_json

def save(method,seed,pred,data,extra):
    out=data['metadata'][['ID','annot_type','fine_annot_type']].copy(); out['label']=data['labels']; out['prediction']=pred
    tag=f'{method.lower()}_seed_{seed}'; path=ROOT/f'predictions/{tag}.csv'; out.to_csv(path,index=False)
    m={'method':method,'seed':seed,'ari':float(adjusted_rand_score(data['labels'],pred)),'n_clusters_predicted':int(len(np.unique(pred))),'prediction_file':str(path.relative_to(ROOT)),**extra}; write_json(ROOT/f'metrics/{tag}.json',m); return m

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]); a=p.parse_args(); ensure_dirs(); data=build_or_load_processed(); X=data['features']; res=[]
    for s in a.seeds:
        set_seed(s); res.append(save('KMeans',s,KMeans(n_clusters=N_CLUSTERS,n_init=30,random_state=s).fit_predict(X),data,{'features':'rebuilt PCA'}))
    g=nx.from_scipy_sparse_array(kneighbors_graph(X,15,mode='connectivity',include_self=False).maximum(kneighbors_graph(X,15,mode='connectivity',include_self=False).T)); cands=[]
    for r in np.linspace(0.2,3.0,29):
        comm=nx.algorithms.community.louvain_communities(g,resolution=float(r),seed=a.seeds[0]); pred=np.empty(X.shape[0],int)
        for i,c in enumerate(comm): pred[list(c)]=i
        cands.append((abs(len(comm)-N_CLUSTERS),float(r),len(comm),pred))
    _,r,n,pred=min(cands,key=lambda x:(x[0],x[1])); res.append(save('Louvain',a.seeds[0],pred,data,{'graph':'15-NN PCA graph','selected_resolution':r,'selection_rule':'closest cluster count to 20 before ARI','selected_n_clusters':n}))
    write_json(ROOT/'metrics/baseline_summary.json',res); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
