from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import dgl, networkx as nx, numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from common import ROOT,N_CLUSTERS,build_or_load_processed,ensure_dirs,make_graphs,mi_neighbors,set_seed,write_json
sys.path.insert(0, str(ROOT/'code/mcfst_compat'))
from gae_v4 import GAE, GraphEmbedding, Dis

def allow_zero(m):
    for x in m.modules():
        if hasattr(x,'set_allow_zero_in_degree'): x.set_allow_zero_in_degree(True)

def shuffle(x):
    return x[torch.randperm(x.shape[0], device=x.device)]

def run(seed, epochs, device_name):
    ensure_dirs(); set_seed(seed); data=build_or_load_processed(); X=data['features'].astype('float32'); y=data['labels']; pos=data['positions']; device=torch.device(device_name)
    graphs=[dgl.from_networkx(g).to(device) for g in make_graphs(X,pos)]
    feats=[torch.tensor(X,device=device) for _ in range(4)]
    adjs=[g.adjacency_matrix().to_dense().to(device) for g in graphs]
    edges=sum(g.number_of_edges() for g in graphs); mi=mi_neighbors(X,3)
    model=GAE([X.shape[1]]*4,[64,32],[32,10],N_CLUSTERS,4).to(device); allow_zero(model)
    model_g=GraphEmbedding(X.shape[0], X.shape[0]//2, N_CLUSTERS).to(device); model_d=Dis(latent_dim=10).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.01); optg=torch.optim.Adam(model_g.parameters(),lr=0.01)
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(adjs[0].shape[0]**2-edges/2)/edges*2],device=device)); mse=torch.nn.MSELoss()
    km=KMeans(n_clusters=N_CLUSTERS,n_init=30,random_state=seed); rows=[]; start=time.time(); graph=None
    for ep in range(epochs):
        ar=model_g(sum(adjs)); loss_g=sum(mse(ar,a) for a in adjs)/4; optg.zero_grad(); loss_g.backward(); optg.step()
        ap=torch.round(torch.clamp(ar,0,1)+0.1).detach().cpu().numpy(); ap=ap+ap.T; graph=dgl.from_networkx(nx.from_numpy_array(ap)).to(device)
        logits,h=model(graphs[0],graphs[1],graphs[2],graphs[3],feats[0],feats[1],feats[2],feats[3],graph)
        loss_rec=sum(bce(logits[i],adjs[i]) for i in range(4))/4
        gil=torch.tensor(0.,device=device)
        for j in range(mi.shape[1]):
            hs=shuffle(h); gil += -torch.mean(torch.log(model_d(torch.cat((h,h[mi[:,j]]),1))+1e-6)+torch.log(1-model_d(torch.cat((h,hs),1))+1e-6))
        z=h.detach().cpu().numpy(); pred_ep=km.fit_predict(z); ari_ep=adjusted_rand_score(y,pred_ep)
        mu=km.cluster_centers_; q=1/(1+np.sum((z[:,None,:]-mu[None,:,:])**2,axis=2)); q=q/q.sum(1,keepdims=True); q=torch.tensor(q,dtype=torch.float32,device=device); p=(q*q/q.sum(0)); p=(p.t()/p.sum(1)).t(); kl=torch.mean((p-q)**2)*5000
        loss=loss_rec+0.0005*gil+kl; opt.zero_grad(); loss.backward(); opt.step()
        rows.append({'epoch':ep+1,'ari':float(ari_ep),'loss':float(loss.detach().cpu()),'loss_rec':float(loss_rec.detach().cpu()),'loss_g':float(loss_g.detach().cpu()),'elapsed_sec':time.time()-start})
        print(json.dumps(rows[-1]), flush=True)
    model.eval();
    with torch.no_grad(): _,z=model(graphs[0],graphs[1],graphs[2],graphs[3],feats[0],feats[1],feats[2],feats[3],graph)
    pred=KMeans(n_clusters=N_CLUSTERS,n_init=30,random_state=seed).fit_predict(z.cpu().numpy()); ari=float(adjusted_rand_score(y,pred))
    rdir=ROOT/f'runs/mcfst_seed_{seed}'; rdir.mkdir(parents=True,exist_ok=True)
    out=data['metadata'][['ID','annot_type','fine_annot_type']].copy(); out['label']=y; out['prediction']=pred
    pred_path=ROOT/f'predictions/mcfst_seed_{seed}.csv'; out.to_csv(pred_path,index=False); np.save(rdir/'embedding.npy', z.cpu().numpy())
    metrics={'method':'MCFST','seed':seed,'epochs':epochs,'device':device_name,'ari':ari,'n_spots':int(len(y)),'n_clusters':N_CLUSTERS,'prediction_file':str(pred_path.relative_to(ROOT)),'epoch_records':rows}
    write_json(rdir/'metrics.json',metrics); write_json(ROOT/f'metrics/mcfst_seed_{seed}.json',metrics); return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]); p.add_argument('--epochs',type=int,default=130); p.add_argument('--device',default='cpu'); a=p.parse_args()
    res=[run(s,a.epochs,a.device) for s in a.seeds]; write_json(ROOT/'metrics/mcfst_summary.json',res); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
