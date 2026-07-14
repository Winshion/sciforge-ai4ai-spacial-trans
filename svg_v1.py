from scanpy import read_10x_h5
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc
import SpaGCN as spg
import matplotlib.colors as clr
import matplotlib.pyplot as plt

adata = read_10x_h5("data/v1/v1_filtered_feature_bc_matrix.h5")
spatial=pd.read_csv("data/v1/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)

adata.obs["x1"]=spatial[1]
adata.obs["x2"]=spatial[2]
adata.obs["x3"]=spatial[3]
adata.obs["x4"]=spatial[4]
adata.obs["x5"]=spatial[5]
adata.obs["x_array"]=adata.obs["x2"]
adata.obs["y_array"]=adata.obs["x3"]
adata.obs["x_pixel"]=adata.obs["x4"]
adata.obs["y_pixel"]=adata.obs["x5"]

adata=adata[adata.obs["x1"]==1]
adata.var_names=[i.upper() for i in list(adata.var_names)]
adata.var["genename"]=adata.var.index.astype("str")

y_pred=np.load('result/ari_0.6943678004426476.npy')
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')

x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()

#Read in raw data
raw=adata
raw.var_names_make_unique()
raw.obs["pred"]=adata.obs["pred"].astype('category')
raw.obs["x_array"]=raw.obs["x2"]
raw.obs["y_array"]=raw.obs["x3"]
raw.obs["x_pixel"]=raw.obs["x4"]
raw.obs["y_pixel"]=raw.obs["x5"]
raw.X=(raw.X.A if issparse(raw.X) else raw.X)
raw.raw=raw
sc.pp.log1p(raw)



target=2
min_in_group_fraction=0.8
min_in_out_group_ratio=1
min_fold_change=1.9
adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
r=spg.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs["pred"].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)

nbr_domians=spg.find_neighbor_clusters(target_cluster=target,
                                   cell_id=raw.obs.index.tolist(),
                                   x=raw.obs["x_array"].tolist(),
                                   y=raw.obs["y_array"].tolist(),
                                   pred=raw.obs["pred"].tolist(),
                                   radius=r,
                                   ratio=1/2)

nbr_domians=nbr_domians[0:3]
de_genes_info=spg.rank_genes_groups(input_adata=raw,
                                target_cluster=target,
                                nbr_list=nbr_domians,
                                label_col="pred",
                                adj_nbr=True,
                                log=True)
#Filter genes
de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
filtered_info=de_genes_info
filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
                            (filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
                            (filtered_info["in_group_fraction"]>min_in_group_fraction) &
                            (filtered_info["fold_change"]>min_fold_change)]
filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
filtered_info["target_dmain"]=target
filtered_info["neighbors"]=str(nbr_domians)
print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())

print(filtered_info)
filtered_info.to_csv('result/svg/'+str(target)+'.csv', index=True)



color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
for g in filtered_info["genes"].tolist():
    raw.obs["exp"]=raw.X[:,raw.var.index==g]
    ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=g,color_map=color_self,show=False,size=100000/raw.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig("result/svg/plt/"+str(target)+"/"+g+".png", dpi=600)
    plt.close()









