import argparse
from sklearn.decomposition import PCA
import scanpy as sc
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import os
import time
from typing import Tuple, Union
import anndata
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
from scipy.sparse import issparse
import psutil
from matplotlib.collections import LineCollection

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :][dis_df.loc[index, :] > 0]
        nbs = dis_tmp[0: num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if self_pred in nbs_pred:
            if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
                refined_pred.append(v_c.idxmax())
            else:
                refined_pred.append(self_pred)
        else:
            refined_pred.append(v_c.idxmax())
    return refined_pred

def record_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)
    return memory_usage_gb


from sklearn.neighbors import NearestNeighbors


# from utils import plot_edge_histogram, generate_spatial_weights_fixed_nbrs
def row_normalize(graph: csr_matrix,
                  copy: bool = False,
                  verbose: bool = True):
    """
    Normalize a compressed sparse row (CSR) matrix by row
    """
    if copy:
        graph = graph.copy()

    data = graph.data

    for start_ptr, end_ptr in zip(graph.indptr[:-1], graph.indptr[1:]):

        row_sum = data[start_ptr:end_ptr].sum()

        if row_sum != 0:
            data[start_ptr:end_ptr] /= row_sum

        if verbose:
            print(f"normalized sum from ptr {start_ptr} to {end_ptr} "
                  f"({end_ptr - start_ptr} entries)",
                  np.sum(graph.data[start_ptr:end_ptr]))

    return graph

def remove_greater_than(graph: csr_matrix,
                        threshold: float,
                        copy: bool = False,
                        verbose: bool = True):
    """
    Remove values greater than a threshold from a CSR matrix
    """
    if copy:
        graph = graph.copy()

    greater_indices = np.where(graph.data > threshold)[0]

    if verbose:
        print(f"CSR data field:\n{graph.data}\n"
              f"compressed indices of values > threshold:\n{greater_indices}\n")

    # delete the entries in data and index fields
    # -------------------------------------------

    graph.data = np.delete(graph.data, greater_indices)
    graph.indices = np.delete(graph.indices, greater_indices)

    # update the index pointer
    # ------------------------

    hist, _ = np.histogram(greater_indices, bins=graph.indptr)
    cum_hist = np.cumsum(hist)
    graph.indptr[1:] -= cum_hist

    if verbose:
        print(f"\nCumulative histogram:\n{cum_hist}\n"
              f"\n___ New CSR ___\n"
              f"pointers:\n{graph.indptr}\n"
              f"indices:\n{graph.indices}\n"
              f"data:\n{graph.data}\n")

    return graph
def generate_spatial_distance_graph(
        locations: np.ndarray,
        nbr_object: NearestNeighbors = None,
        num_neighbours: int = None,
        radius: Union[float, int] = None,
) -> csr_matrix:
    if nbr_object is None:
        nbrs = NearestNeighbors(algorithm='ball_tree').fit(locations)
    else:
        nbrs = nbr_object

    if num_neighbours is None:
        return nbrs.radius_neighbors_graph(radius=radius, mode='distance')
    else:
        assert isinstance(num_neighbours, int), (
            f"number of neighbours {num_neighbours} is not an integer"
        )

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbours,
                                          mode="distance")
        if radius is not None:
            assert isinstance(radius, (float, int)), (
                f"Radius {radius} is not an integer or float"
            )

            graph_out = remove_greater_than(graph_out, radius,
                                            copy=False, verbose=False)

        return graph_out


def plot_edge_histogram(graph: csr_matrix,
                        ax: mpl.axes.Axes,
                        title: str = "edge weights",
                        bins: int = 100):
    """
    plot a histogram of the edge-weights a graph
    """
    counts, bins, patches = ax.hist(graph.data, bins=bins)

    median_dist = np.median(graph.data)
    mode_dist = bins[np.argmax(counts)]
    ax.axvline(median_dist, color="r", alpha=0.8)
    ax.axvline(mode_dist, color="g", alpha=0.8)
    ax.set_title("Histogram of " + title)

    print(f"\nEdge weights ({title}): "
          f"median = {median_dist}, mode = {mode_dist}\n")

    return median_dist, mode_dist


def generate_spatial_weights_fixed_nbrs(
        locations: np.ndarray,
        num_neighbours: int = 10,
        decay_type: str = 'reciprocal',
        nbr_object: NearestNeighbors = None,
        verbose: bool = True,
) -> Tuple[csr_matrix, csr_matrix]:
    distance_graph = generate_spatial_distance_graph(
        locations, nbr_object=nbr_object, num_neighbours=num_neighbours, radius=None,
    )
    graph_out = distance_graph.copy()
    if decay_type == 'uniform':
        graph_out.data = np.ones_like(graph_out.data)
    else:
        graph_out.data = 1 / graph_out.data
    return row_normalize(graph_out, verbose=verbose), distance_graph


# from utils import weighted_concatenate, zscore, matrix_to_adata
def matrix_to_adata(matrix, adata: anndata.AnnData) -> anndata.AnnData:
    var_nbrs = adata.var.copy()
    var_nbrs.index += "_nbr"
    nbr_bool = np.zeros((var_nbrs.shape[0] * 2,), dtype=bool)
    nbr_bool[var_nbrs.shape[0]:] = True
    print("num_nbrs:", sum(nbr_bool))

    var_combined = pd.concat([adata.var, var_nbrs])
    var_combined["is_nbr"] = nbr_bool

    return anndata.AnnData(matrix, obs=adata.obs, var=var_combined, uns=adata.uns, obsm=adata.obsm)


def weighted_concatenate(cell_genes: Union[np.ndarray, csr_matrix],
                         neighbours: Union[np.ndarray, csr_matrix],
                         neighbourhood_contribution: float,
                         ) -> Union[np.ndarray, csr_matrix]:
    cell_genes *= np.sqrt(1 - neighbourhood_contribution)
    neighbours *= np.sqrt(neighbourhood_contribution)

    if issparse(cell_genes) and issparse(neighbours):

        return sparse.hstack((cell_genes, neighbours))

    else:  # at least one is a dense array
        if issparse(cell_genes):
            cell_genes = cell_genes.todense()
        elif issparse(neighbours):
            neighbours = neighbours.todense()

        return np.concatenate((cell_genes, neighbours), axis=1)


def zscore(matrix: Union[np.ndarray, csr_matrix],
           axis: int = 0,
           ) -> np.ndarray:
    E_x = matrix.mean(axis=axis)

    if issparse(matrix):
        squared = matrix.copy()
        squared.data **= 2
        E_x2 = squared.mean(axis=axis)
        del squared

    else:

        E_x2 = np.square(matrix).mean(axis=axis)

    variance = E_x2 - np.square(E_x)
    zscored_matrix = (matrix - E_x) / np.sqrt(variance)
    if isinstance(zscored_matrix, np.matrix):
        zscored_matrix = np.array(zscored_matrix)
    zscored_matrix = np.nan_to_num(zscored_matrix)
    return zscored_matrix
import numpy as np
import pandas as pd
from tqdm import trange
import scipy.sparse as sp


def soft_numpy(x, T):
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def create_sppmi_mtx(G, k):
    node_degrees = np.array(G.sum(axis=0)).flatten()
    node_degrees2 = np.array(G.sum(axis=1)).flatten()
    W = np.sum(node_degrees)

    sppmi = G.copy()

    # Use a loop to calculate Wij*W/(di*dj)
    col, row, weights = sp.find(G)
    for i in range(len(col)):
        score = np.log(weights[i] * W / (node_degrees2[col[i]] * node_degrees[row[i]])) - np.log(k)
        sppmi[col[i], row[i]] = max(score, 0)

    return sppmi


def softth(F, lambda_val):
    temp = F.copy()
    U, S, Vt = np.linalg.svd(temp, full_matrices=False)
    Vt = Vt.T

    svp = len(np.flatnonzero(S > lambda_val))
    # svp = np.count_nonzero(S > lambda_val)

    diagS = np.maximum(0, S - lambda_val)

    if svp < 1:
        svp = 1

    E = U[:, :svp] @ np.diag(diagS[:svp]) @ Vt[:, 0: svp].T
    return E


def sparse_self_representation(x, init_w, alpha=1, beta=1):
    # x \in R^{d \times n}
    max_epoch = 100
    n = x.shape[1]
    T1 = np.zeros((n, n))

    C = init_w.copy()
    J1 = C.copy()
    mu = 50
    D = np.diag(np.sum(init_w, axis=0))

    epoch_iter = trange(max_epoch)
    for epoch in epoch_iter:
        C = C * ((x.T @ x + mu * (J1 - np.diag(np.diag(J1))) - T1 + beta * init_w @ C) /
                 (x.T @ x @ C + mu * C + beta * D @ C))
        C[np.isnan(C)] = 0
        C = C - np.diag(np.diag(C))
        J1 = np.array(soft_numpy(C + T1 / mu, alpha / mu))
        J1 = J1 - np.diag(np.diag(J1))
        T1 = T1 + mu * (C - J1)

        #
        err = np.linalg.norm(x - x @ C, 'fro')
        if err < 1e-2:
            break

        epoch_iter.set_description(f"# Epoch {epoch}, loss: {err.item():.3f}")
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    return C


def solve_l1l2(W, lambda_val):
    n = W.shape[0]
    E = W.copy()

    for i in range(n):
        E[i, :] = solve_l2(W[i, :], lambda_val)
    return E


def solve_l2(w, lambda_val):
    nw = np.linalg.norm(w)

    if nw > lambda_val:
        x = (nw - lambda_val) * w / nw
    else:
        x = np.zeros_like(w)
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='')
parser.add_argument('--data_name', type=str, default='v1', help='')
parser.add_argument('--preprocessed_data_path', type=str, default='preprocessed_data', help='')
parser.add_argument('--Data_name', type=str, default='v1', help='')
parser.add_argument('--DLPFC', type=str, default='151509', help='')
parser.add_argument('--min_cells', type=int, default=5, help='')
parser.add_argument('--highly_variable', type=int, default=3000, help='')
parser.add_argument('--highly_variable_adj', type=int, default=2000, help='')
parser.add_argument('--Dim_PCA', type=int, default=1000, help='')
parser.add_argument('--k', type=int, default=10, help='')
parser.add_argument('--Dim_PCA_net', type=int, default=50, help='')
args = parser.parse_args()

data_fold = args.data_path + '/'+ args.data_name + '/'
preprocessed_data_fold = args.preprocessed_data_path + args.data_name + '/'
data_spatial = data_fold + 'spatial/'
if not os.path.exists(args.preprocessed_data_path):
    os.makedirs(args.preprocessed_data_path)
if not os.path.exists(preprocessed_data_fold):
    os.makedirs(preprocessed_data_fold)
if not os.path.exists(preprocessed_data_fold + 'input/'):
    os.makedirs(preprocessed_data_fold + 'input/')


if args.Data_name == "DLPFC":
    adata = sc.read_visium(path=data_fold, count_file=args.data_name + '_filtered_feature_bc_matrix.h5')
    metadata_notNa = pd.read_csv(args.data_path + args.data_name + "/cluster_labels_"+args.DLPFC+".csv")
    label_notNa = pd.Categorical(metadata_notNa['ground_truth']).codes
    adata = adata[label_notNa != -1]
else:
    adata = sc.read_visium(path=data_fold, count_file=args.data_name + '_filtered_feature_bc_matrix.h5')

adata.var_names_make_unique()
adata_srl=adata
# Quality control filtering
sc.pp.filter_genes(adata, min_cells=args.min_cells)
sc.pp.normalize_total(adata, target_sum=1, inplace=False)
sc.pp.log1p(adata)
adata_adj=adata
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=args.highly_variable)
adata_HVG = adata[:, adata.var.highly_variable]
adata_HVG_X = adata_HVG.X

sc.pp.highly_variable_genes(adata_adj, flavor="seurat", n_top_genes=args.highly_variable_adj)
adata_adj_HVG = adata_adj[:, adata_adj.var.highly_variable]
adata_adj_HVG_X = adata_adj_HVG.X

# Data scaling
adata_HVG_X = sc.pp.scale(adata_HVG_X)

# Dimensionality reduction and preservation of cellular genetic features
pca = PCA(n_components=args.Dim_PCA)
pca_data = pca.fit_transform(adata_HVG_X)
inputData = preprocessed_data_fold + 'input/'
if not os.path.exists(inputData):
    os.makedirs(inputData)
np.save(inputData + "features_" + str(args.highly_variable) + "_PCA.npy", pca_data)


# Coordinate network, transcriptome similarity network, and tissue image similarity network
nbrs_xy = NearestNeighbors(n_neighbors=args.k + 1, algorithm='auto').fit(adata.obsm['spatial'])
distances_xy, indices_xy = nbrs_xy.kneighbors(adata.obsm['spatial'])
A1_dict = {}
for i in range(len(indices_xy)):
    A1_dict[i] = indices_xy[i][1:args.k + 1].tolist()
f = open(inputData +'A1_dict.txt', 'w')
f.write(str(A1_dict))
f.close()
A1_mi_3_dict = {}
for i in range(len(indices_xy)):
    A1_mi_3_dict[i] = indices_xy[i][1:4].tolist()
f = open(inputData +'A1_mi_3_dict.txt', 'w')
f.write(str(A1_mi_3_dict))
f.close()
A1_mi_8_dict = {}
for i in range(len(indices_xy)):
    A1_mi_8_dict[i] = indices_xy[i][1:9].tolist()
f = open(inputData +'A1_mi_8_dict.txt', 'w')
f.write(str(A1_mi_8_dict))
f.close()


adata_adj_HVG_X = sc.pp.scale(adata_adj_HVG_X)
# Dimensionality reduction and preservation of cellular genetic features
pca_adj = PCA(n_components=args.Dim_PCA)
pca_data = pca.fit_transform(adata_HVG_X)
inputData = preprocessed_data_fold + 'input/'
if not os.path.exists(inputData):
    os.makedirs(inputData)
np.save(inputData + "features_" + str(args.highly_variable) + "_PCA.npy", pca_data)

pca_ = PCA(n_components=args.Dim_PCA_net)
pcaData = pca_.fit_transform(adata_adj_HVG_X)
nbrs = NearestNeighbors(n_neighbors=args.k + 1, algorithm='ball_tree').fit(pcaData)
distances, indices = nbrs.kneighbors(pcaData)
A2_dict = {}
for i in range(len(indices)):
    A2_dict[i] = indices[i][1:args.k + 1].tolist()
f = open(inputData + 'A2_' + str(args.highly_variable_adj) + '_' + str(args.k) + '_dict.txt', 'w')
f.write(str(A2_dict))
f.close()

np.random.seed(0)
sc.pp.filter_genes(adata_srl, min_cells=10)
sc.pp.highly_variable_genes(adata_srl, flavor='seurat_v3', n_top_genes=3000)
hvg_filter = adata_srl.var['highly_variable']
sc.pp.normalize_total(adata_srl, inplace=True)
adata_all_genes = adata_srl.copy()
adata_srl = adata_srl[:, hvg_filter]
num_neighbours = 6
nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata_srl.obsm['spatial'])
distances, indices = nbrs.kneighbors(n_neighbors=num_neighbours)
median_cell_distance = np.median(distances)
print(f"\nMedian distance to closest cell = {median_cell_distance}")

weights_graph, distance_graph = generate_spatial_weights_fixed_nbrs(adata_srl.obsm['spatial'],
                                                                    num_neighbours=num_neighbours,
                                                                    decay_type='reciprocal', nbr_object=nbrs,
                                                                    verbose=False)

gene_list = adata_srl.var.index
nbrhood_contribution = 0.2
neighbour_agg_matrix = weights_graph @ adata_srl.X
if sparse.issparse(adata_srl.X):
    concatenated = sparse.hstack((adata_srl.X, neighbour_agg_matrix), )
else:
    concatenated = np.concatenate((adata_srl.X, neighbour_agg_matrix), axis=1, )
matrix = weighted_concatenate(zscore(adata_srl.X, axis=0), zscore(neighbour_agg_matrix, axis=0), nbrhood_contribution)

if sparse.issparse(matrix):
    st_dev_pergene = matrix.toarray().std(axis=0)
else:
    st_dev_pergene = matrix.std(axis=0)

enhanced_data = matrix_to_adata(matrix, adata_srl)
start_time = time.time()
sc.pp.pca(enhanced_data, n_comps=100)
low_dim_x = enhanced_data.obsm['X_pca']
sc.pp.pca(adata_srl, n_comps=50)
from sklearn.metrics.pairwise import cosine_similarity

n_spot = low_dim_x.shape[0]
n_neighbor = 6
init_W = cosine_similarity(low_dim_x)
cos_init = np.zeros((n_spot, n_spot))
for i in range(n_spot):
    vec = init_W[i, :]
    distance = vec.argsort()[:: -1]
    for t in range(n_neighbor + 1):
        y = distance[t]
        cos_init[i, y] = init_W[i, y]
C = sparse_self_representation(low_dim_x.T, init_w=cos_init, alpha=1, beta=1)
AC_dict = {}
for i in range(C.shape[0]):
    AC_dict[i] = []
    for j in range(C.shape[1]):
        if C[i, j] != 0:
            AC_dict[i].append(j)
f = open(inputData +'AC_dict.txt', 'w')
f.write(str(AC_dict))
f.close()


# Read label
# If there is NA in DLPFC, replace it
if args.Data_name == 'DLPFC':
    metadata = pd.read_csv(args.data_path + '/'+ args.data_name + "/cluster_labels_" + args.DLPFC + ".csv")
    label = pd.Categorical(metadata['ground_truth']).codes
    label = label[label != -1]
    np.save(inputData + "label.npy", label)
else:
    if args.Data_name == 'V1_Mouse_Brain_Sagittal_Anterior':
        metadata = pd.read_csv(args.data_path + '/'+ args.data_name + "/metadata.tsv", sep="\t")
        metadata.fillna('N____A', inplace=True)  # Fill empty values in DataFrame with 0
        metadata = metadata.rename(columns={'ground_truth': 'fine_annot_type'})
        metadata_list = metadata.iloc[:, 9].tolist()
    else:
        metadata = pd.read_csv(args.data_path + '/'+ args.data_name + "/metadata.tsv", sep="\t")
        metadata_list = metadata.iloc[:, 2].tolist()

    metadata_list = list(set(metadata_list))
    print(len(metadata_list))
    size_mapping = {}
    size_mapping_idx = 0
    for i in metadata_list:
        size_mapping[i] = size_mapping_idx
        size_mapping_idx += 1
    metadata['fine_annot_type'] = metadata['fine_annot_type'].map(size_mapping)
    metadata_array = metadata['fine_annot_type'].to_numpy()
    np.save(inputData + "label.npy", metadata_array)




















