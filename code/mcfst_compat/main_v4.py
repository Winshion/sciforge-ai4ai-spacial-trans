import os
os.environ["OMP_NUM_THREADS"] = '11'
import argparse
import torch
import dgl
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from gae_v4 import GAE, GraphEmbedding, Dis
from dataset import *
from utils import *
from time import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy import linalg as LA
from sklearn.preprocessing import normalize
import csv
import process
from sklearn.metrics import adjusted_rand_score as ari_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import clip_grad_value_
import random


parser = argparse.ArgumentParser(description='GAE')
parser.add_argument('--train_epochs', '-te', type=int, default=130, help='number of train_epochs')
parser.add_argument('--save_dir', '-s', type=str, default='./result', help='result directry')
parser.add_argument('--hidden_dimsV', type=int, nargs='+', default=[64, 32], help='')
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 10], help='')
parser.add_argument('--heads', type=int, nargs='+', default=[4, 4], help='')
parser.add_argument('--tlr', type=float, default=0.01, help='Adam learning rate')
parser.add_argument('--lambda1', type=float, default=0.0005)
parser.add_argument('--lambda2', type=float, default=1)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda:0" if args.cuda else "cpu")
print(args)


def main():

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load and preprocess dataset
    feature_spot, graph_dict_1, graph_dict_2, graph_dict_3, graph_dict_4, mi_dict1, mi_dict2, mi_dict3, mi_dict4, label_label = process.load_data()
    views = 4
    print('Number of Samples: {:02d}'.format(feature_spot.shape[0]))
    print('Views:', views)

    feature0 = torch.FloatTensor(feature_spot)
    feature1 = torch.FloatTensor(feature_spot)
    feature2 = torch.FloatTensor(feature_spot)
    feature3 = torch.FloatTensor(feature_spot)

    print('VIEW-0 DateSize: {:02d} * {:02d} '.format(feature0.shape[0], feature0.shape[1]))
    print('VIEW-1 DateSize: {:02d} * {:02d} '.format(feature1.shape[0], feature1.shape[1]))
    print('VIEW-2 DateSize: {:02d} * {:02d} '.format(feature2.shape[0], feature2.shape[1]))
    print('VIEW-3 DateSize: {:02d} * {:02d} '.format(feature3.shape[0], feature3.shape[1]))

    in_feats = [feature0.shape[1], feature1.shape[1], feature2.shape[1],feature3.shape[1]]

    graph0 = dgl.from_networkx(graph_dict_1)
    graph1 = dgl.from_networkx(graph_dict_2)
    graph2 = dgl.from_networkx(graph_dict_3)
    graph3 = dgl.from_networkx(graph_dict_4)

    print('VIEW-0 Edges: {:02d} '.format(graph0.number_of_edges()))
    print('VIEW-1 Edges: {:02d} '.format(graph1.number_of_edges()))
    print('VIEW-2 Edges: {:02d} '.format(graph2.number_of_edges()))
    print('VIEW-3 Edges: {:02d} '.format(graph3.number_of_edges()))

    mik0 = mi_dict1
    mik1 = mi_dict2
    mik2 = mi_dict3
    mik3 = mi_dict4
    mik = np.hstack((mik0, mik1, mik2, mik3))
    print('Mutual Information Matrix Size: ', mik.shape)

    degs0 = graph0.in_degrees().float()
    degs1 = graph1.in_degrees().float()
    degs2 = graph2.in_degrees().float()
    degs3 = graph3.in_degrees().float()

    norm0 = torch.pow(degs0, -0.5)
    norm1 = torch.pow(degs1, -0.5)
    norm2 = torch.pow(degs2, -0.5)
    norm3 = torch.pow(degs3, -0.5)

    norm0[torch.isinf(norm0)] = 0
    norm1[torch.isinf(norm1)] = 0
    norm2[torch.isinf(norm2)] = 0
    norm3[torch.isinf(norm3)] = 0


    graph0.ndata['norm'] = norm0.unsqueeze(1)
    graph1.ndata['norm'] = norm1.unsqueeze(1)
    graph2.ndata['norm'] = norm2.unsqueeze(1)
    graph3.ndata['norm'] = norm3.unsqueeze(1)

    adj0 = graph0.adjacency_matrix().to_dense()
    adj1 = graph1.adjacency_matrix().to_dense()
    adj2 = graph2.adjacency_matrix().to_dense()
    adj3 = graph3.adjacency_matrix().to_dense()

    edges = graph0.number_of_edges() + graph1.number_of_edges() + graph2.number_of_edges() + graph3.number_of_edges()

    y = label_label
    n_clusters = len(np.unique(y))
    print(n_clusters)

    # model
    model = GAE(in_feats, args.hidden_dimsV, args.hidden_dims, n_clusters, views)

    # print(model)
    model.train()
    model_g = GraphEmbedding(feature0.shape[0], int(feature0.shape[0] / 2), n_clusters)
    model_g.train()
    model_d = Dis(latent_dim=args.hidden_dims[-1])
    # optimizer
    optim_gae_t = torch.optim.Adam(model.parameters(), lr=args.tlr)
    optim_ge_t = torch.optim.Adam(model_g.parameters(), lr=args.tlr)

    # loss
    pos_weight = torch.Tensor([float(graph0.adjacency_matrix().to_dense().shape[0] ** 2 - edges / 2) / edges * 2])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_m = torch.nn.MSELoss()

    # To GPU
    criterion.cuda(device=device)
    criterion_m.cuda(device=device)
    model = model.to(device)
    model_g = model_g.to(device)
    model_d = model_d.to(device)

    graph0 = graph0.to(device)
    graph1 = graph1.to(device)
    graph2 = graph2.to(device)
    graph3 = graph3.to(device)

    feature0 = feature0.to(device)
    feature1 = feature1.to(device)
    feature2 = feature2.to(device)
    feature3 = feature3.to(device)

    adj0 = adj0.to(device)
    adj1 = adj1.to(device)
    adj2 = adj2.to(device)
    adj3 = adj3.to(device)

    begin_time = time()
    kmeans = KMeans(n_clusters=n_clusters,n_init=30)

    mu = torch.zeros([n_clusters, 128], requires_grad=True)
    alpha = 1

    learning_all = []
    # Training
    print('Training Start')
    for epoch in range(args.train_epochs):

        # train GFN
        adjin = adj0
        adjin = torch.add(adjin, adj1)
        adjin = torch.add(adjin, adj2)
        adjin = torch.add(adjin, adj3)

        adj_r = model_g.forward(adjin)
        loss_gre = (criterion_m(adj_r, adj0) + criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2) + criterion_m(adj_r, adj3)) / views
        loss_ge = loss_gre
        optim_ge_t.zero_grad()
        loss_ge.backward()
        optim_ge_t.step()


        # normalization
        adj_p = torch.clamp(adj_r, 0, 1)
        adj_p = torch.round(adj_p + 0.1)
        # build symmetric adjacency matrix
        adj_pn = adj_p.detach().cpu().numpy()
        adj_pn += adj_pn.T
        graph = nx.from_numpy_array(adj_pn)
        graph = dgl.from_networkx(graph)
        graph = graph.to(device)

        # Train GAE
        adj_logits, h = model.forward(graph0, graph1, graph2,  graph3,feature0, feature1, feature2, feature3, graph)
        loss_rec = (criterion(adj_logits[0], adj0) + criterion(adj_logits[1], adj1) + criterion(adj_logits[2], adj2) + criterion(adj_logits[3], adj3)) / views
        global_info_loss = 0
        for i in range(mik.shape[1]):
            h_shuffle = shuffling(h, latent=args.hidden_dims[-1])
            h_h_shuffle = torch.cat((h, h_shuffle), 1)
            h_h_shuffle_scores = model_d(h_h_shuffle)
            h_h = torch.cat((h, h[mik[:, i]]), 1)
            h_h_scores = model_d(h_h)
            global_info_loss += - torch.mean(torch.log(h_h_scores + 1e-6) + torch.log(1 - h_h_shuffle_scores + 1e-6))






        y_pred_h = kmeans.fit_predict(h.data.cpu().numpy())
        ari_h = eva_only(y, y_pred_h)

        kmeans.fit(h.data.cpu().numpy())
        mu = kmeans.cluster_centers_
        q = _soft_assignment(n_clusters, feature0.shape[0], alpha, h.data.cpu().numpy(), mu)
        p = target_distribution(q)

        kl_loss = _kl_divergence(p, q) * 5000

        loss_gae = loss_rec + args.lambda1 * global_info_loss + args.lambda2 * kl_loss


        optim_gae_t.zero_grad()
        loss_gae.backward()
        optim_gae_t.step()

        learning_all.append(optim_gae_t.state_dict()['param_groups'][0]['lr'])



        if (epoch + 1) % 5 == 0:
            end_time = time()
            run_time = end_time - begin_time
            print(
                'Epoch: {:02d} | GAE-Loss: {:.5f} + {:.5f} + {:.5f} =  {:.5f}| GE-Loss: {:.5f}  =  {:.5f} | Time: {:.2f}'.format(
                    epoch + 1, loss_rec, args.lambda1 * global_info_loss, kl_loss, loss_gae, loss_gre, loss_ge,
                    run_time))
            print('ari' + str(ari_h))
            print(learning_all)
            learning_all = []


    model.eval()
    _, z = model.forward(graph0, graph1, graph2, graph3, feature0, feature1, feature2, feature3, graph)


    y_pred = kmeans.fit_predict(z.data.cpu().numpy())


    ari = eva(y, y_pred)
    np.save('result/ari_' + str(ari) + '.npy',y_pred)
    return ari


def shuffling(x, latent):
    idxs = torch.arange(0, x.shape[0]).cuda()
    a = torch.randperm(idxs.size(0)).cuda()
    aa = idxs[a].unsqueeze(1)
    aaa = aa.repeat(1, latent)
    return torch.gather(x, 0, aaa)


def _kl_divergence(target, pred):
    return torch.mean((target - pred) ** 2)

def target_distribution(q):
    p = q ** 2 / q.sum(axis=0)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def _soft_assignment(n_cluster,input_batch_size,alpha,embeddings, cluster_centers):
    def _pairwise_euclidean_distance(a, b):
        p1 = torch.matmul(
            torch.unsqueeze(torch.sum(a**2, 1), 1),
            torch.ones(1, n_cluster)
        )

        p2 = torch.permute(torch.matmul(

            torch.reshape(torch.sum(b**2, dim=1), (-1, 1)),
            torch.ones(1,input_batch_size)), (1, 0))

        b_t=torch.permute(b,(1, 0))
        xx = torch.add(p1, p2) - torch.matmul(a, b_t)
        xxx=xx.numpy()
        for i in range(xxx.shape[0]):
            for j in range(xxx.shape[1]):
                if xxx[i,j]<0:
                    print("xiao yu 0")
                    xxx[i, j]=1e-8
        xxxx=torch.from_numpy(xxx)
        res = torch.sqrt(xxxx)
        return res

    embeddings=torch.from_numpy(embeddings)
    cluster_centers = torch.from_numpy(cluster_centers)
    dist = _pairwise_euclidean_distance(embeddings, cluster_centers)

    q = 1.0 / (1.0 + dist ** 2 / alpha) ** ((alpha + 1.0) / 2.0)

    q = (q / torch.sum(q, dim=1, keepdims=True))

    return q


def purity_func(cluster, label):
    cluster = np.array(cluster)
    label = np.array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)
    return sum(np.max(count_all, axis=0)) / len(cluster)

if __name__ == '__main__':

    ariA = []
    for i in range(10):
        ari= main()
        ariA.append(ari)
    print('ARI: ave|{:04f} std|{:04f}'.format(np.mean(ariA), np.std(ariA, ddof=1)))
    print(str(ariA))