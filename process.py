import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
from scipy import sparse







def load_data():
    datafolder = 'preprocessed_datav1/input/'
    name1 = ['A1_dict.txt', 'A2_2000_10_dict.txt','A1_dict.txt', 'AC_dict.txt']
    name2 = ['A1_mi_8_dict.txt', 'A1_mi_8_dict.txt','A1_mi_3_dict.txt', 'A1_mi_3_dict.txt']
    graph_ = {}
    mi_graph = {}
    for i in name1:
        f = open(datafolder + i, 'r')
        a = f.read()
        graph_[i] = eval(a)
        f.close()
    graph1 = graph_[name1[0]]
    graph2 = graph_[name1[1]]
    graph3 = graph_[name1[2]]
    graph4 = graph_[name1[3]]

    for i in name2:
        f = open(datafolder + i, 'r')
        a = f.read()
        mi_graph[i] = eval(a)
        f.close()
    mi_graph1 = mi_graph[name2[0]]
    mi_graph2 = mi_graph[name2[1]]
    mi_graph3 = mi_graph[name2[2]]
    mi_graph4 = mi_graph[name2[3]]

    features = np.load(datafolder+'features_3000_PCA.npy')
    label = np.load(datafolder+'label.npy')


    nx_graph1 = nx.from_dict_of_lists(graph1)
    nx_graph2 = nx.from_dict_of_lists(graph2)
    nx_graph3 = nx.from_dict_of_lists(graph3)
    nx_graph4 = nx.from_dict_of_lists(graph4)

    mi_graph1_arr=[value for value in mi_graph1.values()]
    mi_graph2_arr = [value for value in mi_graph2.values()]
    mi_graph3_arr = [value for value in mi_graph3.values()]
    mi_graph4_arr = [value for value in mi_graph4.values()]

    return features, nx_graph1, nx_graph2, nx_graph3, nx_graph4, mi_graph1_arr, mi_graph2_arr,mi_graph3_arr, mi_graph4_arr, label



load_data()