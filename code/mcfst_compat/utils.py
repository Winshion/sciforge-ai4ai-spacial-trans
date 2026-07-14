import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
import os



def eva(y_true, y_pred):
    ari = ari_score(y_true, y_pred)
    print('ari {:.5f}'.format(ari))
    return ari

def eva_only(y_true, y_pred):
    ari = ari_score(y_true, y_pred)
    return  ari

