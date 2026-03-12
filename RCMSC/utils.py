import os
import torch
import random
import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from metrics import clustering_metrics

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform


def load_data(data_name, current_directory):
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    data_folder_path = os.path.join(parent_directory, './datasets')
    data_path = os.path.join(data_folder_path, data_name)

    data = loadmat(data_path)
    n_samples = np.shape(data['gt'])[0]
    n_views = np.shape(data['X'])[1]

    X = data['X'][0]
    for i in range(n_views):
        if not isinstance(X[i], np.ndarray):
            X[i] = X[i].toarray().astype(np.float32)
        else:
            X[i] = X[i].astype(np.float32)
        X[i] = X[i].transpose()
    return X, data['gt'], n_views, n_samples


def get_soft_cluster_label(C, n_clusters):
    S = 0.5 * (np.abs(C) + np.abs(C.T))
    u, s, v = sp.linalg.svds(S, k=n_clusters, which='LM')
    return u


def get_cluster_results(C, labels, n_clusters):
    u = get_soft_cluster_label(C, n_clusters)
    clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=23).fit(u)
    predict_labels = clustering.labels_
    cm = clustering_metrics(labels, predict_labels)
    acc, nmi, f1, ari = cm.evaluationClusterModelFromLabel_simple()
    return acc, nmi, f1, ari


def data_normalize_l2(X, n_views):
    for i in range(n_views):
        norms = np.linalg.norm(X[i], axis=1)
        X[i] = X[i] / (1e-8 + norms[:, np.newaxis])
    return X


def data_normalize_01(X, n_views):
    for i in range(n_views):
        row_maxes = np.max(X[i], axis=1)
        X[i] = X[i] / row_maxes[:, np.newaxis]
    return X


def single_view_adj_graph(index_adj, n):
    adj_graph = np.zeros((n, n))
    for i in range(n):
        adj_graph[index_adj[:, i], i] = 1
    return adj_graph


def adj_graphs(X, n_samples, k, type):
    positive_adj_graphs = []
    negative_adj_graphs = []
    # for i in range(X.shape[0]):
    for i in range(len(X)):
        if type == "cosine":
            positive_pairs_graph_i = squareform(pdist(X[i], 'cosine'))
            index_adj_i = np.argsort(positive_pairs_graph_i, axis=0)
            positive_pairs_graph_i = single_view_adj_graph(index_adj_i[:k, :], n_samples)
            positive_adj_graphs.append(positive_pairs_graph_i)

        if type == "euclidean":
            positive_pairs_graph_i = squareform(pdist(X[i], 'euclidean'))
            index_adj_i = np.argsort(positive_pairs_graph_i, axis=0)
            positive_pairs_graph_i = single_view_adj_graph(index_adj_i[:k, :], n_samples)
            positive_adj_graphs.append(positive_pairs_graph_i)

    return positive_adj_graphs

# 没有移除自连接的
def get_negative_graph(positive_pairs_graph, n_samples):
    negative_pairs_graph = np.ones((n_samples, n_samples))
    negative_pairs_graph[positive_pairs_graph > 0] = 0
    return negative_pairs_graph


def reformulate_positive_graph(positive_pairs_graph, n_samples):
    positive_pairs_graph = positive_pairs_graph - np.eye(n_samples)
    positive_pairs_graph = positive_pairs_graph / np.sum(positive_pairs_graph, axis=0)
    positive_pairs_graph = positive_pairs_graph + np.eye(n_samples)
    return positive_pairs_graph


def fused_adj_graph(positive_adj_graphs, n_samples, n_views):
    fused_positive_pairs_graph = np.zeros((n_samples, n_samples))
    for i in range(n_views):
        fused_positive_pairs_graph = np.maximum(positive_adj_graphs[i], fused_positive_pairs_graph)

    fused_positive_pairs_graph[fused_positive_pairs_graph > 0] = 1
    fused_positive_pairs_graph = reformulate_positive_graph(fused_positive_pairs_graph, n_samples)
    fused_negative_pairs_graph = get_negative_graph(fused_positive_pairs_graph, n_samples)

    adj_graph = np.stack((fused_positive_pairs_graph.T, fused_negative_pairs_graph.T), axis=0)
    return adj_graph


def get_n_classes(labels):
    n_classes = np.max(np.unique(labels))
    if np.min(np.unique(labels)) == 0:
        n_classes = n_classes + 1
    return n_classes


def write_splitter(data_name, deepmode=False):
    file_path = './results/' + data_name + '_results.txt'
    if deepmode:
        file_path = data_name + '_deep_results.txt'
    with open(file_path, 'a+') as file:
        file.write('\n')
        file.write('Perform Time:' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        file.write('-' * 100 + '\n')

def write_best_results(data_name, temperature, k, q, alpha, beta, gamma, acc_max, nmi_max, f1_max, ari_max, loss, exec_time,
                       deepmode=False):

    file_path = './results/' + data_name + '_results.txt'
    if deepmode:
        file_path = data_name + '_deep_results.txt'

    with open(file_path, 'a+') as file:
        file.write("temperature: " + str(temperature) + ",")
        file.write(" k: " + str(k) + ",")
        file.write(" q: " + str(q) + ",")
        file.write(" alpha: " + ("%0.4f" % alpha) + ",")
        if beta > 0:
            file.write(" beta: " + ("%0.4f" % beta) + ",")
        if gamma > 0:
            file.write(" gamma: " + ("%0.4f" % gamma) + ",")
        file.write(" acc: " + ("%2.2f" % acc_max) + ",")
        file.write(" nmi: " + ("%2.2f" % nmi_max) + ",")
        file.write(" ari: " + ("%2.2f" % ari_max) + ",")
        file.write(" f1: " + ("%2.2f" % f1_max) + ",")
        file.write(" loss: " + ("%4.6f" % loss) + ",")
        file.write(" exec_time: " + ("%0.4f" % exec_time) + "\n")

    # with open('./results/abla_d_a.txt', 'a+') as f:
    #         f.write('{}\n RCMVSC,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} \n'.format(
    #             data_name, acc_max, nmi_max, ari_max, f1_max, exec_time ))
    #         f.flush()
    print('RCMVSC,{:.2f},{:.2f},{:.2f},{:.2f},{:.6f} '.format(
                    acc_max, nmi_max, ari_max, f1_max, exec_time ))


def get_results_variation(data_name):
    current_directory = os.getcwd()
    results_file_path = current_directory + "\\" + data_name + "_results.txt"
    data = []
    with open(results_file_path) as f:
        for line in f.readlines():
            str = line.split('\n')
            ss = str[0]
            ss = ss.split(',')
            data_line = []
            for i in range(len(ss)):
                data_line.append(float(ss[i].split(':')[1]))
            data.append(data_line)
    data = np.array(data)
    return data


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_tsne(data, labels, title, db_name=None):

    # Data standardization
    scaler = StandardScaler()
    total_fusion_scaled = scaler.fit_transform(data)

    # Use t-SNE for dimensionality reduction
    tsne = TSNE(metric='euclidean', random_state=42)
    total_fusion_2d = tsne.fit_transform(total_fusion_scaled)

    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        plt.scatter(total_fusion_2d[labels == label, 0],
                    total_fusion_2d[labels == label, 1],
                    label=f'Class {label}')
    plt.title(f'{title}')

    if db_name:
        save_path = f'./{db_name} {title}.png'
        plt.savefig(save_path)
    else:
        plt.show()

