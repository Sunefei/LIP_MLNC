import json
import os
import pickle
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import numpy as np
import torch

import torch.nn as nn

import scipy.sparse as sp
import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
from scipy.sparse import linalg
import networkx as nx
from torch_geometric.data import Data


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(
        transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


class Dataset:
    def __init__(
        self,
        name="tfinance",
        homo=True,
        anomaly_alpha=None,
        anomaly_std=None,
        # lbltype=0,
    ):
        self.name = name
        graph = None
        if name == "dblp":
            path = "/data/syf/LIP_MLNC/mlncData/dblp/"
            labels = np.genfromtxt(
                path + "labels.txt", dtype=np.dtype(float), delimiter=","
            )
            lbl = torch.tensor(labels).float().t()
            x = torch.FloatTensor(
                np.genfromtxt(
                    os.path.join(path, "features.txt"), delimiter=",", dtype=np.float64
                )
            )
            edge_list = torch.tensor(
                np.genfromtxt(os.path.join(path, "dblp.edgelist"))
            ).long()
            edge_list_other_half = torch.hstack(
                (edge_list[:, 1].reshape(-1, 1),
                 edge_list[:, 0].reshape(-1, 1))
            )
            edge_index = torch.transpose(edge_list, 0, 1)
            edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
            edge_index = torch.hstack((edge_index, edge_index_other_half))
            graph = dgl.graph(
                (edge_index[0], edge_index[1]),
                # ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
            )
            graph.ndata["feature"] = x
            if "label" not in graph.ndata:
                graph.ndata["label"] = lbl[0]
        elif name in ["delve", "blogcatalog"]:
            path = "/data/syf/LIP_MLNC/mlncData/" + name + "/"
            edge_index = torch.load(path + "edge_index.pt")
            graph = dgl.graph(
                (edge_index[0], edge_index[1]),
                # ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
            )
            if os.path.exists(path + "feats.pt"):
                x = torch.load(path + "feats.pt")
            else:
                print("No feature!")
            if name == "yelp":
                ys = [-1] * x.size(0)
                with open("/data/syf/LIP_MLNC/yelp/class_map.json") as f:
                    class_map = json.load(f)
                    for key, item in class_map.items():
                        ys[int(key)] = item
                lbl = torch.tensor(ys).T
            else:
                lbl = torch.load(path + "labels.pt").T

            graph.ndata["feature"] = x
            if "label" not in graph.ndata:
                graph.ndata["label"] = lbl[0]
        elif name in ['pcg', 'HumLoc', 'EukLoc']:
            print('Loading dataset ' + name + '.csv...')
            if name == 'pcg':
                path = "/data/syf/LIP_MLNC/mlncData/pcg_removed_isolated_nodes/"
                edges = torch.tensor(np.genfromtxt(os.path.join(path, "edges_undir.csv"),
                                                   dtype=np.dtype(float), delimiter=','))
                edge_index = torch.transpose(edges, 0, 1).long()
            elif name == 'EukLoc':
                path = "/data/syf/LIP_MLNC/mlncData/EukaryoteGo/"
                edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "edge_list.csv"),
                                                       skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
                edge_list_other_half = torch.hstack(
                    (edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
                edge_index = torch.transpose(edge_list, 0, 1)
                edge_index_other_half = torch.transpose(
                    edge_list_other_half, 0, 1)
                edge_index = torch.hstack((edge_index, edge_index_other_half))

            elif name == 'HumLoc':
                path = "/data/syf/LIP_MLNC/mlncData/HumanGo/"
                edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "edge_list.csv"),
                                                       skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
                edge_list_other_half = torch.hstack(
                    (edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
                edge_index = torch.transpose(edge_list, 0, 1)
                edge_index_other_half = torch.transpose(
                    edge_list_other_half, 0, 1)
                edge_index = torch.hstack((edge_index, edge_index_other_half))

            labels = np.genfromtxt(os.path.join(path, "labels.csv"),
                                   dtype=np.dtype(float), delimiter=',')
            lbl = torch.tensor(labels).float().t()
            x = torch.tensor(np.genfromtxt(os.path.join(path, "features.csv"),
                                           dtype=np.dtype(float), delimiter=',')).float()

            graph = dgl.graph(
                (edge_index[0], edge_index[1]),
                num_nodes=x.shape[0]
                # ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
            )
            graph.ndata["feature"] = x
            if "label" not in graph.ndata:
                graph.ndata["label"] = lbl[0]

        else:
            print("no such dataset")
            exit(1)

        graph.ndata["label"] = graph.ndata["label"].long().squeeze(-1)
        graph.ndata["feature"] = graph.ndata["feature"].float()
        print(graph)
        self.labels = lbl
        self.graph = graph


def load_mask(data_name, tt, num_samples):
    if data_name in ['dblp']:
        file_path = os.path.join(
            "/data/syf/LIP_MLNC/mlncData/dblp/dblp_0.6", 'split_{}.pt'.format(tt))
    elif data_name in ['blogcatalog']:
        file_path = os.path.join(
            "/data/syf/LIP_MLNC/mlncData/blogcatalog_0.6", 'split_{}.pt'.format(tt))
    elif data_name in ['pcg']:
        file_path = os.path.join(
            "/data/syf/LIP_MLNC/mlncData/pcg_removed_isolated_nodes_0.6", 'split_{}.pt'.format(tt))
    elif data_name in ['HumLoc']:
        file_path = os.path.join(
            '/data/syf/LIP_MLNC/mlncData/HumanGo/split.pt')
    elif data_name in ['EukLoc']:
        file_path = os.path.join(
            '/data/syf/LIP_MLNC/mlncData/EukaryoteGo/split.pt')

    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
