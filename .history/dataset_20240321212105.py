import json
import os
import pickle
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import numpy as np
import torch
from readGrab import readGrabData

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
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
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
        if name == "tfinance":
            graph, label_dict = load_graphs("dataset/tfinance")
            graph = graph[0]
            graph.ndata["label"] = graph.ndata["label"].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs("dataset/tfinance")
                graph = graph[0]
                feat = graph.ndata["feature"].numpy()
                anomaly_id = graph.ndata["label"][:, 1].nonzero().squeeze(1)
                feat = (feat - np.average(feat, 0)) / np.std(feat, 0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata["feature"] = torch.tensor(feat)
                graph.ndata["label"] = graph.ndata["label"].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs("dataset/tfinance")
                graph = graph[0]
                feat = graph.ndata["feature"].numpy()
                anomaly_id = list(graph.ndata["label"][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata["label"][:, 0].nonzero().squeeze(1))
                label = graph.ndata["label"].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random

                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0
            graph.ndata["label"] = graph.ndata["label"].long().squeeze(-1)
        elif name == "tsocial":
            graph, label_dict = load_graphs("dataset/tsocial")
            graph = graph[0]
        # elif name == "yelp":
        #     dataset = FraudYelpDataset()
        #     graph = dataset[0]
        #     if homo:
        #         graph = dgl.to_homogeneous(
        #             dataset[0],
        #             ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
        #         )
        #         graph = dgl.add_self_loop(graph)
        elif name == "amazon":
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(
                    dataset[0],
                    ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
                )
                graph = dgl.add_self_loop(graph)
        elif name == "grab":
            # edge_index, x, lbl = readGrabData()
            edge_index = torch.load(
                "/data/syf/workspace/Rethinking-Anomaly-Detection/GrabEdge.pt"
            )
            x = torch.load("/data/syf/workspace/Rethinking-Anomaly-Detection/GrabX.pt")
            with open(
                "/data/syf/workspace/Rethinking-Anomaly-Detection/GrabLbl.pkl", "rb"
            ) as file:
                lbl = pickle.load(file)
            graph = dgl.graph(
                (edge_index[0], edge_index[1]),
                # ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
            )
            if "label" not in graph.ndata:
                graph.ndata["label"] = lbl[0]
            graph.ndata["feature"] = x
            # dataset=
        elif name == "dblp":
            path = "/data/syf/Datasets/GrabMultiAnomaly/data/dblp/"
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
                (edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1))
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
        elif name in ["delve", "yelp", "blogcatalog", "youtube", "flickr"]:
            path = "/data/syf/Datasets/alldata/" + name + "/"
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
                with open("/data/syf/Datasets/alldata/yelp/class_map.json") as f:
                    class_map = json.load(f)
                    for key, item in class_map.items():
                        ys[int(key)] = item
                lbl = torch.tensor(ys).T
            else:
                lbl = torch.load(path + "labels.pt").T

            graph.ndata["feature"] = x
            if "label" not in graph.ndata:
                graph.ndata["label"] = lbl[0]
        elif name == "delveM":
            labels = sp.load_npz(
                "/data/syf/delve_multi2/delve_multi2_matrices/label.npz"
            ).toarray()
            graph = nx.read_edgelist(
                "/data/syf/delve_multi2/delve_multi2.edgelist",
                delimiter="\t",
            )
            features = torch.FloatTensor(
                sp.load_npz(
                    "/data/syf/delve_multi2/delve_multi2_matrices/lsi_ngram.npz"
                ).toarray()
            )
            
        else:
            print("no such dataset")
            exit(1)

        graph.ndata["label"] = graph.ndata["label"].long().squeeze(-1)
        graph.ndata["feature"] = graph.ndata["feature"].float()
        print(graph)
        self.labels = lbl
        self.graph = graph
