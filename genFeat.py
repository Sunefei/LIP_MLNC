import json
import os
import pickle
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from readGrab import readGrabData

import torch.nn as nn
from dataset import _add_undirected_graph_positional_embedding

for name in ["delve", "yelp", "blogcatalog", "youtube", "flickr"]:
    path = "/data/syf/Datasets/alldata/" + name + "/"
    edge_index = torch.load(path + "edge_index.pt")
    graph = dgl.graph(
        (edge_index[0], edge_index[1]),
        # ndata=["feature", "label", "train_mask", "val_mask", "test_mask"],
    )
    if os.path.exists(path + "feats.pt"):
        x = torch.load(path + "feats.pt")
        print(name + "  have")
    else:
        degrees = graph.in_degrees()
        max_degree = 128
        degree_embedding = nn.Embedding(num_embeddings=max_degree + 1, embedding_dim=32)
        graph = _add_undirected_graph_positional_embedding(graph, 32, retry=10)

        x = torch.cat(
            (
                graph.ndata["pos_undirected"],
                degree_embedding(degrees.clamp(0, max_degree)),
                # graph.ndata["seed"].unsqueeze(1).float(),
            ),
            dim=-1,
        )
        torch.save(x, path + "feats.pt")
