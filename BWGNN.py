import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import (
    GraphConv,
    EdgeWeightNorm,
    ChebConv,
    # GATConv,
    HeteroGraphConv,
    APPNPConv,
)

from torch_geometric.nn import GCNConv,GATConv


class PolyConv(nn.Module):
    def __init__(
        self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False
    ):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """Operation Feat * D^-1/2 A D^-1/2"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = (
                torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(
        self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False
    ):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """Operation Feat * D^-1/2 A D^-1/2"""
            block.srcdata["h"] = feat * D_invsqrt
            block.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - block.srcdata.pop("h") * D_invsqrt

        with block.local_scope():
            D_invsqrt = (
                torch.pow(block.out_degrees().float().clamp(min=1), -0.5)
                .unsqueeze(-1)
                .to(feat.device)
            )
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k] * feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols("x")
    for i in range(d + 1):
        f = sympy.poly(
            (x / 2) ** i
            * (1 - x / 2) ** (d - i)
            / (scipy.special.beta(i + 1, d + 1 - i))
        )
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, graph, num_lbls=1):
        # def __init__(self, in_feats, h_feats, num_classes, graph, num_lbls=1):
        super().__init__()
        self.gnn1 = GATConv(in_channels, hidden_channels * 2, dropout=0.2)
        self.gnn2 = GATConv(hidden_channels * 2, hidden_channels, dropout=0.2)

        self.num_lbls = num_lbls
        self.graph = graph
        self.mlps = nn.ModuleList()
        for i in range(self.num_lbls):
            self.mlps.append(
                SimpleMLP(hidden_channels, hidden_channels // 2, out_channels)
            )

    def forward(self, x, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        edge_index = torch.stack(self.graph.edges())
        x = self.gnn1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        h_last = self.gnn2(x, edge_index, edge_weight)

        outputs = [mlp(h_last) for mlp in self.mlps]
        return outputs, h_last


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, graph, num_lbls=1):
        # def __init__(self, in_feats, h_feats, num_classes, graph, num_lbls=1):
        super().__init__()
        self.gnn1 = GCNConv(in_channels, hidden_channels * 2, normalize=True)
        self.gnn2 = GCNConv(hidden_channels * 2, hidden_channels, normalize=True)

        self.num_lbls = num_lbls
        self.graph = graph
        self.mlps = nn.ModuleList()
        for i in range(self.num_lbls):
            self.mlps.append(
                SimpleMLP(hidden_channels, hidden_channels // 2, out_channels)
            )

    def forward(self, x, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        edge_index = torch.stack(self.graph.edges())
        x = self.gnn1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        h_last = self.gnn2(x, edge_index, edge_weight)

        outputs = [mlp(h_last) for mlp in self.mlps]
        return outputs, h_last


# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes, graph, num_lbls=1):
#         super(GCN, self).__init__()
#         self.graph = graph
#         self.gnn1 = dgl.nn.GraphConv(in_feats, h_feats * 2)
#         self.gnn2 = dgl.nn.GraphConv(h_feats * 2, h_feats)

#         self.num_lbls = num_lbls
#         self.mlps = nn.ModuleList()
#         for i in range(self.num_lbls):
#             self.mlps.append(SimpleMLP(h_feats, h_feats // 2, num_classes))

#     def forward(self, input_feat):
#         h = self.gnn1(self.graph, input_feat)
#         h = F.relu(h)
#         h_last = self.gnn2(self.graph, h)

#         outputs = [mlp(h_last) for mlp in self.mlps]
#         return outputs, h_last


class MultiAPPNP(nn.Module):
    def __init__(
        self, in_feats, h_feats, num_classes, graph, num_lbls=1, K=10, alpha=0.1
    ):
        super(MultiAPPNP, self).__init__()
        self.graph = graph
        self.gnn1 = APPNPConv(K, alpha)
        self.gnn2 = APPNPConv(K, alpha)
        self.num_lbls = num_lbls
        self.mlps = nn.ModuleList()

        self.mlp = SimpleMLP(in_feats, h_feats, h_feats)
        for i in range(self.num_lbls):
            self.mlps.append(SimpleMLP(h_feats, h_feats // 2, num_classes))

    def forward(self, input_feat):
        h = self.gnn1(self.graph, input_feat)
        h_last = self.gnn2(self.graph, h)
        h_last = self.mlp(h_last)
        outputs = [mlp(h_last) for mlp in self.mlps]
        return outputs, h_last

    # def reset_parameters(self):
    #     self.gnn1.reset_parameters()
    #     self.gnn2.reset_parameters()


class BWGNN(nn.Module):
    def __init__(
        self, in_feats, h_feats, num_classes, graph, d=2, batch=False, num_lbls=1
    ):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList()
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
            else:
                self.conv.append(
                    PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False)
                )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

        self.num_lbls = num_lbls
        self.mlps = nn.ModuleList()
        for i in range(self.num_lbls):
            self.mlps.append(SimpleMLP(h_feats, h_feats // 2, num_classes))

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=in_feat.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h_last = self.act(h)

        # h = self.linear4(h_last)
        outputs = [mlp(h_last) for mlp in self.mlps]
        return outputs, h_last

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h


# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.conv = [
            PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas
        ]
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.LeakyReLU()
        # print(self.thetas)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        for relation in self.g.canonical_etypes:
            # print(relation)
            h_final = torch.zeros([len(in_feat), 0])
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
                # print(h_final.shape)
            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
