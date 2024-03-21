import pickle
import dask.dataframe as dd
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import networkx as nx

import matplotlib.pyplot as plt

def compute_ppr(edge_index):
    # 创建图
    # G = nx.Graph()
    # G.add_edges_from(edge_index.T)

    # # 计算xset2中所有节点的Personalized PageRank
    # personalization = {node: 1 if node in xset2 else 0 for node in G.nodes()}
    # ppr_values = nx.pagerank(G, personalization=personalization)

    # # 计算xset1中所有节点的PPR平均值
    # avg_ppr = np.mean([ppr_values[node] for node in xset1])
    
    
    # 将edge_index转换为NetworkX图
    G = nx.Graph()
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)

    # 计算Personalized PageRank
    # 这里使用默认的alpha值(0.85)和默认的个性化向量(均匀分布)
    ppr = nx.pagerank(G, alpha=0.85, personalization=None)

    return ppr

if __name__ == "__main__":
    edge_index = torch.load("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabEdge.pt")
    # x = torch.load("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabX.pt")
    # with open("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabLbl.pkl", "rb") as file:
    #     lbl = pickle.load(file)
    
    
    # node_sets=[]
    # for k in lbl:
    #     node_sets.append(torch.nonzero(lbl[k]))
        
    # ppr_matrix = np.zeros((15, 15))

    # 计算每一对节点组之间的PPR平均值
    # for i in range(15):
    #     for j in tqdm(range(15)):
    #         if i!=j:
    #             ppr_matrix[i, j] = compute_avg_ppr(edge_index, node_sets[i], node_sets[j])
    ppr_matrix=compute_ppr(edge_index)
    # 绘制热力图
    np.save("ppr.npy",ppr_matrix)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(ppr_matrix, cmap="YlGnBu")
    # plt.title("Personalized PageRank Average Values Between Node Groups")
    # plt.xlabel("Node Index")
    # plt.ylabel("Node Index")
    # plt.savefig("ppr.png",dpi=2048)