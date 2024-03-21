import pickle
import dask.dataframe as dd
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import networkx as nx

import matplotlib.pyplot as plt

if __name__ == "__main__":
    edge_index = torch.load("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabEdge.pt")
    x = torch.load("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabX.pt")
    with open("/data0/syf/workspace/Rethinking-Anomaly-Detection/GrabLbl.pkl", "rb") as file:
        lbl = pickle.load(file)
    
        
    # 初始化共现矩阵
    num_labels=15
    co_occurrence_matrix = torch.zeros(num_labels, num_labels)

    # Create a graph
    G = nx.Graph()
    
    # 计算共现矩阵
    for i in range(num_labels):
        for j in range(num_labels):
            if i != j:
                co_occurrence_matrix[i, j] = (
                    ((lbl[i] == 1) & (lbl[j] == 1)).sum().float() /
                    ((lbl[i] == 1) | (lbl[j] == 1)).sum().float()
                )
                if co_occurrence_matrix[i][j] != 0: # Assuming 0 indicates no edge
                    G.add_edge(i, j, weight=co_occurrence_matrix[i][j])


    # 绘制热力图
    sns.heatmap(co_occurrence_matrix)
    # plt.show()
    plt.savefig("avg_coocur.png",dpi=2048)
    plt.clf()

    # 使用spring_layout算法进行布局，以避免边重合和遮盖
    pos = nx.spring_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos)

    # 将边权转换为颜色值
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = np.array(edge_colors)
    # 标准化颜色值到[0, 1]范围
    edge_colors = (edge_colors - edge_colors.min()) / (edge_colors.max() - edge_colors.min())

    # 绘制边，使用热力图颜色
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.hot)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos)

    # 显示图表
    plt.title("Graph Visualization with Edge Weights Represented as Heatmap Colors")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.hot), label='Edge Weight')
    plt.axis('off')
    # # Generate positions for each node
    # pos = nx.spring_layout(G)
    # # pos = nx.kamada_kawai_layout(G)

    # # Extract weights and normalize them for the color map
    # weights = [G[u][v]['weight'] for u, v in G.edges()]
    # max_weight = max(weights)
    # normalized_weights = [w / max_weight for w in weights]
    # normalized_weights = np.array(normalized_weights).astype(float)
    

    # # Draw the nodes
    # nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    # for i, (u, v, data) in enumerate(G.edges(data=True)):
    #     # rad = 0.1 + i * 0.1
    #     # rad = min(rad, np.pi / 2)
    #     # Draw the edges
    #     nx.draw_networkx_edges(
    #         G, pos, edgelist=[(u, v)], width=2,
    #         edge_color=[data['weight']], edge_cmap=plt.cm.plasma,
    #         connectionstyle='arc3,rad=0.1', alpha=0.7
    #         )

    # # Draw the labels
    # nx.draw_networkx_labels(G, pos)
    # # Show the plot
    # plt.title("Weighted Graph with Heatmap Colors")
    # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.plasma), label='Edge Weight')
    # plt.axis('off')
    # plt.show()
    plt.savefig("nxFig_coocur.png",dpi=2048)