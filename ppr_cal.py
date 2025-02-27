import os
import os.path as osp
import numpy as np
import networkx as nx
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_sparse import spspmm  # Sparse-sparse matrix multiplication


def compute_ppr(adj_matrix, alpha=0.15, max_iter=100, tol=1e-6):
    """
    计算个性化PageRank (PPR) 矩阵

    参数:
    adj_matrix: 邻接矩阵，可以是稀疏矩阵或密集矩阵
    alpha: 重启概率，默认为0.15
    max_iter: 最大迭代次数，默认为100
    tol: 收敛容差，默认为1e-6

    返回:
    ppr_matrix: 个性化PageRank矩阵，其中ppr_matrix[i,j]表示以节点i为重启点时，节点j的PPR值
    """

    # 确保输入是numpy数组
    # adj_matrix = np.array(adj_matrix, dtype=float)
    n = adj_matrix.shape[0]

    # 计算转移概率矩阵 (列归一化)
    out_degrees = np.sum(adj_matrix, axis=1)

    # 处理出度为0的节点（悬挂节点）
    zero_out_degree = (out_degrees == 0)
    if np.any(zero_out_degree):
        # 对于出度为0的节点，随机跳转到任意节点
        adj_matrix[zero_out_degree, :] = 1.0 / n

    # 归一化邻接矩阵来创建转移概率矩阵
    transition_matrix = adj_matrix / out_degrees[:, np.newaxis]

    # 初始化PPR矩阵，每行是以对应节点为重启点的个性化向量
    ppr_matrix = np.zeros((n, n))

    # 对每个节点计算其作为重启点的PPR向量
    for i in tqdm(range(n)):
        # 创建个性化向量 (重启向量)
        personalization = np.zeros(n)
        personalization[i] = 1.0

        # 初始化PPR向量为均匀分布
        ppr = np.ones(n) / n

        # 幂迭代法计算PPR
        for _ in range(max_iter):
            prev_ppr = ppr.copy()

            # PPR更新公式: (1-alpha) * (转移矩阵 * PPR) + alpha * 个性化向量
            ppr = (1 - alpha) * np.dot(transition_matrix.T, ppr) + \
                alpha * personalization

            # 检查收敛性
            if np.linalg.norm(ppr - prev_ppr, 1) < tol:
                break

        # 保存计算得到的PPR向量到矩阵中
        ppr_matrix[i] = ppr

    return ppr_matrix


def compute_ppr_sparse(adj, alpha=0.85, num_iterations=100, epsilon=1e-6):
    """
    计算基于SparseTensor的Personalized PageRank (PPR)。

    参数:
    - adj (SparseTensor): 图的邻接矩阵表示为SparseTensor。
    - alpha (float): 随机游走的概率，即继续前进的概率。
    - num_iterations (int): 最大迭代次数。
    - epsilon (float): 收敛阈值，当变化小于此值时停止迭代。

    返回:
    - ppr (torch.Tensor): 节点的PPR值。
    """
    n = adj.size(0)
    # 个性化向量，这里我们假设是均匀的
    p = torch.full((n,), 1/n, dtype=torch.float32)
    # 初始PPR值
    ppr = p.clone()

    # 执行幂迭代法计算PPR
    for _ in range(num_iterations):
        # 计算新的PPR值
        new_ppr = alpha * adj @ ppr + (1 - alpha) * p
        # 检查收敛性
        if torch.max(torch.abs(new_ppr - ppr)) < epsilon:
            break
        ppr = new_ppr

    return ppr


def compute_ppr_adj(adj_matrix, alpha=0.15, self_loop_weight=None):
    """
    Compute the Personalized PageRank (PPR) matrix for a graph represented as a SparseTensor.

    :param adj_matrix: SparseTensor representing the adjacency matrix of the graph.
    :param alpha: Teleportation probability, typically set to around 0.15.
    :param self_loop_weight: Weight of the self-loops to be added for regularization. 
                             If None, it is set to 1.0.
    :return: A SparseTensor representing the PPR matrix of the graph.
    """
    num_nodes = adj_matrix.size(0)

    if self_loop_weight is not None:
        # Adding self-loops to the diagonal of the adjacency matrix
        loop_indices = torch.arange(0, num_nodes, dtype=torch.long)
        loop_weight = torch.full(
            (num_nodes,), self_loop_weight, dtype=adj_matrix.dtype())
        adj_matrix += SparseTensor(row=loop_indices, col=loop_indices, value=loop_weight,
                                   sparse_sizes=(num_nodes, num_nodes))

    # Compute the degree of nodes
    degree = adj_matrix.sum(dim=1)
    # Inverse of the degree matrix
    inv_degree = torch.pow(
        degree, -1).fill_(0).where(degree == 0, torch.pow(degree, -1))

    # Transition matrix M
    M = adj_matrix.set_value(adj_matrix.storage.value()
                             * inv_degree[adj_matrix.storage.row()])

    # Personalized PageRank Matrix computation
    I = SparseTensor.eye(
        num_nodes, dtype=adj_matrix.dtype())  # Identity matrix
    # A_hat = alpha * M + (1 - alpha) * I
    A_hat = SparseTensor.from_dense(
        M.to_dense() * alpha) + SparseTensor.from_dense(I.to_dense() * (1 - alpha))
    # Use a sparse matrix multiplication to compute (I - (1-alpha)*M)^-1
    # It's a simplified implementation; for a large graph, iterative methods or approximation might be needed.
    # PPR = spspmm(A_hat, I, num_nodes, num_nodes, num_nodes)
    indexA, valueA = torch.stack(A_hat.coo()[:2]), A_hat.coo()[-1]
    indexB, valueB = torch.stack(I.coo()[:2]), I.coo()[-1]

    m, k, n = num_nodes, num_nodes, num_nodes  # 根据矩阵维度设置
    ind, val = spspmm(indexA, valueA, indexB, valueB, m, k, n, coalesced=True)
    PPR = torch.sparse.FloatTensor(ind, val, size=(num_nodes, num_nodes))

    return PPR.to_dense().numpy()


if __name__ == "__main__":
    # dataset = PygNodePropPredDataset(
    #     name='ogbn-proteins', root='/data/syf/workspace/MultiLblNode/', transform=T.ToSparseTensor(attr='edge_attr'))
    # data = dataset[0]
    # data_name='ogbn-proteins'
    # Move edge features to node features.
    # data.x = data.adj_t.mean(dim=1)
    # data.adj_t.set_value_(None)

    name = 'EukLoc'
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
        edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
        edge_index = torch.hstack((edge_index, edge_index_other_half))

    elif name == 'HumLoc':
        path = "/data/syf/LIP_MLNC/mlncData/HumanGo/"
        edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "edge_list.csv"),
                                               skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
        edge_list_other_half = torch.hstack(
            (edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
        edge_index = torch.transpose(edge_list, 0, 1)
        edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
        edge_index = torch.hstack((edge_index, edge_index_other_half))

    x = torch.tensor(np.genfromtxt(os.path.join(path, "features.csv"),
                                   dtype=np.dtype(float), delimiter=',')).float()

    num_nodes = x.shape[0]
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]),
                       sparse_sizes=(num_nodes, num_nodes)).to_dense().numpy()
    ppr = compute_ppr_adj(adj)
    # ppr = compute_ppr(adj)
    np.save(osp.join("/data/syf/LIP_MLNC/PR/", name+'_PRcooc.npy'), ppr)
