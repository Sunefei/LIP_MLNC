import pickle
import dask.dataframe as dd
import numpy as np
import torch
from tqdm import tqdm


def readEdge(npEdge):
    nodeID = {}
    next_id = 0
    # edge_attr = []
    edge_index = []
    for npa in npEdge:
        if npa[0] not in nodeID:
            nodeID[npa[0]] = next_id
            next_id += 1
        if npa[1] not in nodeID:
            nodeID[npa[1]] = next_id
            next_id += 1
        edge_index.append((nodeID[npa[0]], nodeID[npa[1]]))
        # edge_attr.append(npa[-1].tolist())
    edge_index = torch.tensor(edge_index).T
    return nodeID, edge_index


def readNode(npNode, nodeID):
    nodeXdict = {}
    for nd in npNode:
        if nd[1] in nodeXdict:
            nodeXdict[nd[1]][nodeID[nd[0]]] = torch.tensor(nd[-1])
        else:
            nodeXdict[nd[1]] = {nodeID[nd[0]]: torch.tensor(nd[-1])}

    padX = torch.zeros((len(nodeID), 13))
    for k in nodeXdict.keys():
        data = nodeXdict[k]
        for kid in data:
            padX[kid][: data[kid].shape[0]] = data[kid]
    return padX


def readLbl(npLbl, nodeID):
    lblDict = {}
    for lbl in npLbl:
        lblDict[nodeID[lbl[0]]] = lbl[-1]

    lblRevDict = {}
    for key, value in lblDict.items():
        for val in value:
            if val not in lblRevDict:
                lblRevDict[val] = [
                    key,
                ]
            else:
                lblRevDict[val].append(key)

    mapDict = {
        "Label00": 0,
        "Label01": 1,
        "Label02": 2,
        "Label07": 7,
        "Label05": 5,
        "Label03": 3,
        "Label04": 4,
        "Label06": 6,
        "Label09": 9,
        "Label11": 11,
        "Label10": 10,
        "Label13": 13,
        "Label14": 14,
        "Label12": 12,
        "Label08": 8,
    }
    for key, val in lblDict.items():
        for i in range(len(val)):
            lblDict[key][i] = mapDict[lblDict[key][i]]
    ylabel = {}
    singLbl = {}
    for key, val in lblRevDict.items():
        ylabel[int(key[-2:])] = val
        singLbl[int(key[-2:])] = torch.zeros(len(nodeID))

    for i in tqdm(range(len(nodeID))):
        if i in lblDict:
            for lbl in lblDict[i]:
                singLbl[lbl][i] = 1

    return singLbl


def readGrabData():
    df_edge = dd.read_parquet("/data0/syf/Datasets/small/edges/").compute()
    df_node = dd.read_parquet("/data0/syf/Datasets/small/nodes/").compute()
    df_label = dd.read_parquet("/data0/syf/Datasets/small/labels/").compute()
    npEdge = df_edge.values
    npNode = df_node.values
    npLbl = df_label.values

    nodeID, edge_index = readEdge(npEdge)
    padX = readNode(npNode, nodeID)
    singLbl = readLbl(npLbl, nodeID)
    return edge_index, padX, singLbl


if __name__ == "__main__":
    edge_index, padX, singLbl = readGrabData()
    torch.save(edge_index, "GrabEdge.pt")
    torch.save(padX, "GrabX.pt")
    with open("GrabLbl.pkl", "wb") as file:
        pickle.dump(singLbl, file)
