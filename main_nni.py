import os
import warnings
import dgl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from dataset import Dataset, load_mask
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from AllModel import *
from tqdm import tqdm
import networkx as nx
from utils import cosine_similarity
import nni


def Lbltrace(A, Y):
    D = torch.diag(A.sum(1))
    L = D - A
    YLY_t = Y @ L @ Y.T
    trace_YLY_t = torch.trace(YLY_t)
    return trace_YLY_t


def train(model, g, args, masks, labels, cooc, tasks):
    features = g.ndata["feature"].to(args.device)
    num_lbls = len(tasks)
    label_rest = {}
    coefs = []
    train_mask, val_mask, test_mask = masks
    if args.learnCoef == "auto":
        coefs = [torch.nn.Parameter(torch.ones(
            1, requires_grad=True, device=args.device)) for _ in range(num_lbls)]
        coefs = torch.cat([coef.to(args.device) for coef in coefs], dim=0)
        optimizer = torch.optim.Adam(
            coefs + list(model.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(
            list(model.parameters()), lr=args.lr, weight_decay=args.wd)

    best_ap_, avg_final_tauc = 0.0, 0.0
    time_start = time.time()
    model_parameters = model.get_gnn_parameters()
    pbar = tqdm(range(args.epoch))

    # 早停机制参数
    patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    for e in tqdm(range(args.epoch)):
        model.train()
        logits, h = model(features)
        logits_all = torch.stack(logits).permute(1, 0, 2)

        loss = 0.0
        argmax_list = []
        loss_list = []
        grads_loss = []

        losses = torch.zeros(labels.shape[1], device=args.device)
        for no in range(labels.shape[1]):
            label_rest = labels[:, no]
            out = logits_all[:, no, :]
            weight = (1 - label_rest[train_mask]).sum().item() / \
                label_rest[train_mask].sum().item()
            losses[no] = F.cross_entropy(
                out[train_mask],
                label_rest[train_mask].to(torch.long).to(args.device),
                weight=torch.tensor([1.0, weight], device=args.device),
            )

        if args.learnCoef == "auto":
            loss = torch.sum(coefs * losses, dim=-1)
        elif args.learnCoef == "none":
            loss = torch.sum(losses, dim=-1)
        elif args.learnCoef in ["our", "grad"]:
            for ls in losses:
                gw_real = torch.autograd.grad(
                    ls, model_parameters, retain_graph=True)
                gw_real = list((_.detach().clone() for _ in gw_real))
                grads_loss.append(gw_real)

        if args.learnCoef in ["our", "grad", "cooc"]:
            if args.learnCoef != "cooc":
                cos_similarities = torch.zeros(
                    (len(grads_loss), len(grads_loss)))
                for i in range(len(grads_loss) - 1):
                    for j in range(i + 1, len(grads_loss)):
                        for kk, name in enumerate(grads_loss[i]):
                            cos_sim = cosine_similarity(
                                grads_loss[i][kk], grads_loss[j][kk])
                            cos_similarities[i][j] += cos_sim.item()
                            cos_similarities[j][i] += cos_sim.item()

            if args.learnCoef == "grad":
                cooc = cos_similarities
            elif args.learnCoef == "our":
                cooc *= cos_similarities

            coocF = F.softmax(cooc, dim=0)
            num_task = "&".join(str(num) for num in tasks)
            G_ourPR = nx.from_numpy_array(coocF.numpy())
            outPGPage = nx.pagerank(G_ourPR)
            for ii, loss_i in enumerate(losses):
                loss += loss_i * outPGPage[ii]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits_all.cpu().detach()[:, :, -1]

        val_ap = average_precision_score(
            labels[val_mask], probs[val_mask].numpy())
        num_sample = labels[val_mask].sum(0)
        if 0 in num_sample:
            indW = torch.where(num_sample == 0)[0].item()
            labels_new = torch.cat(
                (labels[val_mask][:, :indW], labels[val_mask][:, indW+1:]), dim=1)
            probs_new = torch.cat(
                (probs[val_mask][:, :indW], probs[val_mask][:, indW+1:]), dim=1)
            val_auc = roc_auc_score(labels_new, probs_new)
        else:
            val_auc = roc_auc_score(
                labels[val_mask], probs[val_mask].numpy())

        if val_ap > best_ap_:
            best_ap_ = val_ap
            test_ap = average_precision_score(
                labels[test_mask], probs[test_mask].numpy())
            test_auc = roc_auc_score(
                labels[test_mask], probs[test_mask].numpy())

        pbar.set_postfix(Epoch=e, loss=loss.cpu().item(), test_auc=test_auc,
                         test_ap=test_ap, best_ap=best_ap_, val_auc=val_auc, val_ap=val_ap)

        # 早停机制
        val_loss = loss.cpu().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        nni.report_intermediate_result(test_ap)

    nni.report_final_result(test_ap)
    time_end = time.time()
    print(f"Test: AUC {test_auc * 100:.2f}, AP {test_ap * 100:.2f}")
    return test_ap, test_auc


def main(args):
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    data = Dataset(dataset_name, homo)
    graph = data.graph.to(args.device)
    all_labels = data.labels

    args.lbls = list(range(all_labels.shape[0]))
    in_feats = graph.ndata["feature"].shape[1]
    num_classes = 2

    path = "/data/syf/LIP_MLNC/PR/"
    if args.learnCoef in ["cooc", "our"]:
        coocPage = np.load(path + dataset_name +
                           "_PRcooc.npy", allow_pickle=True)
        indexes = args.lbls
        coocPage = torch.tensor(coocPage[np.ix_(indexes, indexes)])
    else:
        coocPage = None

    using_lbl = all_labels.T

    activation = F.relu if args.activation == "relu" else F.leaky_relu

    if args.run == 1:
        tt = args.run
        if args.model_type == "homo":
            model = BWGNN(in_feats, h_feats, num_classes, graph, args.dropout,
                          d=order, num_lbls=len(args.lbls)).to(args.device)
        elif args.model_type == "gcn":
            model = GCN(in_feats, h_feats, num_classes, graph, args.dropout,
                        num_layers=args.num_layers, activation=activation, num_lbls=len(args.lbls)).to(args.device)
        elif args.model_type == "appnp":
            model = MultiAPPNP(in_feats, h_feats, num_classes, args.dropout,
                               graph, num_lbls=len(args.lbls)).to(args.device)
        elif args.model_type == "hetero":
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, args.dropout,
                                 d=order, num_lbls=len(args.lbls)).to(args.device)
        else:
            print("model wrong")
        masks = load_mask(args.dataset, tt, graph.ndata["feature"].shape[0])
        ap, auc = train(model, graph, args, masks,
                        using_lbl, coocPage, args.lbls)
        print(ap, auc)
    else:
        final_ap, final_aucs = [], []
        for tt in range(args.run):
            if args.model_type == "homo":
                model = BWGNN(in_feats, h_feats, num_classes, graph, args.dropout,
                              d=order, num_lbls=len(args.lbls)).to(args.device)
            elif args.model_type == "gcn":
                model = GCN(in_feats, h_feats, num_classes, graph, args.dropout,
                            num_layers=args.num_layers, activation=activation, num_lbls=len(args.lbls)).to(args.device)
            elif args.model_type == "gat":
                model = GAT(in_feats, h_feats, num_classes, graph, args.dropout,
                            num_layers=args.num_layers, activation=activation, num_lbls=len(args.lbls)).to(args.device)
            elif args.model_type == "appnp":
                model = MultiAPPNP(in_feats, h_feats, num_classes, args.dropout,
                                   graph, num_lbls=len(args.lbls)).to(args.device)
            elif args.model_type == "hetero":
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, args.dropout, d=order, num_lbls=len(
                    args.lbls)).to(args.device)
            else:
                print("model wrong")
            masks = load_mask(args.dataset, tt,
                              graph.ndata["feature"].shape[0])
            ap, auc = train(model, graph, args, masks,
                            using_lbl, coocPage, args.lbls)
            final_ap.append(ap)
            final_aucs.append(auc)

        final_ap = np.array(final_ap)
        final_aucs = np.array(final_aucs)
        print(f"AP-mean: {100 * np.mean(final_ap):.2f}, AP-std: {100 * np.std(final_ap):.2f}, AUC-mean: {100 * np.mean(final_aucs):.2f}, AUC-std: {100 * np.std(final_aucs):.2f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="BWGNN")
    parser.add_argument("--lbltype", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lbls", nargs="+", type=int,
                        default=[0, 1, 2, 3], help="一个整数列表")
    parser.add_argument("--resdir", type=str,
                        default="/data/syf/LIP_MLNC/res/")
    parser.add_argument("--dataset", type=str, default="pcg",
                        help="Dataset for this model")
    parser.add_argument("--embdir", type=str,
                        default="/data/syf/LIP_MLNC/emb/")
    parser.add_argument("--model_type", type=str, default="gcn")
    parser.add_argument("--moddir", type=str,
                        default="/data/syf/LIP_MLNC/mdls/")
    parser.add_argument("--train_ratio", type=float,
                        default=0.6, help="Training ratio")
    parser.add_argument("--test_ratio", type=float,
                        default=0.2, help="Training ratio")

    parser.add_argument("--hid_dim", type=int, default=64,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout rate")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0,
                        help="weight decay")

    parser.add_argument("--order", type=int, default=2,
                        help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1,
                        help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=200,
                        help="The max number of epochs")
    parser.add_argument("--run", type=int, default=3, help="Running times")
    parser.add_argument("--learnCoef", type=str, default="our")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of GCN layers")
    parser.add_argument("--activation", type=str, default="relu",
                        help="Activation function (relu, leaky_relu, etc.)")
    # parser.add_argument("--coocPath", type=str,
    #                     default="/data/syf/LIP_MLNC/data/cooc_dblp.pt")

    args = parser.parse_args()
    print(args)

    # 从 NNI 获取超参数
    tuner_params = nni.get_next_parameter()
    args.hid_dim = tuner_params.get('hid_dim', args.hid_dim)
    args.dropout = tuner_params.get('dropout', args.dropout)
    args.lr = tuner_params.get('lr', args.lr)
    args.wd = tuner_params.get('wd', args.wd)
    args.num_layers = tuner_params.get('num_layers', args.num_layers)
    args.activation = tuner_params.get('activation', args.activation)

    main(args)
