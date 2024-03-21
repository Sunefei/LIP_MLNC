import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from dataset import Dataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    precision_score,
    confusion_matrix,
    hamming_loss,
)
from BWGNN import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import networkx as nx
from utils import cosine_similarity


def Lbltrace(A, Y):
    # 计算度矩阵D
    D = torch.diag(A.sum(1))

    # 计算拉普拉斯矩阵L
    L = D - A

    # 计算Y乘以L再乘以Y的转置
    YLY_t = Y @ L @ Y.T

    # 计算迹
    trace_YLY_t = torch.trace(YLY_t)

    # 输出结果
    trace_YLY_t
    return trace_YLY_t


def train(model, g, args, tt, labels, cooc, tasks):
    features = g.ndata["feature"].to(args.device)
    # labels = g.ndata["label"]
    num_lbls = len(tasks)
    train_mask = torch.zeros([num_lbls, len(labels[tasks[0]])]).bool()
    val_mask = torch.zeros([num_lbls, len(labels[tasks[0]])]).bool()
    test_mask = torch.zeros([num_lbls, len(labels[tasks[0]])]).bool()
    cnt = 0
    label_rest = {}
    coefs = []
    for key, lbl in labels.items():
        index = list(range(len(lbl)))
        # if dataset_name == "amazon":
        #     index = list(range(3305, len(labels)))
        label_rest[cnt] = lbl.long()

        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index,
            lbl[index],
            stratify=lbl[index],
            train_size=args.train_ratio,
            random_state=2,
            shuffle=True,
        )
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest,
            y_rest,
            stratify=y_rest,
            test_size=args.test_ratio,
            random_state=2,
            shuffle=True,
        )  # 0.67

        train_mask[cnt][idx_train] = 1
        val_mask[cnt][idx_valid] = 1
        test_mask[cnt][idx_test] = 1
        print(
            str(cnt) + "train/dev/test samples: ",
            train_mask[cnt].sum().item(),
            val_mask[cnt].sum().item(),
            test_mask[cnt].sum().item(),
        )
        # coef = torch.nn.Parameter(torch.ones(1, requires_grad=True))  # 创建一个可学习的系数
        # coef = coef.to(args.device)  # 将系数移动到指定设备
        # coefs.append(coef)

        cnt += 1
    if args.learnCoef == "auto":
        coefs = [
            torch.nn.Parameter(torch.ones(1, requires_grad=True, device=args.device))
            for _ in range(num_lbls)
        ]
        coefs = [coef.to(args.device) for coef in coefs]
        optimizer = torch.optim.Adam(coefs + list(model.parameters()), lr=0.01)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = (
        [0.0] * num_lbls,
        [0.0] * num_lbls,
        [0.0] * num_lbls,
        [0.0] * num_lbls,
        [0.0] * num_lbls,
        [0.0] * num_lbls,
    )

    best_f1_, avg_final_trec, avg_final_tpre, avg_final_tmf1, avg_final_tauc = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    # weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    # print("cross entropy weight: ", weight)
    time_start = time.time()
    model_parameters = list(model.gnn1.parameters()) + list(model.gnn2.parameters())
    for e in tqdm(range(args.epoch)):
        model.train()
        logits, h = model(features)
        loss = 0.0
        argmax_list = []
        loss_list = []
        grads_loss = []
        for no, out in enumerate(logits):
            weight = (1 - label_rest[no][train_mask[no]]).sum().item() / label_rest[no][
                train_mask[no]
            ].sum().item()
            # print("cross entropy weight: ", weight)
            if args.learnCoef == "auto":
                loss += coefs[no] * F.cross_entropy(
                    out[train_mask[no]],
                    label_rest[no].to(args.device)[train_mask[no].to(args.device)],
                    # weight=torch.tensor([1.0, weight], device=args.device),
                )
            elif args.learnCoef == "cooc":
                loss += cooc[no].sum() * F.cross_entropy(
                    out[train_mask[no]],
                    label_rest[no].to(args.device)[train_mask[no].to(args.device)],
                    # weight=torch.tensor([1.0, weight], device=args.device),
                )
            elif args.learnCoef == "none":
                loss += F.cross_entropy(
                    out[train_mask[no]],
                    label_rest[no].to(args.device)[train_mask[no].to(args.device)],
                    weight=torch.tensor([1.0, weight], device=args.device),
                )
            elif args.learnCoef == "our*lbl" or args.learnCoef == "grad":
                loss_i = F.cross_entropy(
                    out[train_mask[no]],
                    label_rest[no].to(args.device)[train_mask[no].to(args.device)],
                    weight=torch.tensor([1.0, weight], device=args.device),
                )
                loss_list.append(loss_i)
                # argmax_list.append(torch.argmax(out, dim=1))
                argmax_list.append(out[:, 1].to(float))
                gw_real = torch.autograd.grad(
                    loss_i,
                    model_parameters,
                    retain_graph=True,
                    # create_graph=True,
                    # allow_unused=True,
                )
                # print("hhh")
                gw_real = list((_.detach().clone() for _ in gw_real))
                # print(gw_real)
                grads_loss.append(gw_real)
            # print(loss)
        if args.learnCoef == "our*lbl" or args.learnCoef == "grad":
            # yPred = torch.stack(argmax_list)
            # # 计算点积矩阵
            # dot_product_matrix = torch.matmul(yPred, yPred.T)

            # # 计算每个向量的范数
            # norms = torch.norm(yPred, dim=1)

            # # 计算范数的外积，得到一个k x k的矩阵，其中每个元素[i, j]是向量i和向量j的范数乘积
            # norm_matrix = torch.matmul(norms.unsqueeze(1), norms.unsqueeze(0))

            # # 计算余弦相似度矩阵
            # cosine_similarity_matrix = dot_product_matrix / norm_matrix

            # TODO
            cos_similarities = torch.zeros((len(grads_loss), len(grads_loss)))
            for i in range(len(grads_loss) - 1):
                for j in range(i + 1, len(grads_loss)):
                    for kk, name in enumerate(grads_loss[i]):
                        cos_sim = cosine_similarity(
                            grads_loss[i][kk], grads_loss[j][kk]
                        )
                        cos_similarities[i][j] += cos_sim.item()
                        cos_similarities[j][i] += cos_sim.item()

            # x_min = cos_similarities.min(dim=1, keepdim=True).values
            # x_max = cos_similarities.max(dim=1, keepdim=True).values

            # # 将每一行归一化到[0, 1]
            # x_normalized = (cos_similarities - x_min) / (x_max - x_min)

            # # 将[0, 1]范围内的值缩放到[-1, 1]
            # x_scaled = x_normalized * 2 - 1
            if args.learnCoef == "grad":
                # cooc = x_scaled
                cooc = cos_similarities
            else:
                cooc *= cos_similarities
                # cooc *= x_scaled
                # cooc = F.softmax(cos_similarities, dim=0) + F.softmax(cooc, dim=0)
            cooc = F.softmax(cooc)
            # cooc = torch.sum(cooc, dim=1)
            G_ourPR = nx.from_numpy_array(cooc.numpy())
            outPGPage = nx.pagerank(G_ourPR)
            for ii, loss_i in enumerate(loss_list):
                # loss += loss_i * cooc[ii].to(args.device)
                loss += loss_i * outPGPage[ii]
            # cooc
        #     trace_YLY_t = Lbltrace(cooc.to(args.device), y.to())
        #     loss += trace_YLY_t
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        f1 = []
        trec = []
        tpre = []
        tmf1 = []
        tauc = []
        PredAll = []
        yAll = []
        for no, out in enumerate(logits):
            probs = out.cpu().softmax(1)
            f1_, thres = get_best_f1(label_rest[no][val_mask[no]], probs[val_mask[no]])
            f1.append(f1_)
            preds = numpy.zeros_like(label_rest[no])
            preds[probs[:, 1] > thres] = 1
            trec.append(
                recall_score(label_rest[no][test_mask[no]], preds[test_mask[no]])
            )
            tpre.append(
                precision_score(label_rest[no][test_mask[no]], preds[test_mask[no]])
            )
            tmf1.append(
                f1_score(
                    label_rest[no][test_mask[no]], preds[test_mask[no]], average="macro"
                )
            )
            tauc.append(
                roc_auc_score(
                    label_rest[no][test_mask[no]],
                    probs[test_mask[no]][:, 1].detach().numpy(),
                )
            )
            PredAll.append(label_rest[no][test_mask[no]])
            yAll.append(torch.tensor(preds[test_mask[no]]))
        preds_all = torch.stack(PredAll, dim=1)
        y_all = torch.stack(yAll, dim=1)

        hammingloss = hamming_loss(preds_all, y_all)
        if best_f1_ < np.mean(f1):
            best_f1_ = np.mean(f1)
            best_f1 = f1
            avg_final_trec = np.mean(trec)
            final_trec = trec
            avg_final_tpre = np.mean(tpre)
            final_tpre = tpre
            avg_final_tmf1 = np.mean(tmf1)
            final_tmf1 = tmf1
            avg_final_tauc = np.mean(tauc)
            final_tauc = tauc
            final_hamloss = hammingloss
            best_h = h
            best_model = model
        print(
            "Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})".format(
                e, loss.cpu().item(), np.mean(f1), best_f1_
            )
        )

    torch.save(
        best_h, args.embdir + str(args.lbltype) + "_embedding_run_" + str(tt) + ".pt"
    )
    torch.save(
        best_model.state_dict(),
        args.moddir + str(args.lbltype) + "_model_parameters" + str(tt) + ".pth",
    )
    # "&".join(str(num) for num in tasks)
    number_str = 'all'
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_f1.npy",
        best_f1,
    )
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_trec.npy",
        final_trec,
    )
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_tpre.npy",
        final_tpre,
    )
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_tmf1.npy",
        final_tmf1,
    )
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_tauc.npy",
        final_tauc,
    )
    np.save(
        args.resdir
        + args.dataset
        + "_"
        + args.model_type
        + "_task"
        + number_str
        + "_best_hamloss.npy",
        final_hamloss,
    )

    time_end = time.time()
    print("time cost: ", time_end - time_start, "s")
    print(
        "Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}".format(
            avg_final_trec * 100,
            avg_final_tpre * 100,
            avg_final_tmf1 * 100,
            avg_final_tauc * 100,
            final_hamloss * 100,
        )
    )
    return avg_final_tmf1, avg_final_tauc, final_hamloss


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average="macro")
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BWGNN")
    parser.add_argument("--lbltype", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--lbls", nargs="+", type=int, default=[0, 1, 2, 3], help="一个整数列表"
    )
    parser.add_argument(
        "--resdir",
        type=str,
        default="/data/syf/workspace/Rethinking-Anomaly-Detection/res/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dblp",
        help="Dataset for this model (yelp/amazon/tfinance/tsocial)",
    )
    parser.add_argument(
        "--embdir",
        type=str,
        default="/data/syf/workspace/Rethinking-Anomaly-Detection/emb/",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gcn",
    )
    parser.add_argument(
        "--moddir",
        type=str,
        default="/data/syf/workspace/Rethinking-Anomaly-Detection/mdls/",
    )
    parser.add_argument("--train_ratio", type=float, default=0.2, help="Training ratio")
    parser.add_argument("--test_ratio", type=float, default=0.6, help="Training ratio")
    parser.add_argument(
        "--hid_dim", type=int, default=64, help="Hidden layer dimension"
    )
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument(
        "--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="The max number of epochs"
    )
    parser.add_argument("--run", type=int, default=3, help="Running times")

    parser.add_argument("--learnCoef", type=str, default="cooc")
    parser.add_argument(
        "--coocPath",
        type=str,
        default="/data/syf/workspace/GrabMultiLabel/cooc_dblp.pt",
    )

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    data = Dataset(dataset_name, homo)
    graph = data.graph.to(args.device)
    all_labels = data.labels
    in_feats = graph.ndata["feature"].shape[1]
    num_classes = 2

    # coocPage = torch.load("/data/syf/workspace/GrabMultiLabel/ourPgRank.pt")
    path = "/data/syf/workspace/GrabMultiLabel/"
    if args.learnCoef in ["cooc", "our*lbl"]:
        coocPage = torch.tensor(np.load(path + dataset_name + "_PRcooc.npy"))
    else:
        coocPage = None
    # .to(
    #     args.device
    # )
    # print(coocPage.shape)
    # G_cooc = nx.from_numpy_array(cooc)
    # coocPage = nx.pagerank(G_cooc)

    using_lbl = {key: all_labels[key] for key in args.lbls}

    if args.run == 1:
        tt = args.run
        if args.model_type == "homo":
            model = BWGNN(
                in_feats, h_feats, num_classes, graph, d=order, num_lbls=len(args.lbls)
            ).to(args.device)
        elif args.model_type == "gcn":
            model = GCN(
                in_feats, h_feats, num_classes, graph, num_lbls=len(args.lbls)
            ).to(args.device)
        elif args.model_type == "appnp":
            print(args.model_type)
            model = MultiAPPNP(
                in_feats, h_feats, num_classes, graph, num_lbls=len(args.lbls)
            ).to(args.device)
        elif args.model_type == "hetero":
            model = BWGNN_Hetero(
                in_feats, h_feats, num_classes, graph, d=order, num_lbls=len(args.lbls)
            ).to(args.device)
        else:
            print("fucking wrong")
        train(model, graph, args, tt, using_lbl, coocPage, args.lbls)

    else:
        final_mf1s, final_aucs, final_hamlosses = [], [], []
        for tt in range(args.run):
            if args.model_type == "homo":
                model = BWGNN(
                    in_feats,
                    h_feats,
                    num_classes,
                    graph,
                    d=order,
                    num_lbls=len(args.lbls),
                ).to(args.device)
            elif args.model_type == "gcn":
                model = GCN(
                    in_feats, h_feats, num_classes, graph, num_lbls=len(args.lbls)
                ).to(args.device)
            elif args.model_type == "gat":
                model = GAT(
                    in_feats, h_feats, num_classes, graph, num_lbls=len(args.lbls)
                ).to(args.device)
            elif args.model_type == "appnp":
                print(args.model_type)
                model = MultiAPPNP(
                    in_feats, h_feats, num_classes, graph, num_lbls=len(args.lbls)
                ).to(args.device)
            elif args.model_type == "hetero":
                model = BWGNN_Hetero(
                    in_feats,
                    h_feats,
                    num_classes,
                    graph,
                    d=order,
                    num_lbls=len(args.lbls),
                ).to(args.device)
            else:
                print("fucking wrong")
            mf1, auc, hamloss = train(
                model, graph, args, tt, using_lbl, coocPage, args.lbls
            )
            final_mf1s.append(mf1)
            final_aucs.append(auc)
            final_hamlosses.append(hamloss)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        final_hamlosses = np.array(final_hamlosses)
        print(
            "MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}, ham-mean: {:.2f}, ham-std: {:.2f}".format(
                100 * np.mean(final_mf1s),
                100 * np.std(final_mf1s),
                100 * np.mean(final_aucs),
                100 * np.std(final_aucs),
                np.mean(final_hamlosses),
                np.std(final_hamlosses),
            )
        )
