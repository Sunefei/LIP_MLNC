# Multi-Label Node Classification on Graphs with Label Influence Propagation

## Label Influence Propagation (LIP)
This repo is the implementation of paper “Multi-Label Node Classification on Graphs with Label Influence Propagation" submitted to KDD' 24.

## Abstract
Graph neural networks (GNNs) attract extensively research interests for modeling the relational structures of data, ranging from drug design in molecular graphs to anomaly detection in financial networks and beyond.  However, in real-world graph data, each node inherently possesses multiple sets of labels. For instance, in gene interaction networks, each protein may have several functions; in citation networks, papers may belong to multiple domains; and in financial networks, each user might exhibit various types of anomalies. Existing methods can only model each set of labels independently, which on one hand, wastes computational power and time by learning from scratch for each label set, and on the other hand, fails to fully leverage the beneficial correlation implied by the multi-labels, making current methods ineffective in practical multi-label tasks. This paper aims to thoroughly analysis and quantify of the influence between labels across the entire training process. Moreover, based on the analysis and modeling of multi-label relationships, we propose a novel model, **Label Influence Propagation (LIP)**, specifically designed for multi-label node classification, which represents a strategic advancement by fine-tuning the training dynamics among multiple labels to improve overall performance. Our framework is evaluated through several benchmark datasets, which demonstrates the superior performance beyond state-of-the-art methods.

## Requirements
```
dgl-cu111==0.8.2
torch>=1.9.0 (mine==2.0.1)
scikit-learn==1.3.2
tqdm==4.66.1
argparse
```

## Dataset Download
[Datasets Here at GoogleDrive](https://drive.google.com/drive/folders/1-rKI4CAQq144Deca-f4o1R5YR-d-xjht?usp=sharing)

## Usage Example
```python
python main.py --device cuda:0 --dataset dblp --model_type gcn --train_ratio 0.6 --test_ratio 0.2 --learnCoef "our*lbl" --lbls 0 1 2 3
```

# Note that this file is build since the 'anonymous.4open.science' does not seems to support the update of the readme file from github repo.