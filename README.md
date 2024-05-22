# Multi-Label Node Classification with Label Influence Propagation

## Label Influence Propagation (LIP)
This repo is the implementation of paper “Multi-Label Node Classification on Graphs with Label Influence Propagation" submitted to NeurIPS' 24.

## Abstract
Graphs are a prevalent and complex data structure utilized across various domains, with multi-label nodes being particularly significant. Examples include proteins in PPI networks possessing multiple properties and users in social or e-commerce networks exhibiting diverse interests. Addressing multi-label node classification (MLNC) in such contexts has prompted the development of various methods, including those leveraging Graph Neural Networks (GNNs) and label-label networks to enhance classification capabilities. However, existing approaches often fall short in capturing the unique challenges posed by the non-Euclidean nature of graph data and the intricate correlations between multiple labels. This results in suboptimal performance for practical multi-label node classification. 
This paper aims to thoroughly analyze and quantify the influence between labels throughout the entire training process. Building on this analysis, we introduce a novel model, Label Information Propagation (LIP), specifically designed for multi-label node classification. LIP strategically modulates training dynamics among multiple labels to improve overall performance. Our framework is evaluated using several benchmark datasets, and the results demonstrate that our approach surpasses state-of-the-art methods, showing its effectiveness in multi-label node classification.

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

