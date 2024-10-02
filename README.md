# Multi-Label Node Classification with Label Influence Propagation

## Label Influence Propagation (LIP)
This repo is the implementation of paper “Multi-Label Node Classification on Graphs with Label Influence Propagation" submitted to ICLR' 25.

## Abstract
Graphs are a complex and versatile data structure used across various domains, with multi-label nodes playing a particularly crucial role. 
Examples include proteins in PPI networks with multiple functions and users in social or e-commerce networks exhibiting diverse interests. 
Tackling multi-label node classification on graphs (MLNC) has led to the development of various approaches. Some methods leverage graph neural networks (GNNs) to exploit label co-occurrence correlations, while others incorporate label embeddings to capture label proximity. However, these approaches fail to account for the intricate influences between labels in non-Euclidean graph data.
To address this limitation, we decompose the message passing process into two components: propagation and transformation operations. 
We then conduct a comprehensive analysis and quantification of the influence correlations between labels in each component. 
Building on these insights, we propose a novel model, Label Influence Propagation (\model). 
Specifically, we construct a label influence graph based on the integrated label correlations. 
Then, we propagate high-order influences through this graph, dynamically adjusting the learning process by amplifying labels with positive contributions and mitigating those with negative influence.
Finally, our framework is evaluated on comprehensive benchmark datasets, consistently outperforming SOTA methods across various settings, demonstrating its effectiveness on MLNC tasks.

## Requirements
```
dgl-cu111==0.8.2
torch>=1.9.0 (mine==2.0.1)
scikit-learn==1.3.2
tqdm==4.66.1
argparse
```

## Dataset Download
[Datasets Here at GoogleDrive](https://drive.google.com/file/d/1x7cSD9HB6TkB7G6HZG4oFFBpym38PrWI/view?usp=drive_link)

## Usage Example
```python
python main.py --device cuda:0 --dataset dblp --model_type gcn --train_ratio 0.6 --test_ratio 0.2 --learnCoef "our*lbl" --lbls 0 1 2 3
```

