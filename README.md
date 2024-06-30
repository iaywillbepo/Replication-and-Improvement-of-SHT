# Replication and Improvement of Self-Supervised Hypergraph Transformer for Recommender Systems

## Introduction

This article incorporates the pre-training process of graph neural networks into the original paper's code, which can be found at https://github.com/akaxlh/SHT.

## Environment
The code of SHT is implemented and tested under the following development environment:

PyTorch:
* python=3.10.4
* torch=1.11.0
* numpy=1.22.3
* scipy=1.7.3

## Datasets

Three datasets are adopted to evaluate SHT: <i> Yelp, Gowalla, </i>and <i>Tmall</i>. The user-item pair $(u_i, v_j)$ in the adjacent matrix is set as 1, if user $u_i$ has rated item $v_j$ in Yelp, or if user $u_i$ has check in venue $v_j$ in Gowalla, or if user $u_i$ has purchased item $v_j$ in Tmall. We filtered out users and items with too few interactions.

## Usage
Please unzip the datasets first. Also you need to create the `Models/` directory. The following command lines start training and testing on the three datasets, respectively, which also specify the hyperparameter settings for the reported results in the paper. Training and testing logs for trained models are contained in the `History/` directory.

For the pytorch version, switch your working directory to `torchVersion` and then run the commands as below. The implementation has been simplified and improved, to highlight the effect of the proposed self-supervised learning method.

### PyTorch
* Yelp
```
python Main.py --data yelp --ssl1 1 --ssl2 1 --temp 0.2 --reg 3e-7 --edgeSampRate 0.1
```
* Gowalla
```
python Main.py --data gowalla --ssl1 1 --ssl2 1 --temp 0.2 --reg 3e-8 --edgeSampRate 0.1
```
* Tmall
```
python Main.py --data tmall --ssl1 1 --ssl2 1 --temp 0.5 --reg 3e-7 --edgeSampRate 0.01
```

Important arguments:
* `reg`: This is the weight for weight-decay regularization. Empirically recommended tuning range is `{1e-2, 1e-3, 1e-4, 1e-5}`. For the pytorch version, it is tuned from `{1e-8, 3e-8, 1e-7, 3e-7, 1e-6}`.
* `ssl_reg`: This is the weight for the solidity prediction loss of self-supervised learning task. The value is tuned from `{1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`. For the pytorch version, it is split into two hyperparameters `ssl1_reg` and `ssl2_reg`, but in our experiments they are set the same. And it is tuned from `{10, 3, 1, 0.3, 0.1, 0.03}`.
* `mult`: This hyperparameter is to emplify the ssl loss for better performance, which is tuned from `{16, 64, 1e1, 1e2, 1e3}`.
* `edgeSampRate`: This parameter determines the ratio of edges to conduct the solidity differentiation task on. It should be balanced to consider both model performance and training efficiency.


