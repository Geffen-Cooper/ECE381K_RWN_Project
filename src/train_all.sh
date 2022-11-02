#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

python train.py GCN 1 cora 0
python train.py GCN 2 cora 0
python train.py GCN 5 cora 0
python train.py GCN 10 cora 0
python train.py GCN 20 cora 0

python train.py GCN 1 citeseer 0
python train.py GCN 2 citeseer 0
python train.py GCN 5 citeseer 0
python train.py GCN 10 citeseer 0
python train.py GCN 20 citeseer 0

# python train.py GCN 1 arxiv 0
# python train.py GCN 2 arxiv 0
# python train.py GCN 3 arxiv 0

python train.py GAT 1 cora 3
python train.py GAT 2 cora 3
python train.py GAT 5 cora 3
python train.py GAT 10 cora 3
python train.py GAT 20 cora 3

python train.py GAT 1 citeseer 3
python train.py GAT 2 citeseer 3
python train.py GAT 5 citeseer 3
python train.py GAT 10 citeseer 3
python train.py GAT 20 citeseer 3

# python train.py GAT 1 arxiv 3
# python train.py GAT 2 arxiv 3
# python train.py GAT 3 arxiv 3

python train.py GSAGE 1 cora 0
python train.py GSAGE 2 cora 0
python train.py GSAGE 5 cora 0
python train.py GSAGE 10 cora 0
python train.py GSAGE 20 cora 0

python train.py GSAGE 1 citeseer 0
python train.py GSAGE 2 citeseer 0
python train.py GSAGE 5 citeseer 0
python train.py GSAGE 10 citeseer 0
python train.py GSAGE 20 citeseer 0

# python train.py GSAGE 1 arxiv 0
# python train.py GSAGE 2 arxiv 0
# python train.py GSAGE 3 arxiv 0