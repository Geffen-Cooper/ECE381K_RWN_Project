#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

python train.py --gnn=GCN --k=1 --dataset=cora
python train.py --gnn=GCN --k=2 --dataset=cora
python train.py --gnn=GCN --k=5 --dataset=cora
python train.py --gnn=GCN --k=10 --dataset=cora
python train.py --gnn=GCN --k=20 --dataset=cora

python train.py --gnn=GCN --k=1 --dataset=citeseer
python train.py --gnn=GCN --k=2 --dataset=citeseer
python train.py --gnn=GCN --k=5 --dataset=citeseer
python train.py --gnn=GCN --k=10 --dataset=citeseer
python train.py --gnn=GCN --k=20 --dataset=citeseer

python train.py --gnn=GCN --k=1 --dataset=arxiv
python train.py --gnn=GCN --k=2 --dataset=arxiv
python train.py --gnn=GCN --k=5 --dataset=arxiv
python train.py --gnn=GCN --k=10 --dataset=arxiv
python train.py --gnn=GCN --k=20 --dataset=arxiv

python train.py --gnn=GAT --k=1 --dataset=cora
python train.py --gnn=GAT --k=2 --dataset=cora
python train.py --gnn=GAT --k=5 --dataset=cora
python train.py --gnn=GAT --k=10 --dataset=cora
python train.py --gnn=GAT --k=20 --dataset=cora

python train.py --gnn=GAT --k=1 --dataset=citeseer
python train.py --gnn=GAT --k=2 --dataset=citeseer
python train.py --gnn=GAT --k=5 --dataset=citeseer
python train.py --gnn=GAT --k=10 --dataset=citeseer
python train.py --gnn=GAT --k=20 --dataset=citeseer

python train.py --gnn=GAT --k=1 --dataset=arxiv
python train.py --gnn=GAT --k=2 --dataset=arxiv
python train.py --gnn=GAT --k=5 --dataset=arxiv
python train.py --gnn=GAT --k=10 --dataset=arxiv
python train.py --gnn=GAT --k=20 --dataset=arxiv

python train.py --gnn=GSAGE --k=1 --dataset=cora
python train.py --gnn=GSAGE --k=2 --dataset=cora
python train.py --gnn=GSAGE --k=5 --dataset=cora
python train.py --gnn=GSAGE --k=10 --dataset=cora
python train.py --gnn=GSAGE --k=20 --dataset=cora

python train.py --gnn=GSAGE --k=1 --dataset=citeseer
python train.py --gnn=GSAGE --k=2 --dataset=citeseer
python train.py --gnn=GSAGE --k=5 --dataset=citeseer
python train.py --gnn=GSAGE --k=10 --dataset=citeseer
python train.py --gnn=GSAGE --k=20 --dataset=citeseer

python train.py --gnn=GSAGE --k=1 --dataset=arxiv
python train.py --gnn=GSAGE --k=2 --dataset=arxiv
python train.py --gnn=GSAGE --k=5 --dataset=arxiv
python train.py --gnn=GSAGE --k=10 --dataset=arxiv
python train.py --gnn=GSAGE --k=20 --dataset=arxiv