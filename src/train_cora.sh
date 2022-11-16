#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

START=$(date +%s.%N)

python train.py --gnn=GCN --k=1 --dataset=cora --compression_rate=big
python train.py --gnn=GCN --k=2 --dataset=cora --compression_rate=big
python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=big
python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=big
python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=big

python train.py --gnn=GCN --k=1 --dataset=cora --compression_rate=medium
python train.py --gnn=GCN --k=2 --dataset=cora --compression_rate=medium
python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=medium
python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=medium
python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=medium

python train.py --gnn=GCN --k=1 --dataset=cora --compression_rate=small
python train.py --gnn=GCN --k=2 --dataset=cora --compression_rate=small
python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=small
python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=small
python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=small

ENDCORA=$(date +%s.%N)

TOTAL=$(echo "$ENDCORA - $START" | bc)
echo total $TOTAL