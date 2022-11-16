#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

START=$(date +%s.%N)

python train.py --gnn=GCN --k=1 --dataset=ogbn_products --compression_rate=big
python train.py --gnn=GCN --k=2 --dataset=ogbn_products --compression_rate=big
python train.py --gnn=GCN --k=5 --dataset=ogbn_products --compression_rate=big
python train.py --gnn=GCN --k=10 --dataset=ogbn_products --compression_rate=big
python train.py --gnn=GCN --k=20 --dataset=ogbn_products --compression_rate=big

python train.py --gnn=GCN --k=1 --dataset=ogbn_products --compression_rate=medium
python train.py --gnn=GCN --k=2 --dataset=ogbn_products --compression_rate=medium
python train.py --gnn=GCN --k=5 --dataset=ogbn_products --compression_rate=medium
python train.py --gnn=GCN --k=10 --dataset=ogbn_products --compression_rate=medium
python train.py --gnn=GCN --k=20 --dataset=ogbn_products --compression_rate=medium

python train.py --gnn=GCN --k=1 --dataset=ogbn_products --compression_rate=small
python train.py --gnn=GCN --k=2 --dataset=ogbn_products --compression_rate=small
python train.py --gnn=GCN --k=5 --dataset=ogbn_products --compression_rate=small
python train.py --gnn=GCN --k=10 --dataset=ogbn_products --compression_rate=small
python train.py --gnn=GCN --k=20 --dataset=ogbn_products --compression_rate=small

ENDPRODUCTS=$(date +%s.%N)

TOTAL=$(echo "$ENDPRODUCTS - $START" | bc)
echo total $TOTAL