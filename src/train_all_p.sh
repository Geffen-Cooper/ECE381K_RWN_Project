#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

python init_parallel.py --dataset ogbn_products --k 10

START=$(date +%s.%N)
# python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big


# taskset -c 0 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 1 &
# taskset -c 1 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 2 &
# taskset -c 2 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 3 &
# taskset -c 3 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 4 &
# taskset -c 4 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 5 &
# taskset -c 5 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 6 &
# taskset -c 6 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 7 &
# taskset -c 7 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 8 &
# taskset -c 8 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 9 &
# taskset -c 9 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 10 &

# python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=big


# python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=big 
# python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=big
# python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=big

# python train.py --gnn=GCN --k=1 --dataset=cora --compression_rate=medium
# python train.py --gnn=GCN --k=2 --dataset=cora --compression_rate=medium
# python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=medium
# python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=medium
# python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=medium

# python train.py --gnn=GCN --k=1 --dataset=cora --compression_rate=small
# python train.py --gnn=GCN --k=2 --dataset=cora --compression_rate=small
# python train.py --gnn=GCN --k=5 --dataset=cora --compression_rate=small 
# python train.py --gnn=GCN --k=10 --dataset=cora --compression_rate=small
# python train.py --gnn=GCN --k=20 --dataset=cora --compression_rate=small
wait
ENDCORA=$(date +%s.%N)
CORADIFF=$(echo "$ENDCORA - $START" | bc)
echo cora $CORADIFF

# ENDPRODUCTS=$(date +%s.%N)
# PRODUCTSDIFF=$(echo "$ENDPRODUCTS - $ENDARXIV" | bc)

# TOTAL=$(echo "$ENDPRODUCTS - $START" | bc)