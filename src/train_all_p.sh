#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

START=$(date +%s.%N)
# python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big

# python init_parallel.py --dataset arxiv --k 1
# taskset -c 0 python train_parallel.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big --i 1 &
# wait
# python init_parallel.py --dataset arxiv --k 2
# taskset -c 0 python train_parallel.py --gnn=GCN --k=2 --dataset=arxiv --compression_rate=big --i 1 &
# taskset -c 1 python train_parallel.py --gnn=GCN --k=2 --dataset=arxiv --compression_rate=big --i 2 &
# wait
# python init_parallel.py --dataset arxiv --k 5
# taskset -c 0 python train_parallel.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big --i 1 &
# taskset -c 1 python train_parallel.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big --i 2 &
# taskset -c 2 python train_parallel.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big --i 3 &
# taskset -c 3 python train_parallel.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big --i 4 &
# taskset -c 4 python train_parallel.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big --i 5 &
# wait
python init_parallel.py --dataset arxiv --k 10
taskset -c 0 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 1 &
taskset -c 1 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 2 &
taskset -c 2 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 3 &
taskset -c 3 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 4 &
taskset -c 4 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 5 &
taskset -c 5 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 6 &
taskset -c 6 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 7 &
taskset -c 7 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 8 &
taskset -c 8 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 9 &
taskset -c 9 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 10 &
# wait
# python init_parallel.py --dataset arxiv --k 20
# taskset -c 0 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 1 &
# taskset -c 1 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 2 &
# taskset -c 2 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 3 &
# taskset -c 3 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 4 &
# taskset -c 4 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 5 &
# taskset -c 5 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 6 &
# taskset -c 6 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 7 &
# taskset -c 7 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 8 &
# taskset -c 8 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 9 &
# taskset -c 9 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 10 &
# taskset -c 10 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 11 &
# taskset -c 11 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 12 &
# taskset -c 12 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 13 &
# taskset -c 13 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 14 &
# taskset -c 14 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 15 &
# taskset -c 15 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 16 &
# taskset -c 0 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 17 &
# taskset -c 1 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 18 &
# taskset -c 2 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 19 &
# taskset -c 3 python train_parallel.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big --i 20 &

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