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
CORADIFF=$(echo "$ENDCORA - $START" | bc)
echo cora $CORADIFF

python train.py --gnn=GCN --k=1 --dataset=citeseer --compression_rate=big
python train.py --gnn=GCN --k=2 --dataset=citeseer --compression_rate=big
python train.py --gnn=GCN --k=5 --dataset=citeseer --compression_rate=big 
python train.py --gnn=GCN --k=10 --dataset=citeseer --compression_rate=big
python train.py --gnn=GCN --k=20 --dataset=citeseer --compression_rate=big

python train.py --gnn=GCN --k=1 --dataset=citeseer --compression_rate=medium
python train.py --gnn=GCN --k=2 --dataset=citeseer --compression_rate=medium
python train.py --gnn=GCN --k=5 --dataset=citeseer --compression_rate=medium
python train.py --gnn=GCN --k=10 --dataset=citeseer --compression_rate=medium
python train.py --gnn=GCN --k=20 --dataset=citeseer --compression_rate=medium

python train.py --gnn=GCN --k=1 --dataset=citeseer --compression_rate=small
python train.py --gnn=GCN --k=2 --dataset=citeseer --compression_rate=small
python train.py --gnn=GCN --k=5 --dataset=citeseer --compression_rate=small 
python train.py --gnn=GCN --k=10 --dataset=citeseer --compression_rate=small
python train.py --gnn=GCN --k=20 --dataset=citeseer --compression_rate=small

ENDCITE=$(date +%s.%N)
CITEDIFF=$(echo "$ENDCITE - $ENDCORA" | bc)
echo citeseer $CITEDIFF

python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big
python train.py --gnn=GCN --k=2 --dataset=arxiv --compression_rate=big
python train.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=big 
python train.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big
python train.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=big

python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=medium
python train.py --gnn=GCN --k=2 --dataset=arxiv --compression_rate=medium
python train.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=medium
python train.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=medium
python train.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=medium

python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=small
python train.py --gnn=GCN --k=2 --dataset=arxiv --compression_rate=small
python train.py --gnn=GCN --k=5 --dataset=arxiv --compression_rate=small 
python train.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=small
python train.py --gnn=GCN --k=20 --dataset=arxiv --compression_rate=small

ENDARXIV=$(date +%s.%N)
ARXIVDIFF=$(echo "$ENDARXIV - $ENDCITE" | bc)

TOTAL=$(echo "$ENDARXIV - $START" | bc)
echo cora $CORADIFF citeseer $CITEDIFF arxiv $ARXIVDIFF
echo total $TOTAL
# python train.py --gnn=GCN --k=1 --dataset=proteins --compression_rate=big
# python train.py --gnn=GCN --k=2 --dataset=proteins --compression_rate=big
# python train.py --gnn=GCN --k=5 --dataset=proteins --compression_rate=big 
# python train.py --gnn=GCN --k=10 --dataset=proteins --compression_rate=big
# python train.py --gnn=GCN --k=20 --dataset=proteins --compression_rate=big

# python train.py --gnn=GCN --k=1 --dataset=proteins --compression_rate=medium
# python train.py --gnn=GCN --k=2 --dataset=proteins --compression_rate=medium
# python train.py --gnn=GCN --k=5 --dataset=proteins --compression_rate=medium
# python train.py --gnn=GCN --k=10 --dataset=proteins --compression_rate=medium
# python train.py --gnn=GCN --k=20 --dataset=proteins --compression_rate=medium

# python train.py --gnn=GCN --k=1 --dataset=proteins --compression_rate=small
# python train.py --gnn=GCN --k=2 --dataset=proteins --compression_rate=small
# python train.py --gnn=GCN --k=5 --dataset=proteins --compression_rate=small 
# python train.py --gnn=GCN --k=10 --dataset=proteins --compression_rate=small
# python train.py --gnn=GCN --k=20 --dataset=proteins --compression_rate=small