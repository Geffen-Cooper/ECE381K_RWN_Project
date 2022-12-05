#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

START=$(date +%s.%N)

<<<<<<< HEAD
# train cora dataset
gnn=GCN
dataset=cora

# train 3 different students and 5 different partition numbers on cora
for cr in small medium big
do
    for k in 1 2 5 10 20
    do
        FILE=graph_partitions/${dataset}_p1_k${k}.bin
        if test -f "$FILE"; then
            echo "partitions exists."
        else
            python generate_partitions.py --dataset $dataset --k $k
        fi
        for (( c=0; c<$k; c++  ));
        do
            if [ $c -gt 15 ]; then
                d=$(($c%16))
                e=$((d+17))
                taskset -c $d python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$e &
            else
                taskset -c $c python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$(($c+1)) &
            fi
        done
        wait
    done
done


# train citeseer dataset
dataset=citeseer

# train 3 different students and 5 different partition numbers on citeseer
for cr in small medium big
do
    for k in 1 2 5 10 20
    do
        FILE=graph_partitions/${dataset}_p1_k${k}.bin
        if test -f "$FILE"; then
            echo "partitions exists."
        else
            python generate_partitions.py --dataset $dataset --k $k
        fi
        for (( c=0; c<$k; c++  ));
        do
            if [ $c -gt 15 ]; then
                d=$(($c%16))
                e=$((d+17))
                taskset -c $d python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$e &
            else
                taskset -c $c python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$(($c+1)) &
            fi
        done
        wait
    done
done


# train arxiv dataset
dataset=arxiv

# train 3 different students and 5 different partition numbers on arxiv
for cr in small medium big
do
    for k in 1 2 5 10 20
    do
        FILE=graph_partitions/${dataset}_p1_k${k}.bin
        if test -f "$FILE"; then
            echo "partitions exists."
        else
            python generate_partitions.py --dataset $dataset --k $k
        fi
        for (( c=0; c<$k; c++  ));
        do
            if [ $c -gt 15 ]; then
                d=$(($c%16))
                e=$((d+17))
                taskset -c $d python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$e &
            else
                taskset -c $c python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$(($c+1)) &
            fi
        done
        wait
    done
done
=======
python init_parallel.py --dataset arxiv --k 10
# python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big


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

# python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big


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
ENDTIME=$(date +%s.%N)
TIMEDIFF=$(echo "$ENDTIME - $START" | bc)
echo cora $TIMEDIFF

# ENDPRODUCTS=$(date +%s.%N)
# PRODUCTSDIFF=$(echo "$ENDPRODUCTS - $ENDARXIV" | bc)

# TOTAL=$(echo "$ENDPRODUCTS - $START" | bc)
>>>>>>> 36cfd688553f43d94045b5c3d63c80a796db4a9c
