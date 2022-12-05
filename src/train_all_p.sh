#!/bin/bash

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

START=$(date +%s.%N)

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