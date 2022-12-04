'''
    This file takes is used for profiling memory during inference using similar
    command line arguments to training. It uses checkpoints to log differnt sections.
'''

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

import time 

# initial memory consumption
time.sleep(1)
t = time.time()
print("pre-import\n",t,flush=True)
time.sleep(1)

import argparse
from torch import no_grad,save,load
from torch.nn import KLDivLoss
import torch.nn.functional as F
from torch.optim import Adam

from graphviz import Digraph
from torch.autograd import Variable
import torch

# our modules
from models import *
from datasets import *
from partition_graph import *

# memory concumption of libraries
t = time.time()
print("import\n", t,flush=True)
time.sleep(1)


# the function loads the graph data and does a forward pass
def inference(args):
    device = "cpu"

    # if profiling a single partioin/core
    if args.parallel == "y":
        print("parallel")
        # first load the dataset partition
        subgraph_path = 'graph_partitions/' + str(args.dataset) + '_p' + str(args.i) + '_k' + str(args.k) + '.bin'
        partition = dgl.load_graphs(subgraph_path)[0][0]
        partitions = [partition]
    else:
        # first load the dataset, no partitioning during inference
        dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset,1)
        partitions = [dataset_dgl]
        
    # memory concumption of graph data
    t = time.time()
    print("load-data\n", t,flush=True)
    time.sleep(1)

    # exit after the first partition (use as an estimate since all partitions are roughly the same size)
    for idx, partition in enumerate(partitions):
        # graph parameters
        features = partition.ndata['feat']
        labels = partition.ndata['label']  # all labels
        train_mask = partition.ndata['train_mask']
        num_classes = len(torch.unique(labels))

        partition = partition.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # create a gnn for this partition using graph parameters
        if args.ts == "t":
            print("teacher inference")
            model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
        elif args.ts == "s":
            print("student inference")
            model = load_student_model(args.gnn, features, num_classes, args.heads,args.dropout, args.compression_rate) 

        # memory concumption of the model
        t = time.time()
        print("load-model\n", t,flush=True)
        time.sleep(1)

        model.eval()
        
        # Forward
        logits = model(partition, features)

        # Compute prediction
        pred = logits.argmax(1)

        # memory concumption of forward pass
        t = time.time()
        print("forward\n", t,flush=True)
        time.sleep(1)
        exit()


# ================================ datasets =====================================

def load_dataset(dataset,k=None):
    if dataset == "cora":
        return load_cora(k)
    elif dataset == "citeseer":
        return load_citeseer(k)
    elif dataset == "arxiv":
        return load_arxiv(k)

# ================================ models =====================================


def load_model(model, features, num_classes, heads, dropout):
    length = features.shape[1]
    if model == "GCN":
        return GCN(length, int(length), num_classes, dropout)
    elif model == "GAT":
        # return GATConv(length, num_classes, num_heads=3)
        return GAT(length, length//2, num_classes, heads, dropout)
    elif model == "GSAGE":
        return GraphSage(length, length//2, num_classes, dropout)


def load_student_model(model, features, num_classes, heads, dropout, compression_rate):
    length = features.shape[1]
    if model == "GCN":
        if compression_rate == "big":
            return GCN(length, int(length*0.1), num_classes, dropout)
        elif compression_rate == "medium":
            return GCN(length, int(length*.2), num_classes, dropout)
        else: # compression_rate == "small":
            return GCN(length, int(length*.5), num_classes, dropout)

    elif model == "GAT":
        # return GATConv(length, num_classes, num_heads=3)
        return GAT(length, length//4, num_classes, heads, dropout)
    elif model == "GSAGE":
        return GraphSage(length, length//4, num_classes, dropout)


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument("--gnn",help="GNN architecture (GCN, GAT, GSAGE)",type=str, default="GCN")
    parser.add_argument("--k",help="how many partitions to split the input graph into",type=int, default=2)
    parser.add_argument("--dataset",help="name of the dataset (cora, citeseeor,arxiv)",type=str, default="cora")
    parser.add_argument("--heads",help="If using GAT provide num_heads, otherwise enter 0",type=int, default=3)
    parser.add_argument("--dropout", help="Dropout rate. 1 - keep_probability = dropout rate", type=float, default=0.25)
    parser.add_argument("--ts",help="is the model teacher or student (t or s)",type=str, default="t")
    parser.add_argument("--compression_rate", help="compression rate for KD (big, medium, small)", type=str, default="medium")
    parser.add_argument("--parallel", help="if doing inference on a single core/partition (y,n)",type=str,default="n")
    parser.add_argument("--i",help="the partition idx to train",type=int)

    args = parser.parse_args()
    print(args)
    return args


# ===================================== Main =====================================
if __name__ == "__main__":
    args = parse_args()
    inference(args)
