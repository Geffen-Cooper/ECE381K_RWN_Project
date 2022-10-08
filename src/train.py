'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function
'''

import argparse
from imp import load_module
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import networkx as nx
import metis
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

def train(args):
    print("training configuration:")
    print("Model: ",args.gnn_model)
    print("Number of partitions:",args.k)
    print("Dataset:", args.dataset)

    # first load the dataset
    graph = load_dataset(args.dataset)

    # next load the model
    model = load_model(args.gnn_model)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #         e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    return best_test_acc

# ================================ datasets =====================================

def load_dataset(dataset):
    if dataset == "cora":
        return dgl.data.CoraGraphDataset()
    elif dataset == "citeseer":
        return dgl.data.CiteseerGraphDataset()
    elif dataset == "arxiv":
        dataset = DglNodePropPredDataset('ogbn-arxiv')
        arxiv_dgl = dgl.data.AsNodePredDataset(dataset)[0]
        arxiv_dgl = dgl.add_reverse_edges(arxiv_dgl)
        return arxiv_dgl

# ================================ models =====================================

def load_model(model):
    if model == "GCN":
        return None
    elif model == "GAT":
        return None
    elif model == "GSAGE":
        return None


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")
    # logging details
    parser.add_argument("gnn_model",help="GNN architecture (GCN, GAT, GSAGE)",type=str)
    parser.add_argument("k",help="how many partitions to split the input graph into",type=int)
    parser.add_argument("dataset",help="name of the dataset (cora, citeseeor,arxiv)",type=int)



# ===================================== Main =====================================
if __name__ == "__main__":
    args = parse_args()
    train(args)