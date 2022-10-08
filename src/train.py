'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function
'''

import argparse
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
from torch.utils.tensorboard import SummaryWriter

# our modules
from models import GCN,GAT,GraphSage

def train(args):
    print("training configuration:")
    print("Model: ",args.gnn)
    print("Number of partitions:",args.k)
    print("Dataset:", args.dataset)

    # first load the dataset, split into k partitions
    graph = load_dataset(args.dataset)
    partitions = partition(graph,args.k)

    # init tensorboard
    writer = SummaryWriter()

    # training each partition
    for idx,partition in enumerate(partitions):
        # graph parameters
        features = partition.ndata['feat']
        labels = partition.ndata['label']
        train_mask = partition.ndata['train_mask']
        num_classes = partition.num_classes

        # create a gnn for this partition using graph parameters
        model = load_model(args.gnn)

        # create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0

        # train the subgraph
        for e in range(100):
            model.train()

            # Forward
            logits = model(partition, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            train_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            writer.add_scalar("Loss/train", train_loss, e)

            # Compute accuracy on training dataset
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            writer.add_scalar("Accuracy/train", train_acc, e)

            # evaluate on the validation set
            val_acc, val_loss = validate(model, partition)
            writer.add_scalar("Loss/val", val_loss, e)
            writer.add_scalar("Accuracy/val", val_acc, e)

            # Save the best validation accuracy and the corresponding model.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': e+1,
                    'model_state_dict': model.state_dict(),
                    'val_acc': best_val_acc,
                    }, 'best_model_' + str(args.dataset)+'_p'+str(idx+1)+'k'+str(len(graph))+'.pth') # e.g. best_model_cora_p1k5.pth is the best val accuracy for partition 1 of 5 gnn

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # print the 
            if e % 5 == 0:
                print('In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
                    e, train_loss, train_acc, val_loss, val_acc, best_val_acc))
    
    # test the model
    

def validate(model, partition):
    model.eval()

    features = partition.ndata['feat']
    labels = partition.ndata['label']
    val_mask = partition.ndata['val_mask']

    with torch.no_grad():
        # Forward
        logits = model(partition, features)
        pred = logits.argmax(1)

        # Compute loss and accuracy
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        return val_acc


# def test(model, graph):
#     print("============ Testing Model ============")
#     best_checkpoint = torch.load('best_model.pth')
#     best_epoch = best_checkpoint['epoch']
#     best_acc = best_checkpoint['val_acc']
#     model.load_state_dict(best_checkpoint['state_dict'])
#     model.eval()
#     print("Best Val Acc: ", best_acc, " at epoch ", best_epoch)

#     features = graph.ndata['feat']
#     labels = graph.ndata['label']
#     test_mask = graph.ndata['val_mask']

#     with torch.no_grad():
#         # Forward
#         logits = model(graph, features)
#         pred = logits.argmax(1)

#         # Compute loss and accuracy
#         test_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
#         test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
#         return test_acc
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
    parser.add_argument("gnn",help="GNN architecture (GCN, GAT, GSAGE)",type=str)
    parser.add_argument("k",help="how many partitions to split the input graph into",type=int)
    parser.add_argument("dataset",help="name of the dataset (cora, citeseeor,arxiv)",type=str)

    args = parser.parse_args()
    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    print("=================")
    args = parse_args()
    train(args)