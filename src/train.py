'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function
'''

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

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
import os

# our modules
from models import *
from datasets import *
from partition_graph import *

def train(args):
    print("training configuration:")
    print("Model: ",args.gnn)
    print("Number of partitions:",args.k)
    print("Dataset:", args.dataset)
    print("num_heads: ", args.heads)

    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset)
    
    if args.k == 1:
        partitions = [dataset_dgl]
    else:
        partitions, parts_tensor = partition_network(args.k, dataset_nx, dataset_dgl)
    

    # init tensorboard
    writer = SummaryWriter()

    # training each partition
    for idx,partition in enumerate(partitions):
        # graph parameters
        features = partition.ndata['feat']
        labels = partition.ndata['label']
        train_mask = partition.ndata['train_mask']
        num_classes = dataset.num_classes

        # create a gnn for this partition using graph parameters
        model = load_model(args.gnn,features,num_classes, args.heads)
        student_model = load_model(args.gnn,features,num_classes, args.heads) #TODO: Make this smaller model and make model a bigger model

        # create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_val_acc = 0
        # best_distillation_loss = torch.empty(2,3)
        # print(best_distillation_loss)

        model.train()
        student_model.train()

        # Forward
        logits = model(partition, features)
        student_logits = student_model(partition, features)
        distillation_loss = abs(student_logits - logits)
        best_distillation_loss = distillation_loss
        best_student_logits = student_logits
        torch.save({'model_state_dict': student_model.state_dict()},
                   'best_student_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(
                       args.k) + '.pth')

        # Obtain Distilled Model and logits
        for i in range(100):
            model.train()
            student_model.train()

            # Forward
            logits = model(partition, features)
            student_logits = student_model(partition, features)
            distillation_loss = abs(student_logits - logits)
            # print(logits)
            # print(student_logits)
            # print(distillation_loss)
            sum_distillation_loss = torch.sum(distillation_loss)
            # print(sum_distillation_loss)


            if torch.sum(best_distillation_loss) < torch.sum(distillation_loss):
                best_distillation_loss = distillation_loss
                best_student_logits = student_logits
                torch.save({'model_state_dict': student_model.state_dict()}, 'best_student_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth')


        # train the subgraph
        for e in range(100):
            model.train()
            student_model.train()

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
                    }, 'best_'+str(args.gnn)+'_' + str(args.dataset)+'_p'+str(idx+1)+'_k'+str(args.k)+'.pth') # e.g. best_model_cora_p1_k5.pth is the best val accuracy for partition 1 of kth gnn
            # Saved model is here
            # best_model_GCN_cora_p1_k2.pth
            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # print the 
            if e % 5 == 0:
                print('In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
                    e, train_loss, train_acc, val_loss, val_acc, best_val_acc))

        student_pred = best_student_logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        student_train_loss = F.cross_entropy(best_student_logits[train_mask], labels[train_mask])
        writer.add_scalar("Loss/train", student_train_loss, e)

        # Compute accuracy on training dataset
        student_train_acc = (student_pred[train_mask] == labels[train_mask]).float().mean()
        writer.add_scalar("Accuracy/train", student_train_acc, e)

        # evaluate on the validation set
        student_val_acc, student_val_loss = validate(model, partition)
        writer.add_scalar("Loss/val", student_val_loss, e)
        writer.add_scalar("Accuracy/val", student_val_acc, e)

        # Save the validation accuracy
        torch.save({
            'epoch': e + 1,
            'val_acc': student_val_acc,
            }, 'best_student_validation' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(
            args.k) + '.pth')  # e.g. best_student_validation_model_cora_p1_k5.pth is the best val accuracy for partition 1 of kth gnn
        # Saved model is here
        # best_student_validation_GCN_cora_p1_k2.pth

        print("p", idx," best --> ", best_val_acc)
        print("p", idx, " student --> ", student_val_acc)
    
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
        return val_acc, val_loss


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
        return load_cora()
    elif dataset == "citeseer":
        return load_citeseer()
    elif dataset == "arxiv":
        return load_arxiv()

# ================================ models =====================================


def load_model(model, features, num_classes, heads):
    length = features.shape[1]
    if model == "GCN":
        return GCN(length, length//2, num_classes)
    elif model == "GAT":
        #return GATConv(length, num_classes, num_heads=3)
        return GAT(length, length//2, num_classes, num_heads = heads)
    elif model == "GSAGE":
        return GraphSage(length, length//2, num_classes)


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument("gnn",help="GNN architecture (GCN, GAT, GSAGE)",type=str)
    parser.add_argument("k",help="how many partitions to split the input graph into",type=int)
    parser.add_argument("dataset",help="name of the dataset (cora, citeseeor,arxiv)",type=str)
    parser.add_argument("heads",help="If using GAT provide num_heads, otherwise enter 0",type=str)


    args = parser.parse_args()
    print(args)
    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    print("=================")
    #os.environ['METIS_DLL'] = '/home/mustafa/Documents/Git_Repos/ECE381K_RWN_Project/scripts/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    args = parse_args()
    train(args)