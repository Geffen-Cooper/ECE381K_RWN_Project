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
import time
import copy 

# our modules
from models import *
from datasets import *
from partition_graph import *


def train(args):
    print("training configuration:")
    print("Model: ", args.gnn)
    print("Number of partitions:", args.k)
    print("Dataset:", args.dataset)
    print("num_heads: ", args.heads)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(device)

    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset)

    if args.k == 1:
        partitions = [dataset_dgl]
    else:
        partitions, parts_tensor = partition_network(args.k, dataset_nx, dataset_dgl)

    # init tensorboard
    writer = SummaryWriter()

    # graph parameters
    features = partitions[0].ndata['feat']
    num_classes = dataset.num_classes

## ================= Initialize Teacher Models ======================
    # create a gnn for this partition using graph parameters
    teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
    # student_model = load_student_model(args.gnn, features, num_classes, args.heads,
    #                             args.dropout, args.compression_rate) 
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01)

    teacher_models = []
    teacher_optimizers = []
    # student_models = []

    # identical initialization
    for idx, partition in enumerate(partitions):
        teacher_models.append(copy.deepcopy(teacher_model))
        teacher_optimizers.append(copy.deepcopy(optimizer))
        # student_models.append(copy.deepcopy(student_model))

    # training each partition
    for idx, partition in enumerate(partitions):
        # graph parameters
        features = partition.ndata['feat']
        labels = partition.ndata['label']  # all labels
        train_mask = partition.ndata['train_mask']
        num_classes = dataset.num_classes

        partition = partition.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # # NEW ADDITION: count model params
        # print("# teacher params",sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
        # print("# student params",sum(p.numel() for p in student_model.parameters() if p.requires_grad))

        # student_model[idx].to(device)

        # create the optimizer
        # optimizer = torch.optim.Adam(teacher_model[idx].parameters(), lr=0.01)
        # student_optimizer = torch.optim.Adam(student_model[idx].parameters(), lr=0.01)
        best_val_acc = 0
        best_path = 'none'

        best_path = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
        checkpoint = {
            'epoch': 0,
            'model_state_dict': teacher_models[idx].state_dict(),
            'optimizer_state_dict':teacher_optimizers[idx].state_dict(),
            'val_acc': best_val_acc,
            'partition_size': partition.num_nodes(),
            'train_size': sum(partition.ndata['train_mask'] == True),
            'val_size': sum(partition.ndata['val_mask'] == True),
            'total_val_size': sum(dataset_dgl.ndata['val_mask'] == True),
            'val_mask': partition.ndata['val_mask'],
            'test_mask': partition.ndata['test_mask'],
            'args':args
        }
        if args.k > 1:
            checkpoint['node_ids'] = partition.ndata['og_ids']
        torch.save(checkpoint, best_path)

    # aggregation training loop
    e = 0
    for e1 in range(25):
        for idx, partition in enumerate(partitions):
            if e1 == 0:
                teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
                
                teacher_model.to(device)
                optimizer = torch.optim.Adam(teacher_model.parameters(),lr=0.01)
                teacher_model.train()
            else:
                # load teacher model
                best_path = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
                teacher_checkpoint = torch.load(best_path)
                teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
                teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
                
                teacher_model.to(device)
                teacher_model.train()
                
                optimizer = torch.optim.Adam(teacher_model.parameters(),lr=0.01)
                optimizer.load_state_dict(teacher_checkpoint['optimizer_state_dict'])
                
                
                best_val_acc = teacher_checkpoint['val_acc']

            # graph parameters
            features = partition.ndata['feat']
            labels = partition.ndata['label']  # all labels
            train_mask = partition.ndata['train_mask']
            num_classes = dataset.num_classes

            partition = partition.to(device)
            features = features.to(device)
            labels = labels.to(device)

            for e2 in range(4):
                # Forward
                logits = teacher_model(partition, features)

                # Compute prediction
                pred = logits.argmax(1)

                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                train_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                writer.add_scalar("Loss/train", train_loss, e)

                # Compute accuracy on training dataset
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                writer.add_scalar("Accuracy/train", train_acc, e)

                # Backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                e+=1

            # evaluate on the validation set
            val_acc, val_loss = validate(teacher_model, partition)
            writer.add_scalar("Loss/val", val_loss, e*e2)
            writer.add_scalar("Accuracy/val", val_acc, e*e2)
            
            # Save the best validation accuracy and the corresponding model.
            if best_val_acc < val_acc:
                best_val_acc = val_acc.item()
                best_path = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
                checkpoint = {
                    'epoch': e*e2 + 1,
                    'model_state_dict': teacher_model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'partition_size': partition.num_nodes(),
                    'train_size': sum(partition.ndata['train_mask'] == True),
                    'val_size': sum(partition.ndata['val_mask'] == True),
                    'total_val_size': sum(dataset_dgl.ndata['val_mask'] == True),
                    'val_mask': partition.ndata['val_mask'],
                    'test_mask': partition.ndata['test_mask'],
                    'args':args
                }
                if args.k > 1:
                    checkpoint['node_ids'] = partition.ndata['og_ids']
                torch.save(checkpoint, best_path)

            # print the current results
            if e1 % 5 == 0:
                print("partition",idx)
                print(
                    'In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
                        e, train_loss, train_acc, val_loss, val_acc, best_val_acc))

            
        # now aggregate
        # for idx, partition in enumerate(partitions):
        #     # load model
        #     best_path = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
        #     teacher_checkpoint = torch.load(best_path)
        #     teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
        #     teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
        #     # load other models
        #     for i,key in enumerate(teacher_model.state_dict()):
        #         for idx_other, partition_other in enumerate(partitions):
        #             if idx == idx_other and args.k > 1:
        #                 teacher_model.state_dict()[key] *= 0.5
        #             else:
        #                 # load teacher model
        #                 best_path_other = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx_other + 1) + '_k' + str(args.k) + '.pth'
        #                 teacher_checkpoint_other = torch.load(best_path)
        #                 teacher_model_other = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
        #                 teacher_model_other.load_state_dict(teacher_checkpoint_other['model_state_dict'])
        #                 teacher_model.state_dict()[key] += (teacher_model_other.state_dict()[key]/args.k)*.5
                
    # print('(best val acc: {:.3f})'.format(best_val_acc))

#         # Initializing distillation loss
#         distillation_loss = 1000000000  # Very large number
#         best_distillation_loss = distillation_loss


#         # train the subgraph
#         for e in range(100):

#             # Forward
#             logits = teacher_model(partition, features)

#             # Compute prediction
#             pred = logits.argmax(1)

#             # Compute loss
#             # Note that you should only compute the losses of the nodes in the training set.
#             train_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
#             writer.add_scalar("Loss/train", train_loss, e)

#             # Compute accuracy on training dataset
#             train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
#             writer.add_scalar("Accuracy/train", train_acc, e)

#             # evaluate on the validation set
#             val_acc, val_loss = validate(teacher_model, partition)
#             writer.add_scalar("Loss/val", val_loss, e)
#             writer.add_scalar("Accuracy/val", val_acc, e)
            
#             # Save the best validation accuracy and the corresponding model.
#             if best_val_acc < val_acc:
#                 best_val_acc = val_acc.item()
#                 best_path = 'saved_models_agg/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
#                 checkpoint = {
#                     'epoch': e + 1,
#                     'model_state_dict': teacher_model.state_dict(),
#                     'val_acc': best_val_acc,
#                     'partition_size': partition.num_nodes(),
#                     'train_size': sum(partition.ndata['train_mask'] == True),
#                     'val_size': sum(partition.ndata['val_mask'] == True),
#                     'total_val_size': sum(dataset_dgl.ndata['val_mask'] == True),
#                     'val_mask': partition.ndata['val_mask'],
#                     'test_mask': partition.ndata['test_mask'],
#                     'args':args
#                 }
#                 if args.k > 1:
#                     checkpoint['node_ids'] = partition.ndata['og_ids']
#                 torch.save(checkpoint, best_path)

#             # Backward
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             # print the current results
#             if e % 20 == 0:
#                 print(
#                     'In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
#                         e, train_loss, train_acc, val_loss, val_acc, best_val_acc))

# # ########################################STUDENT MODEL STARTS##########################################################

#         # Obtain Distilled Model and logits
#         teacher_checkpoint = torch.load(best_path)
#         teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
#         teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
#         teacher_model.eval()
#         student_model.train()
#         teacher_model.to(device)

#         best_student_val_acc = 0

#         for i in range(100):

#             # Forward
#             logits = teacher_model(partition, features)
#             student_logits = student_model(partition, features)

#             # distillation_loss = abs(student_logits - logits)
#             alpha = 0.1
#             Temperature = 1
#             student_train_loss = loss_fn_kd(student_logits[train_mask], labels[train_mask], logits[train_mask], alpha, Temperature)

#             student_pred = student_logits.argmax(1)

#             # Compute loss
#             # Note that you should only compute the losses of the nodes in the training set.
#             # Compute accuracy on training dataset
#             student_train_acc = (student_pred[train_mask] == labels[train_mask]).float().mean()
#             writer.add_scalar("Loss/train", student_train_loss.item(), i)
#             writer.add_scalar("Accuracy/train", student_train_acc, i)

#             # evaluate on the validation set
#             student_val_acc, student_val_loss = student_validate(teacher_model, student_model, partition, alpha, Temperature)
#             student_val_loss = student_val_loss.item()
#             writer.add_scalar("Loss/val", student_val_loss, i)
#             writer.add_scalar("Accuracy/val", student_val_acc, i)

#             # Save the validation accuracy
#             #if best_distillation_loss > student_val_loss:
#             if best_student_val_acc < student_val_acc:
#                 best_distillation_loss = student_val_loss
#                 best_student_val_acc = student_val_acc.item()
#                 best_path = 'saved_models/best_student_validation_' + str(args.compression_rate) + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth' 
#                 checkpoint = {
#                     'epoch': i + 1,
#                     'model_state_dict': student_model.state_dict(),
#                     'val_acc': best_student_val_acc,
#                     'best_distillation_loss': best_distillation_loss,
#                     'partition_size': partition.num_nodes(),
#                     'train_size': sum(partition.ndata['train_mask'] == True),
#                     'val_size': sum(partition.ndata['val_mask'] == True),
#                     'total_val_size': sum(dataset_dgl.ndata['val_mask'] == True),
#                     'val_mask': partition.ndata['val_mask'],
#                     'test_mask': partition.ndata['test_mask'],
#                     'args':args
#                 }
#                 if args.k > 1:
#                     checkpoint['node_ids'] = partition.ndata['og_ids']
#                 torch.save(checkpoint, best_path)

#             # Backward
#             student_optimizer.zero_grad()
#             student_train_loss.backward()
#             student_optimizer.step()

#             # print the current results
#             if i % 20 == 0:
#                 print(
#                     'In epoch {}, student train loss: {:.3f}, student train acc: {:.3f}, student val loss: {:.3f}, student val acc: {:.3f} (best student val acc: {:.3f}))'.format(
#                         i, student_train_loss, student_train_acc, student_val_loss, student_val_acc, best_student_val_acc))


        # print("p", idx, " best teacher validation accuracy --> ", best_val_acc)
        # print("p", idx, " best student validation accuracy --> ", best_student_val_acc)
        # print("p", idx, " best student validation loss --> ", best_distillation_loss)

# validation function
def validate(model, partition):
    # put in evaluation mode
    model.eval()

    # get the gnn parameters
    features = partition.ndata['feat']
    labels = partition.ndata['label']
    val_mask = partition.ndata['val_mask']

    with torch.no_grad():
        # Forward
        logits = model(partition, features)
        pred = logits.argmax(1)

        # Compute loss and accuracy for this partition
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        return val_acc, val_loss


def student_validate(model, student_model, partition, alpha, Temperature):
    # put in evaluation mode
    model.eval()
    student_model.eval()

    # get the gnn parameters
    features = partition.ndata['feat']
    labels = partition.ndata['label']
    val_mask = partition.ndata['val_mask']

    with torch.no_grad():
        # Forward
        logits = model(partition, features)

        student_logits = student_model(partition, features)
        student_pred = student_logits.argmax(1)

        # Compute loss and accuracy for this partition
        val_loss = loss_fn_kd(student_logits[val_mask], labels[val_mask], logits[val_mask], alpha, Temperature)
        val_acc = (student_pred[val_mask] == labels[val_mask]).float().mean()
        return val_acc, val_loss


# ================================ datasets =====================================
def load_dataset(dataset):
    if dataset == "cora":
        return load_cora()
    elif dataset == "citeseer":
        return load_citeseer()
    elif dataset == "arxiv":
        return load_arxiv()
    elif dataset == "ogbn_products":
        return load_ogbn_products()

# ================================ models =====================================


def load_model(model, features, num_classes, heads, dropout):
    length = features.shape[1]
    if model == "GCN":
        return GCN(length, length, num_classes, dropout)
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
    parser.add_argument("--dataset",help="name of the dataset (cora, citeseeor, arxiv, ogbn_products)",type=str, default="cora")
    parser.add_argument("--heads",help="If using GAT provide num_heads, otherwise enter 0",type=int, default=3)
    parser.add_argument("--dropout", help="Dropout rate. 1 - keep_probability = dropout rate", type=float, default=0.25)
    parser.add_argument("--no_cuda", help="If True then will disable CUDA training", type=bool, default=False)
    parser.add_argument("--student_only", help="If True then will train student model only", type=bool, default=False)
    parser.add_argument("--compression_rate", help="compression rate for KD (big, medium, small)", type=str, default="medium")

    args = parser.parse_args()
    print(args)
    return args


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, Temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = Temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

# ===================================== Main =====================================
if __name__ == "__main__":
    print("=================")
    #os.environ['METIS_DLL'] = '/home/mustafa/Documents/Git_Repos/ECE381K_RWN_Project/scripts/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    args = parse_args()
    train(args)
