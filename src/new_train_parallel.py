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
from torch import multiprocessing
from functools import partial

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
    device = 'cpu' # To override to cpu
    if device == 'cpu':
        print("CPU cores count: " + str(multiprocessing.cpu_count()))
    print(device)

    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset)

    if args.k == 1:
        partitions = [dataset_dgl]
    else:
        partitions, parts_tensor = partition_network(args.k, dataset_nx, dataset_dgl)


    # print("partitions: \n", partitions)

    num_classes = dataset.num_classes

    # training each partition
    # index = 0
    # for idx, partition in enumerate(partitions):
    #      print("idx: \n", idx)
    #      print("partition: \n", partition)
    #     globals()["partition_%d"%idx] = partition

    # Train each partition in parallel. For all k partitions.
    start_time = time.perf_counter()
    with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
        func = partial(train_parallel, num_classes, args)
        pool.map(func, partitions)
        print("Intermediate debug statement")
        pool.close()
        pool.join()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("Time used for all processes: ", total_time) 

    '''
    if args.k == 1:
        start_time = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
            func = partial(train_parallel, num_classes, args)
            pool.map(func, partitions)
            print("Intermediate debug statement")
            pool.close()
            pool.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Time used for all processes: ", total_time)
    elif args.k == 2:
        start_time = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
            func = partial(train_parallel, num_classes, args)
            pool.map(func, partitions)
            print("Intermediate debug statement")
            pool.close()
            pool.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Time used for all processes: ", total_time)
    elif args.k == 5:
        start_time = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
            func = partial(train_parallel, num_classes, args)
            pool.map(func, partitions)
            print("Intermediate debug statement")
            pool.close()
            pool.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Time used for all processes: ", total_time)
    elif args.k == 10:
        start_time = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
            func = partial(train_parallel, num_classes, args)
            pool.map(func, partitions)
            print("Intermediate debug statement")
            pool.close()
            pool.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Time used for all processes: ", total_time)
    else:
        start_time = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(multiprocessing.cpu_count()) as pool:
            func = partial(train_parallel, num_classes, args)
            pool.map(func, partitions)
            print("Intermediate debug statement")
            pool.close()
            pool.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("Time used for all processes: ", total_time)
    '''


# def train_parallel(partitions, dataset, dataset_dgl, args, device, writer, idx):
def train_parallel(num_classes, args, partition):
    start_time = time.perf_counter()
    # print("\n\n start time of parallel process: ", start_time)
    # print("partitions function: \n", partition)
    # init tensorboard
    writer = SummaryWriter()

    # graph parameters
    features = partition.ndata['feat']
    labels = partition.ndata['label']  # all labels
    train_mask = partition.ndata['train_mask']
    print("num_classes: ", num_classes)

    partition = partition.to('cpu')
    features = features.to('cpu')
    labels = labels.to('cpu')

    # create a gnn for this partition using graph parameters
    teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
    student_model = load_student_model(args.gnn, features, num_classes, args.heads, args.dropout, args.compression_rate) 

    # NEW ADDITION: count model params
    print("# teacher params",sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
    print("# student params",sum(p.numel() for p in student_model.parameters() if p.requires_grad))

    teacher_model.to('cpu')
    student_model.to('cpu')

    # create the optimizer
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01)
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)
    best_val_acc = 0
    best_path = 'none'

    teacher_model.train()

    # Initializing distillation loss
    distillation_loss = 1000000000  # Very large number
    best_distillation_loss = distillation_loss


    # # train the subgraph
    for e in range(100):
        # print("e: ", e)
        # print("teacher ID: ", id(teacher_model))
        # print("ID partition: ", id(partition))
        # print("ID feature: ", id(features))

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

        # evaluate on the validation set
        val_acc, val_loss = validate(teacher_model, partition)
        writer.add_scalar("Loss/val", val_loss, e)
        writer.add_scalar("Accuracy/val", val_acc, e)
        
        # Save the best validation accuracy and the corresponding model.
        if best_val_acc < val_acc:
            best_val_acc = val_acc.item()
            best_path = 'saved_models/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p_TODO_' + '_k' + str(args.k) + '.pth' #TODO: Add partition number
            checkpoint = {
                'epoch': e + 1,
                'model_state_dict': teacher_model.state_dict(),
                'val_acc': best_val_acc,
                'partition_size': partition.num_nodes(),
                'train_size': sum(partition.ndata['train_mask'] == True),
                'val_size': sum(partition.ndata['val_mask'] == True),
                'val_mask': partition.ndata['val_mask'],
                'test_mask': partition.ndata['test_mask'],
                'args':args
            }
            if args.k > 1:
                checkpoint['node_ids'] = partition.ndata['og_ids']
            torch.save(checkpoint, best_path)

        # Backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # print the current results
        if e % 20 == 0:
            print(
                'In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
                    e, train_loss, train_acc, val_loss, val_acc, best_val_acc))

########################################STUDENT MODEL STARTS##########################################################

    # Obtain Distilled Model and logits
    teacher_checkpoint = torch.load(best_path)
    teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    student_model.train()

    best_student_val_acc = 0

    for i in range(100):

        # Forward
        logits = teacher_model(partition, features)
        student_logits = student_model(partition, features)

        # distillation_loss = abs(student_logits - logits)
        alpha = 0.1
        Temperature = 1
        student_train_loss = loss_fn_kd(student_logits[train_mask], labels[train_mask], logits[train_mask], alpha, Temperature)

        student_pred = student_logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # Compute accuracy on training dataset
        student_train_acc = (student_pred[train_mask] == labels[train_mask]).float().mean()
        writer.add_scalar("Loss/train", student_train_loss.item(), i)
        writer.add_scalar("Accuracy/train", student_train_acc, i)

        # evaluate on the validation set
        student_val_acc, student_val_loss = student_validate(teacher_model, student_model, partition, alpha, Temperature)
        student_val_loss = student_val_loss.item()
        writer.add_scalar("Loss/val", student_val_loss, i)
        writer.add_scalar("Accuracy/val", student_val_acc, i)

        # Save the validation accuracy
        #if best_distillation_loss > student_val_loss:
        if best_student_val_acc < student_val_acc:
            best_distillation_loss = student_val_loss
            best_student_val_acc = student_val_acc.item()
            best_path = 'saved_models/best_student_validation_' + str(args.compression_rate) + str(args.gnn) + '_' + str(args.dataset) + '_p_TODO_' + '_k' + str(args.k) + '.pth' #TODO: Add partition number
            checkpoint = {
                'epoch': i + 1,
                'model_state_dict': student_model.state_dict(),
                'val_acc': best_student_val_acc,
                'best_distillation_loss': best_distillation_loss,
                'partition_size': partition.num_nodes(),
                'train_size': sum(partition.ndata['train_mask'] == True),
                'val_size': sum(partition.ndata['val_mask'] == True),
                'val_mask': partition.ndata['val_mask'],
                'test_mask': partition.ndata['test_mask'],
                'args':args
            }
            if args.k > 1:
                checkpoint['node_ids'] = partition.ndata['og_ids']
            torch.save(checkpoint, best_path)

        # Backward
        student_optimizer.zero_grad()
        student_train_loss.backward()
        student_optimizer.step()

        # print the current results
        if i % 20 == 0:
            print(
                'In epoch {}, student train loss: {:.3f}, student train acc: {:.3f}, student val loss: {:.3f}, student val acc: {:.3f} (best student val acc: {:.3f}))'.format(
                    i, student_train_loss, student_train_acc, student_val_loss, student_val_acc, best_student_val_acc))


    # TODO: This will no longer be best show each partition's index and accuracy just partition accuracy
    print("partition's best teacher validation accuracy --> ", best_val_acc)
    print("partition's best student validation accuracy --> ", best_student_val_acc)
    print("partition's best student validation loss --> ", best_distillation_loss)

    end_time = time.perf_counter()
    print("end time: "+ str(end_time))
    time_used = end_time - start_time
    print("Total time_used: " + str(time_used) + " seconds")

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
