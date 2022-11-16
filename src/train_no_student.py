'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function
'''

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

import time 
time.sleep(1)
print("pre-import\n",time.time(),flush=True)
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
print("start-prog\n", time.time(),flush=True)
time.sleep(1)

def train(args):
    device = "cpu"
    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset)

    if args.k == 1:
        partitions = [dataset_dgl]
    else:
        partitions, parts_tensor = partition_network(args.k, dataset_nx, dataset_dgl)
    print("load-data\n", time.time(),flush=True)
    time.sleep(1)
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

        # create a gnn for this partition using graph parameters
        teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
        # student_model = load_student_model(args.gnn, features, num_classes, args.heads,
        #                            args.dropout, args.compression_rate) 

        # NEW ADDITION: count model params
        print("# teacher params",sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))
        # print("# student params",sum(p.numel() for p in student_model.parameters() if p.requires_grad))

        teacher_model.to(device)
        # student_model.to(device)

        # create the optimizer
        optimizer = Adam(teacher_model.parameters(), lr=0.01)
        print("load-model\n", time.time(),flush=True)
        time.sleep(1)
        # student_optimizer = Adam(student_model.parameters(), lr=0.01)
        best_val_acc = 0
        best_path = 'none'

        teacher_model.train()

        # Initializing distillation loss
        distillation_loss = 1000000000  # Very large number
        best_distillation_loss = distillation_loss

        
        # train the subgraph
        for e in range(100):
            # Forward
            logits = teacher_model(partition, features)
            # g = make_dot(logits)
            # g.view()
            # Compute prediction
            pred = logits.argmax(1)
            print("forward\n", time.time(),flush=True)
            time.sleep(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            train_loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training dataset
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()

            # evaluate on the validation set
            val_acc, val_loss = validate(teacher_model, partition)
            
            # Save the best validation accuracy and the corresponding model.
            if best_val_acc < val_acc:
                best_val_acc = val_acc.item()
                best_path = 'saved_models/best_' + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth'
                checkpoint = {
                    'epoch': e + 1,
                    'model_state_dict': teacher_model.state_dict(),
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
                save(checkpoint, best_path)
            
            print("loss-val\n", time.time(),flush=True)
            time.sleep(2)
            

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print("backwards\n", time.time(),flush=True)
            time.sleep(1)
            exit()
            

            # print the current results
            if e % 20 == 0:
                print(
                    'In epoch {}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f} (best val acc: {:.3f}))'.format(
                        e, train_loss, train_acc, val_loss, val_acc, best_val_acc))

# ########################################STUDENT MODEL STARTS##########################################################

        # # Obtain Distilled Model and logits
        # teacher_checkpoint = load(best_path)
        # teacher_model = load_model(args.gnn, features, num_classes, args.heads, args.dropout)
        # teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
        # teacher_model.eval()
        # student_model.train()
        # teacher_model.to(device)

        # best_student_val_acc = 0

        # for i in range(100):

        #     # Forward
        #     logits = teacher_model(partition, features)
        #     student_logits = student_model(partition, features)

        #     # distillation_loss = abs(student_logits - logits)
        #     alpha = 0.1
        #     Temperature = 1
        #     student_train_loss = loss_fn_kd(student_logits[train_mask], labels[train_mask], logits[train_mask], alpha, Temperature)

        #     student_pred = student_logits.argmax(1)

        #     # Compute loss
        #     # Note that you should only compute the losses of the nodes in the training set.
        #     # Compute accuracy on training dataset
        #     student_train_acc = (student_pred[train_mask] == labels[train_mask]).float().mean()

        #     # evaluate on the validation set
        #     student_val_acc, student_val_loss = student_validate(teacher_model, student_model, partition, alpha, Temperature)
        #     student_val_loss = student_val_loss.item()

        #     # Save the validation accuracy
        #     #if best_distillation_loss > student_val_loss:
        #     if best_student_val_acc < student_val_acc:
        #         best_distillation_loss = student_val_loss
        #         best_student_val_acc = student_val_acc.item()
        #         best_path = 'saved_models/best_student_validation_' + str(args.compression_rate) + str(args.gnn) + '_' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.pth' 
        #         checkpoint = {
        #             'epoch': i + 1,
        #             'model_state_dict': student_model.state_dict(),
        #             'val_acc': best_student_val_acc,
        #             'best_distillation_loss': best_distillation_loss,
        #             'partition_size': partition.num_nodes(),
        #             'train_size': sum(partition.ndata['train_mask'] == True),
        #             'val_size': sum(partition.ndata['val_mask'] == True),
        #             'total_val_size': sum(dataset_dgl.ndata['val_mask'] == True),
        #             'val_mask': partition.ndata['val_mask'],
        #             'test_mask': partition.ndata['test_mask'],
        #             'args':args
        #         }
        #         if args.k > 1:
        #             checkpoint['node_ids'] = partition.ndata['og_ids']
        #         save(checkpoint, best_path)

        #     # Backward
        #     student_optimizer.zero_grad()
        #     student_train_loss.backward()
        #     student_optimizer.step()

# validation function
def validate(model, partition):
    # put in evaluation mode
    model.eval()

    # get the gnn parameters
    features = partition.ndata['feat']
    labels = partition.ndata['label']
    val_mask = partition.ndata['val_mask']

    with no_grad():
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

    with no_grad():
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
    elif dataset == "proteins":
        return load_ogbn_proteins()

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
    KD_loss = KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss












def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot






# ===================================== Main =====================================
if __name__ == "__main__":
    # print("start-prog\n", time.time(),flush=True)
    time.sleep(1)
    args = parse_args()
    train(args)
