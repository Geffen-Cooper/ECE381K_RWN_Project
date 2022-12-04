'''
    This file runs before train parallel to generate
    the partitions and save them to files.
'''

# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

import argparse
import dgl
import dgl.data

# our modules
from models import *
from datasets import *
from partition_graph import *


def create_partitions(args):
    print("Number of partitions:", args.k)
    print("Dataset:", args.dataset)

    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, dataset = load_dataset(args.dataset)

    if args.k == 1:
        partitions = [dataset_dgl]
    else:
        partitions, parts_tensor = partition_network(args.k, dataset_nx, dataset_dgl)

    # saving each partition
    for idx, partition in enumerate(partitions):
        subgraph_path = 'graph_partitions/' + str(args.dataset) + '_p' + str(idx + 1) + '_k' + str(args.k) + '.bin'
        dgl.save_graphs(subgraph_path,partition)



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



# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument("--k",help="how many partitions to split the input graph into",type=int, default=2)
    parser.add_argument("--dataset",help="name of the dataset (cora, citeseeor, arxiv, ogbn_products)",type=str, default="cora")

    args = parser.parse_args()
    print(args)
    return args



# ===================================== Main =====================================
if __name__ == "__main__":
    args = parse_args()
    create_partitions(args)
