# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from datasets import *


# Partitions the dataset input into k ways
def partition_network(k, dataset_nx, dataset_dgl):
    print("partitioning ", k, " subgraphs")

    # partition into k parts
    _, parts = metis.part_graph(dataset_nx, k)

    parts_tensor = torch.Tensor(parts)
    subgraphs = []
    # for each partition
    for i in range(k):
        # get the node ids for the partition
        sg_nodes = (parts_tensor == i).nonzero()[:, 0].tolist()

        # make it a dgl subgraph and store the og ids from the whole graph
        sg = dgl.node_subgraph(dataset_dgl, sg_nodes)
        sg.ndata['og_ids'] = torch.LongTensor(sg_nodes)

        # get the dgl subgraph from the node ids
        subgraphs.append(sg)

    return subgraphs, parts_tensor
