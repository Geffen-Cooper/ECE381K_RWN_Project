import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from datasets import *


# Partitions the dataset input into k ways
def partition(k, dataset):
    print("partitioning ", k, " subgraphs")

    # partition into k parts
    _, parts = metis.part_graph(dataset, k)

    parts_tensor = torch.Tensor(parts)

    # for each partition
    for i in range(k):
        # get the node ids for the partition
        sg_nodes = (parts_tensor == i).nonzero()[:, 0].tolist()

        # get the dgl subgraph from the node ids
        subgraph = dgl.node_subgraph(dataset, sg_nodes)

    return subgraph, parts_tensor
