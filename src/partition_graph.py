# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

from dgl import node_subgraph
from torch import Tensor,LongTensor
from metis import part_graph
from datasets import *


# Partitions the dataset input into k ways
def partition_network(k, dataset_nx, dataset_dgl):
    print("partitioning ", k, " subgraphs")

    # partition into k parts
    _, parts = part_graph(dataset_nx, k, contig=True)

    parts_tensor = Tensor(parts)
    subgraphs = []
    # for each partition
    for i in range(k):
        # get the node ids for the partition
        sg_nodes = (parts_tensor == i).nonzero()[:, 0].tolist()

        # make it a dgl subgraph and store the og ids from the whole graph
        sg = node_subgraph(dataset_dgl, sg_nodes)
        sg.ndata['og_ids'] = LongTensor(sg_nodes)

        # get the dgl subgraph from the node ids
        subgraphs.append(sg)

    return subgraphs, parts_tensor
