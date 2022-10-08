import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset


def load_cora():
    dataset = dgl.data.CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)

    cora_dgl = dataset[0]
    cora_nx = dgl.to_networkx(cora_dgl)

    components = nx.strongly_connected_components(cora_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = torch.LongTensor(list(set(cora_nx.nodes()).difference(largest_component)))

    cora_dgl.remove_nodes(disconnected_nodes)
    cora_dgl.num_nodes()

    cora_c_nx = dgl.to_networkx(cora_dgl)
    return cora_c_nx, cora_dgl, dataset


def load_citeseer():
    dataset = dgl.data.CiteseerGraphDataset()
    print('Number of categories:', dataset.num_classes)

    citeseer_dgl = dataset[0]
    citeseer_nx = dgl.to_networkx(citeseer_dgl)

    components = nx.strongly_connected_components(citeseer_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = torch.LongTensor(list(set(citeseer_nx.nodes()).difference(largest_component)))

    citeseer_dgl.remove_nodes(disconnected_nodes)
    citeseer_dgl.num_nodes()

    citeseer_c_nx = dgl.to_networkx(citeseer_dgl)
    return citeseer_c_nx, citeseer_dgl, dataset


def load_arxiv():
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    device = 'cpu'  # change to 'cuda' for GPU
    arxiv_dgl = dgl.data.AsNodePredDataset(dataset)[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    arxiv_dgl = dgl.add_reverse_edges(arxiv_dgl)

    print('Number of categories:', dataset.num_classes)

    arxiv_nx = dgl.to_networkx(arxiv_dgl)
    return arxiv_nx, arxiv_dgl, dataset



