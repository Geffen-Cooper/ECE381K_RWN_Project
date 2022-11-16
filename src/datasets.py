# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

from dgl.data import CoraGraphDataset,CiteseerGraphDataset,AsNodePredDataset
from dgl import to_networkx,add_reverse_edges

from torch import LongTensor
import networkx as nx
from ogb.nodeproppred import DglNodePropPredDataset


def load_cora():
    dataset = CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)

    cora_dgl = dataset[0]
    cora_nx = to_networkx(cora_dgl)

    components = nx.strongly_connected_components(cora_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = LongTensor(list(set(cora_nx.nodes()).difference(largest_component)))

    cora_dgl.remove_nodes(disconnected_nodes)
    cora_dgl.num_nodes()

    cora_c_nx = to_networkx(cora_dgl)
    return cora_c_nx, cora_dgl, dataset


def load_citeseer():
    dataset = CiteseerGraphDataset()
    print('Number of categories:', dataset.num_classes)

    citeseer_dgl = dataset[0]
    citeseer_nx = dgl.to_networkx(citeseer_dgl)

    components = nx.strongly_connected_components(citeseer_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = LongTensor(list(set(citeseer_nx.nodes()).difference(largest_component)))

    citeseer_dgl.remove_nodes(disconnected_nodes)
    citeseer_dgl.num_nodes()

    citeseer_c_nx = to_networkx(citeseer_dgl)
    return citeseer_c_nx, citeseer_dgl, dataset


def load_arxiv():
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    device = 'cpu'  # change to 'cuda' for GPU
    arxiv_dgl = AsNodePredDataset(dataset)[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    arxiv_dgl = add_reverse_edges(arxiv_dgl)

    print('Number of categories:', dataset.num_classes)

    arxiv_nx = to_networkx(arxiv_dgl)
    return arxiv_nx, arxiv_dgl, dataset


def load_ogbn_proteins():
    dataset = DglNodePropPredDataset('ogbn-proteins')
    device = 'cpu'  # change to 'cuda' for GPU
    print(dataset)
    proteins_dgl = AsNodePredDataset(dataset)#[0]
    print(proteins_dgl)
    exit()
    # Add reverse edges since ogbn-arxiv is unidirectional.
    proteins_dgl = dgl.add_reverse_edges(proteins_dgl)

    print('Number of categories:', dataset.num_classes)

    proteins_nx = dgl.to_networkx(proteins_dgl)
    return proteins_nx, proteins_dgl, dataset



