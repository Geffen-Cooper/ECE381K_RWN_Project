# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

from dgl.data import CoraGraphDataset,CiteseerGraphDataset,AsNodePredDataset
from dgl import to_networkx,add_reverse_edges

from torch import LongTensor
import networkx as nx
from ogb.nodeproppred import DglNodePropPredDataset

import dgl
import torch

def load_cora(k=None):
    dataset = CoraGraphDataset()
    print('Number of categories:', dataset.num_classes)

    cora_dgl = dataset[0]
    cora_nx = to_networkx(cora_dgl)

    components = nx.strongly_connected_components(cora_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = LongTensor(list(set(cora_nx.nodes()).difference(largest_component)))

    cora_dgl.remove_nodes(disconnected_nodes)
    cora_dgl.num_nodes()

    if k != 1:
        cora_nx = to_networkx(cora_dgl)
        return cora_nx, cora_dgl, dataset
    return None,cora_dgl,None


def load_citeseer(k=None):
    dataset = CiteseerGraphDataset()
    print('Number of categories:', dataset.num_classes)

    citeseer_dgl = dataset[0]
    citeseer_nx = dgl.to_networkx(citeseer_dgl)

    components = nx.strongly_connected_components(citeseer_nx)
    largest_component = max(components, key=len)

    disconnected_nodes = LongTensor(list(set(citeseer_nx.nodes()).difference(largest_component)))

    citeseer_dgl.remove_nodes(disconnected_nodes)
    citeseer_dgl.num_nodes()

    if k != 1:
        citeseer_nx = to_networkx(citeseer_dgl)
        return citeseer_nx, citeseer_dgl, dataset
    return None,citeseer_dgl,None


def load_arxiv(k=None):
    print("LOAD ARXIV")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    print(dataset)
    print("------\n", dataset[0])

    arxiv_dgl = dgl.data.AsNodePredDataset(dataset)[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    arxiv_dgl = add_reverse_edges(arxiv_dgl)

    print('Number of categories:', dataset.num_classes)
    if k != 1:
        arxiv_nx = to_networkx(arxiv_dgl)
        return arxiv_nx, arxiv_dgl, dataset
    return None,arxiv_dgl,None


def load_ogbn_products():
    print("LOAD PRODUCTS")
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(device)
    dataset = DglNodePropPredDataset('ogbn-products')
    print(dataset)
    print("------\n", dataset[0])

    products_dgl = dgl.data.AsNodePredDataset(dataset)[0]
    # print(products_dgl)
    # exit()
    # Add reverse edges not needed since ogbn-products is undirected
    # products_dgl = dgl.add_reverse_edges(products_dgl)

    print('Number of categories:', dataset.num_classes)

    # products_nx = dgl.to_networkx(products_dgl)
    adj = products_dgl.adj()
    print(len(adj))
    # exit()
    al= []
    import time
    s = time.time()
    for idx in range(len(adj)):
        al.append(adj[idx].coalesce().indices()[0].tolist())
        if idx == 100:
            print(time.time()-s)
            exit()
    print("loaded ogbn products\n")
    return al, products_dgl, dataset



if __name__ == "__main__":
    print("=================")
    #os.environ['METIS_DLL'] = '/home/mustafa/Documents/Git_Repos/ECE381K_RWN_Project/scripts/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    # load_cora()
    # load_citeseer()
    # load_arxiv()
    load_ogbn_products()