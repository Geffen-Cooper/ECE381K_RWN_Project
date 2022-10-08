import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import *

#Call train within partition
#Call model of training within partition as well.
def partition():
    # repeat experiment for each k where k is the number of partitions
    accs = torch.zeros(3)
    model = GCN(arxiv_dgl.ndata['feat'].shape[1], 64, dataset.num_classes)
    accs[0] = train(arxiv_dgl, model)
    print(accs[0])
    ks = [2, 3]
    for idx, k in enumerate(ks):
        print("training ", k, " subgraphs")
        sg_accs = 0

        # partition into k parts
        _, parts = metis.part_graph(arxiv_nx, k)

        parts_tensor = torch.Tensor(parts)
        sgs = []

        # for each partition
        for i in range(k):
            # get the nodes ids for the partition
            sg_nodes = (parts_tensor == i).nonzero()[:, 0].tolist()

            # get the dgl subgraph from these ids
            sg = dgl.node_subgraph(arxiv_dgl, sg_nodes)

            # generate the model from the subgraph
            model = GCN(sg.ndata['feat'].shape[1], 16, dataset.num_classes)

            # train the subgraph and accumulate the accuracy
            sg_acc = train(sg, model)
            print("train-val-test split: ", sum(sg.ndata['train_mask']), sum(sg.ndata['val_mask']),
                  sum(sg.ndata['test_mask']))
            print("\t trained partition ", i, " size = ", sg.num_nodes(), " acc = ", sg_acc)
            sg_accs += sg_acc

        # save the avg accuracy for this k
        accs[idx + 1] = sg_accs / k
