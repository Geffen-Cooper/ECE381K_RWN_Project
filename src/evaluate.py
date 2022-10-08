import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import *

models = ["GCN","GAT","GSAGE"]
ks = [1,2,3]
datasets = ["cora","citeseer","arxiv"]

# iterate over the datasets
for dataset in datasets:
    # model_accs = []
    plt.figure()
    # iterate over the models
    for idx,model in enumerate(models):
        partition_accs = []
        # iterate over the possible ks
        for k in ks:
            avg_acc = 0
            # avg over each partition for a given k
            for partition in range(k):
                # checkpoint_path = "best_"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition)+"_k"+str(k)+".pth"
                # checkpoint = torch.load(checkpoint_path)
                # avg_acc += checkpoint['val_acc']
                avg_acc += idx
            avg_acc /= k
            # avg acc for a given partition, should have len(ks) of these
            partition_accs.append(avg_acc)
        # model_accs.append(partition_accs)
        plt.plot(partition_accs)
        plt.suptitle('Plot of accuracy versus k-way partitions: '+dataset, fontsize=12)
        plt.xlabel('K partitions', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.yticks(np.arange(0.5, 1, 0.1))
        plt.xticks(np.arange(1, 4, 1))
    plt.legend(models)
plt.show()
