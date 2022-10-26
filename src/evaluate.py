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
ks = [1,2,5,10,20]
datasets = ["cora"] #,"citeseer"]#,"arxiv"]


# iterate over the datasets
for dataset in datasets:
    # first load the dataset, split into k partitions
    # dataset_nx, dataset_dgl, data = load_dataset(dataset)

    
    # model_accs = []
    plt.figure()
    # iterate over the models
    for idx,model in enumerate(models):
        partition_accs = []
        # iterate over the possible ks
        for k in ks:
            # if k == 1:
            #     partitions = [dataset_dgl]
            # else:
            #     partitions, parts_tensor = partition_network(k, dataset_nx, dataset_dgl)
            avg_acc = 0
            # avg over each partition for a given k
            for partition in range(k):
                checkpoint_path = "best_"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                checkpoint = torch.load(checkpoint_path)

                # create a gnn for this partition using graph parameters
                # features = partitions[partition].ndata['feat']
                # labels = partitions[partition].ndata['label']
                # test_mask = partitions[partition].ndata['test_mask']
                # m = load_model(model,features,data.num_classes, 3)
                # m.load_state_dict(checkpoint['model_state_dict'])

                # with torch.no_grad():
                #     # Forward
                #     logits = m(partitions[partition], features)
                #     pred = logits.argmax(1)

                #     # Compute loss and accuracy
                #     test_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
                #     test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

                # avg_acc += test_acc
                avg_acc += checkpoint['val_acc']
            avg_acc /= k
            # avg acc for a given partition, should have len(ks) of these
            partition_accs.append(avg_acc)
        # model_accs.append(partition_accs)
        plt.plot(ks,partition_accs)
        plt.suptitle('Plot of accuracy versus k-way partitions: '+dataset, fontsize=12)
        plt.xlabel('K partitions', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.yticks(np.arange(0.5, 1, 0.1))
        plt.xticks([1,2,5,10,20])
    plt.legend(models)
plt.show()
