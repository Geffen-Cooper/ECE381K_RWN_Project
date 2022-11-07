import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import *

models = ["GCN"]#,"GAT","GSAGE"]
ks = [1,2,5,10,20]
datasets = ["cora","citeseer"]#,"arxiv"]


# iterate over the datasets
for dataset in datasets:
    # first load the dataset, split into k partitions
    dataset_nx, dataset_dgl, data = load_dataset(dataset)
    val_mask = dataset_dgl.ndata['val_mask']
    labels = dataset_dgl.ndata['label']

    # store the accuracy for each GNN model
    model_accs = []
    plt.figure()
    # iterate over the models
    for idx,model in enumerate(models):
        # store the overall accuracy for each k-way partitioned graph
        k_accs = []
        # iterate over the possible ks
        for k in ks:
            # store the predictions over all partitions
            preds = torch.zeros(len(dataset_dgl.ndata['val_mask']),dtype=torch.long)
            # evaluate each partition in this k
            for partition in range(k):
                # load the partition checkpoint
                checkpoint_path = "saved_models/best_"+str(model)+"_"+str(dataset)+"_"+"p"+str(partition+1)+"_k"+str(k)+".pth"
                checkpoint = torch.load(checkpoint_path)
                args = checkpoint['args']

                # get the partition node ids to create the dgl subgraph
                sg = dataset_dgl
                sg_node_ids = dataset_dgl.nodes()
                if args.k > 1: 
                    sg_node_ids = checkpoint['node_ids'].cpu()
                    sg = dgl.node_subgraph(dataset_dgl, sg_node_ids)

                # create a gnn for this partition using graph parameters
                features = sg.ndata['feat']
                
                # load the model
                m = load_model(args.gnn,features,data.num_classes,args.heads, args.dropout)
                m.load_state_dict(checkpoint['model_state_dict'])
                m.eval()

                # forward pass for evaluation
                with torch.no_grad():
                    # Forward
                    logits = m(sg, features)
                    pred = logits.argmax(1)

                    # store the predictions
                    preds[sg_node_ids] = pred
            # Compute loss and accuracy
            #val_loss = F.cross_entropy(torch.FloatTensor(preds[val_mask]), labels[val_mask]) TODO: gives error, need to compare loss as well
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean()
            k_accs.append(val_acc)

        # model_accs.append(partition_accs)
        plt.plot(ks,k_accs)
        plt.suptitle('Plot of accuracy versus k-way partitions: '+dataset, fontsize=12)
        plt.xlabel('K partitions', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.yticks(np.arange(0.5, 1, 0.1))
        plt.xticks([1,2,5,10,20])
    plt.legend(models)
plt.show()
