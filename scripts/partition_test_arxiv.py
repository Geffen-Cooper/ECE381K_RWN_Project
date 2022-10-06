import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import networkx as nx
import metis
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import matplotlib.pyplot as plt

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats//2)
        self.conv3 = GraphConv(h_feats//2, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #         e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    return best_test_acc


dataset = DglNodePropPredDataset('ogbn-arxiv')
device = 'cpu'      # change to 'cuda' for GPU
arxiv_dgl = dgl.data.AsNodePredDataset(dataset)[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
arxiv_dgl = dgl.add_reverse_edges(arxiv_dgl)

print('Number of categories:', dataset.num_classes)

arxiv_nx = dgl.to_networkx(arxiv_dgl)

# repeat experiment for each k where k is the number of partitions
accs = torch.zeros(3)
model = GCN(arxiv_dgl.ndata['feat'].shape[1], 64, dataset.num_classes)
accs[0] = train(arxiv_dgl, model)
print(accs[0])
ks = [2, 3]
for idx,k in enumerate(ks):
    print("training ", k, " subgraphs")
    sg_accs = 0

    # partition into k parts
    _, parts = metis.part_graph(arxiv_nx,k)

    parts_tensor = torch.Tensor(parts)
    sgs = []

    # for each partition
    for i in range(k):
        # get the nodes ids for the partition
        sg_nodes = (parts_tensor == i).nonzero()[:,0].tolist()

        # get the dgl subgraph from these ids
        sg = dgl.node_subgraph(arxiv_dgl, sg_nodes)

        # generate the model from the subgraph
        model = GCN(sg.ndata['feat'].shape[1], 16, dataset.num_classes)

        # train the subgraph and accumulate the accuracy
        sg_acc = train(sg, model)
        print("train-val-test split: ", sum(sg.ndata['train_mask']),sum(sg.ndata['val_mask']),sum(sg.ndata['test_mask']))
        print("\t trained partition ", i, " size = ", sg.num_nodes()," acc = ", sg_acc)
        sg_accs += sg_acc

    # save the avg accuracy for this k
    accs[idx+1] = sg_accs/k

print(accs)

ks.append(1)
ks.sort()
print(ks)

plt.plot(ks, accs)
plt.suptitle('Arxiv plot of accuracy versus k-way partitions', fontsize=12)
plt.xlabel('K partitions', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.yticks(np.arange(0.5, 0.9, 0.1))
plt.yticks(np.arange(1, 3, 1))
plt.show()