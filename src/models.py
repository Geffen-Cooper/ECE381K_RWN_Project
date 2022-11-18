# ECE381K RWN Project
# Graph Neural Network Compression for Edge Devices
# Mustafa Munir and Geffen Cooper

from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Module
import torch.nn.functional as F

from dgl.nn import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn import SAGEConv


class GCN(Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        #self.bn1 = BatchNorm1d(hidden_feats)
        self.dropout = Dropout(p=dropout)
        self.conv2 = GraphConv(hidden_feats, num_classes)

    def forward(self, g, in_feat):
        # print("running forward")
        # print("self id forward: ", id(self))
        h = self.conv1(g, in_feat)
        # print("running conv1")
        #h = self.bn1(h)
        h = F.relu(h)
        # print("running relu")
        h = self.dropout(h)
        # print("running dropout")
        h = self.conv2(g, h)
        # print("running conv2")
        return h


# GCN L1 STUDENT NOT USED
class GCNStudent(Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout):
        super(GCNStudent, self).__init__()
        self.conv1 = GraphConv(in_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        return h

class GAT(Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_heads, dropout):
        super(GAT, self).__init__()
        # print(in_feats, hidden_feats, num_heads)
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=int(num_heads))
        self.dropout = Dropout(p=dropout)
        self.conv2 = GATConv(hidden_feats*int(num_heads), num_classes, 1)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        # concatenate the last 2 dimensions num_heads * out_dimension
        h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = h.squeeze()
        return h


class GraphSage(Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type = "pool")
        self.dropout = Dropout(p=dropout)
        self.conv2 = SAGEConv(hidden_feats, num_classes, aggregator_type = "pool")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h
