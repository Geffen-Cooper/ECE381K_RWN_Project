import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import matplotlib.pyplot as plt
import numpy as np
import time

from dgl.nn import GraphConv
from dgl.data import load_data
from dgl.nn.pytorch import GATConv
from dgl.nn import SAGEConv


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_heads):
        super(GAT, self).__init__()
        print(in_feats, hidden_feats, num_heads)
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=int(num_heads))
        self.conv2 = GATConv(hidden_feats, num_classes, num_heads=int(num_heads))

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GraphSage(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type = "pool")
        self.conv2 = SAGEConv(hidden_feats, num_classes, aggregator_type = "pool")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
