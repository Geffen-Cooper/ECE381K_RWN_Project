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
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats)
        self.conv2 = GATConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GraphSage(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSage, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
