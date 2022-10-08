import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
import time

from dgl.nn import GraphConv
from dgl.data import load_data
from dgl.nn.pytorch import GATConv

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
    def __init__(self,
                 g,
                 num_layers,
                 in_feats,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=F.elu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.g = g
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats=in_feats,
                                       out_feats=num_hidden,
                                       num_heads=heads[0],
                                       feat_drop=0.,
                                       attn_drop=0.,
                                       negative_slope=negative_slope,
                                       activation=activation))
        # hidden layers
        for l in range(num_layers - 2):
            # due to multi-head, the in_feats = num_hidden * num_heads
            self.gat_layers.append(GATConv(in_feats=num_hidden * heads[l],
                                           out_feats=num_hidden,
                                           num_heads=heads[l + 1],
                                           feat_drop=feat_drop,
                                           attn_drop=attn_drop,
                                           negative_slope=negative_slope,
                                           activation=activation))
        # output projection
        self.gat_layers.append(GATConv(in_feats=num_hidden * heads[-2],
                                       out_feats=num_classes,
                                       num_heads=heads[-1],
                                       feat_drop=feat_drop,
                                       attn_drop=attn_drop,
                                       negative_slope=negative_slope,
                                       activation=None))

    def forward(self, h):
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits