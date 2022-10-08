import dgl
import torch
import dgl.data
import networkx as nx
import metis
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import *


def evaluate(ks, accs):
    print(accs)

    ks.append(1)
    ks.sort()
    print(ks)

    plt.plot(ks, accs)
    plt.suptitle('Plot of accuracy versus k-way partitions', fontsize=12)
    plt.xlabel('K partitions', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.yticks(np.arange(0.5, 1, 0.1))
    plt.xticks(np.arange(1, 4, 1))
    plt.show()
