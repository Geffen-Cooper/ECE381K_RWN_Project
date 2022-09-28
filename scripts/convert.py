'''
Use this script to load csv type graph files into networkx, merge the links and node labels,
and output to a Gephi compatible format
'''

import os
import networkx as nx
import pandas as pd

# file path information
data_dir = "../data"
cora_edges_file = "cora/cora.cites"
cora_attributes_file = "cora/cora.content"

# read the edge lists into pandas dataframe
edgelist = pd.read_csv(os.path.join(data_dir, cora_edges_file), sep='\t', header=None, names=["target", "source"])

# store into networkx graph (directed)
Gnx = nx.from_pandas_edgelist(edgelist).to_directed()

# get the cora features and label as a pandas dataframe
feature_names = ["w_{}".format(ii) for ii in range(1433)]
column_names =  feature_names + ["subject"]
node_data = pd.read_csv(os.path.join(data_dir, cora_attributes_file), sep='\t', header=None, names=column_names)

# convert from dataframe to dictionary (only the labels column)
node_labels = node_data['subject'].to_dict()

# add node attributes to networkx graph
nx.set_node_attributes(Gnx,node_labels,name="label")

# save to output file
nx.write_gexf(Gnx,os.path.join(data_dir,"cora.gexf"))
