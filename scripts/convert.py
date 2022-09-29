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
citeseer_edges_file = "citeseer/citeseer.cites"
citeseer_attributes_file = "citeseer/citeseer.content"

# read the edge lists into pandas dataframe
cora_edgelist = pd.read_csv(os.path.join(data_dir, cora_edges_file), sep='\t', header=None, names=["target", "source"])
citeseer_edgelist = pd.read_csv(os.path.join(data_dir, citeseer_edges_file), sep='\t', header=None, names=["target", "source"], dtype=str)

# store into networkx graph (directed)
cora = nx.from_pandas_edgelist(cora_edgelist, create_using=nx.DiGraph())
citeseer = nx.from_pandas_edgelist(citeseer_edgelist, create_using=nx.DiGraph())

# get the features and label as a pandas dataframe
cora_feature_names = ["w_{}".format(ii) for ii in range(1433)]
cora_column_names =  cora_feature_names + ["subject"]
cora_node_data = pd.read_csv(os.path.join(data_dir, cora_attributes_file), sep='\t', header=None, names=cora_column_names)
citeseer_feature_names = ["w_{}".format(ii) for ii in range(3703)]
citeseer_column_names =  citeseer_feature_names + ["subject"]
citeseer_node_data = pd.read_csv(os.path.join(data_dir, citeseer_attributes_file), sep='\t', header=None, names=citeseer_column_names,low_memory=False)
all_nodes = set(citeseer_node_data.index.values)

# convert from dataframe to dictionary (only the labels column)
cora_node_labels = cora_node_data['subject'].to_dict()
citeseer_node_labels = citeseer_node_data['subject'].to_dict()

# add node attributes to networkx graph
nx.set_node_attributes(cora,cora_node_labels,name="label")
nx.set_node_attributes(citeseer,citeseer_node_labels,name="label")

# remove empty label nodes from citeseer
labeled_nodes = set(citeseer.nodes())
not_labeled = labeled_nodes.difference(all_nodes)
for n in not_labeled:
    citeseer.remove_node(n)

# save to output file
nx.write_gexf(cora,os.path.join(data_dir,"cora.gexf"))
nx.write_gexf(citeseer,os.path.join(data_dir,"citeseer.gexf"))
