import pandas as pd
import networkx as nx

def load_graph(csv_path):
    df = pd.read_csv(csv_path)
    G = nx.Graph()  # Undirected Graph
    for _, row in df.iterrows():
        G.add_edge(int(row["source"]), int(row["target"]), weight=row["weight"])
    return G

# Load both case and control graphs
graph_case = load_graph("graph_case_crc1.csv")
graph_control = load_graph("graph_ctrl_crc1.csv")

print(f"Case Graph: {graph_case.number_of_nodes()} nodes, {graph_case.number_of_edges()} edges")
print(f"Control Graph: {graph_control.number_of_nodes()} nodes, {graph_control.number_of_edges()} edges")

import numpy as np

def load_node_features(csv_path):
    df = pd.read_csv(csv_path)
    df.set_index("taxon_names", inplace=True)
    return df.T.to_numpy()  # Convert to a feature matrix

# Extract node features
features_case = load_node_features("case_crc1.csv")
features_control = load_node_features("control_crc1.csv")

print(f"Case Features Shape: {features_case.shape}")
print(f"Control Features Shape: {features_control.shape}")
