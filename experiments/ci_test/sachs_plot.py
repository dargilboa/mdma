from cdt.data import load_dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

s_data, s_graph = load_dataset('sachs')

G = s_graph
# pos = nx.layout.spring_layout(G)
# pos = nx.layout.kamada_kawai_layout(G)
pos = nx.nx_pydot.graphviz_layout(G, prog='dot')

nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color="w",
    #    node_size=2000,
    #    node_shape='h',
    alpha=0)
edges = nx.draw_networkx_edges(
    G,
    pos,
    edge_color='blue',
    alpha=0.5,
    arrowstyle="->",
    arrowsize=10,
    width=2,
)
labels = nx.draw_networkx_labels(G, pos)

# pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
ax = plt.gca()
ax.set_axis_off()
plt.show()