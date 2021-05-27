from cdt.data import load_dataset
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

data, graph_truth = load_dataset('sachs')

# pos = nx.layout.spring_layout(graph_truth)
# pos = nx.layout.kamada_kawai_layout(graph_truth)
pos = nx.nx_pydot.graphviz_layout(graph_truth, prog='dot')

nodes = nx.draw_networkx_nodes(graph_truth, pos, node_color='w', alpha=0)
edges = nx.draw_networkx_edges(graph_truth,
                               pos,
                               edge_color='blue',
                               alpha=0.5,
                               arrowstyle='->',
                               arrowsize=10,
                               width=2)
labels = nx.draw_networkx_labels(graph_truth, pos)

ax = plt.gca()
ax.set_axis_off()
plt.show()