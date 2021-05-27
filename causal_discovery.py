from cdfnet import fit
from experiments.causal_discovery.pcalg import estimate_skeleton, estimate_cpdag
from cdfnet import utils
from cdt.data import load_dataset
from cdt.metrics import SHD, SID
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import write_gpickle, read_gpickle, read_graphml

import rpy2.robjects as robjects
r = robjects.r

from rpy2.robjects import numpy2ri
numpy2ri.activate()

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.packages import importr
pcalg = importr("pcalg")

r['source']('experiments/causal_discovery/ci_test/parCopCITest.R')
r['source']('experiments/causal_discovery/pc_fit.R')

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
  device = "cuda"
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')
  device = "cpu"


def plot_dag(graph_truth=nx.read_gpickle('data/graph_truth.gpickle'),
             graph_mdma=nx.read_gpickle('data/graph_mdma.gpickle')):
  graph_pc = nx.read_graphml('data/graph_pc.txt')
  graph_pc_nodes = open('data/graph_pc_nodes.txt').read().splitlines()
  graph_pc = nx.relabel_nodes(graph_pc,
                              dict(zip(graph_pc.nodes, graph_pc_nodes)))
  pos = nx.nx_pydot.graphviz_layout(graph_truth, prog='dot')
  figure, axis = plt.subplots(1, 3)
  nx.draw_networkx_nodes(graph_truth, pos, ax=axis[0], node_color='w', alpha=0)
  nx.draw_networkx_nodes(graph_truth, pos, ax=axis[1], node_color='w', alpha=0)
  nx.draw_networkx_nodes(graph_truth, pos, ax=axis[2], node_color='w', alpha=0)
  nx.draw_networkx_edges(graph_truth,
                         pos,
                         ax=axis[0],
                         edge_color='blue',
                         alpha=0.5,
                         arrowstyle='->',
                         arrowsize=10,
                         width=2)
  nx.draw_networkx_edges(graph_pc,
                         pos,
                         ax=axis[1],
                         edge_color='red',
                         alpha=0.5,
                         arrowstyle='->',
                         arrowsize=10,
                         width=2)
  nx.draw_networkx_edges(graph_mdma,
                         pos,
                         ax=axis[2],
                         edge_color='green',
                         alpha=0.5,
                         arrowstyle='->',
                         arrowsize=10,
                         width=2)
  nx.draw_networkx_labels(graph_truth, pos, ax=axis[0])
  nx.draw_networkx_labels(graph_truth, pos, ax=axis[1])
  nx.draw_networkx_labels(graph_truth, pos, ax=axis[2])
  axis[0].set_title("True DAG")
  axis[1].set_title("Recovered CPDAG (Gaussian PC)")
  axis[2].set_title("Recovered CPDAG (MDMA PC)")
  figure.tight_layout()
  plt.show()

  res = [
      SHD(graph_truth, graph_mdma),
      SHD(graph_truth, graph_pc),
      SHD(graph_truth, graph_mdma, False),
      SHD(graph_truth, graph_pc, False)
  ]
  return res


def pc_fit(data_pd):
  with localconverter(robjects.default_converter + pandas2ri.converter):
    data_r = robjects.conversion.py2rpy(data_pd)
  r['pc_fit'](data_r)


def causal_discovery():

  h = fit.get_default_h()

  if h.dataset == 'sachs':
    data_pd, graph_truth = load_dataset('sachs')
    data_np = np.array(data_pd)
    data = utils.create_loaders([data_np, None, None], h.batch_size)
  else:
    raise RuntimeError()

  # Gaussian PC
  print('Gaussian PC')
  pc_fit(data_pd)

  h.d = data_np.shape[1]
  h.M = data_np.shape[0]
  h.eval_validation = False
  h.eval_test = False

  # Fit MDMA
  print('Fitting MDMA')
  model = fit.fit_neural_copula(h, data)

  # MDMA PC
  print('MDMA PC')
  (graph_mdma, sep_set) = estimate_skeleton(model, data, alpha=0.01)
  graph_mdma = estimate_cpdag(skel_graph=graph_mdma, sep_set=sep_set)
  graph_mdma = nx.relabel_nodes(graph_mdma,
                                dict(zip(graph_mdma, graph_truth.nodes)))

  write_gpickle(graph_truth, 'data/graph_truth.gpickle')
  write_gpickle(graph_mdma, 'data/graph_mdma.gpickle')

  d = plot_dag(graph_truth, graph_mdma)
  print(d)
  return d


if __name__ == '__main__':
  causal_discovery()
