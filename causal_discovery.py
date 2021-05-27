from cdfnet import fit
from experiments.ci_test import ci_data
from cdfnet import utils
from cdt.data import load_dataset
import torch as t
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
r = robjects.r

r['source']('experiments/ci_test/partial-copula-CI-test/parCopCITest.R')

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
  device = "cuda"
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')
  device = "cpu"


def get_condCDFs(model, x):
  dd = model.d - 2
  cond_inds = [j + 2 for j in range(0, dd)]
  inds_0 = cond_inds.copy()
  inds_1 = cond_inds.copy()
  inds_0.append(0)
  inds_1.append(1)
  condCDF_0 = model.condCDF(dd, x[:, cond_inds], inds_0)(x[:, 0])
  condCDF_1 = model.condCDF(dd, x[:, cond_inds], inds_1)(x[:, 1])
  return condCDF_0, condCDF_1


def causal_discovery():

  h = fit.get_default_h()

  if h.dataset == 'sachs':
    data_np, graph = load_dataset('sachs')
    data_np = np.array(data_np)
    data = utils.create_loaders([data_np, data_np, data_np], h.batch_size)
  else:
    raise RuntimeError()

  h.d = data_np.shape[1]
  h.M = data_np.shape[0]
  h.M_val = data_np.shape[0]

  # Fit MDMA
  # import pdb
  # pdb.set_trace()
  mdma = fit.fit_neural_copula(h, data)


if __name__ == '__main__':
  causal_discovery()