from MDMA import fit
from experiments.causal_discovery.ci_test import ci_test
from experiments.causal_discovery.ci_data import ci_data
import torch as t
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
r = robjects.r

r['source']('experiments/causal_discovery/ci_test/dgp.R')
r['source']('experiments/causal_discovery/ci_test/parCopCITest.R')

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
  device = "cuda"
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')
  device = "cpu"


def fit_ci_data():
  h = fit.get_default_h()
  fn = r[h.dataset]()

  # Sample data
  data_raw = ci_data(fn)
  data = load_dataset(h, data_raw)

  # Fit MDMA
  model = fit.fit_MDMA(h, data)

  # CI test
  dd = model.d - 2
  cond_inds = [j + 2 for j in range(0, dd)]
  test_ci = ci_test(model, data, 0, 1, cond_inds)
  print(test_ci[1])


def tensor_dataset(dataset):

  trn = t.tensor(np.expand_dims(dataset.trn.x, 1)).float()
  val = t.tensor(np.expand_dims(dataset.val.x, 1)).float()
  tst = t.tensor(dataset.tst.x).float()

  return trn, val, tst


def load_dataset(h, dataset):

  trn, val, tst = tensor_dataset(dataset)
  trn = t.utils.data.TensorDataset(trn)
  val = t.utils.data.TensorDataset(val)
  tst = t.utils.data.TensorDataset(tst)

  data_loader_train = t.utils.data.DataLoader(trn,
                                              batch_size=h.batch_size,
                                              shuffle=True)

  data_loader_valid = t.utils.data.DataLoader(val,
                                              batch_size=h.batch_size,
                                              shuffle=False)

  data_loader_test = t.utils.data.DataLoader(tst,
                                             batch_size=h.batch_size,
                                             shuffle=False)

  h.d = dataset.n_dims
  h.M = len(trn)
  h.M_val = len(val)

  return data_loader_train, data_loader_valid, data_loader_test


if __name__ == '__main__':
  fit_ci_data()