from cdfnet import fit
from experiments.ci_test import ci_data
import torch as t
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
r = robjects.r

r['source']('experiments/ci_test/partial-copula-CI-test/dgp.R')
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


def fit_ci_data():
  h = fit.get_default_h()
  fn = r[h.dataset]()

  # Sample data
  data_raw = ci_data.ci_data(fn)
  data = load_dataset(h, data_raw)

  # Fit MDMA
  outs = fit.fit_neural_copula(h, data)

  # Extract U1/U2
  with t.no_grad():
    u1 = []
    u2 = []
    for batch_idx, batch in enumerate(data[0]):
      batch_data = batch[0].to(device).squeeze()
      condCDF_0, condCDF_1 = get_condCDFs(outs, batch_data)
      u1.append(condCDF_0)
      u2.append(condCDF_1)

  u1 = t.cat(u1).cpu().numpy()
  u2 = t.cat(u2).cpu().numpy()
  test_ci_r = r['test_CI'](u1, u2)
  test_ci = [xx for xx in test_ci_r]
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