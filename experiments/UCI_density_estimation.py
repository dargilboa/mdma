#%%
import fit
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from utils import ROOT_DIR
import os
import utils
os.chdir(ROOT_DIR)
DATA_DIR = '/data/data'
TB_DIR = '/data/tb'

from experiments.BNAF.data.gas import GAS
from experiments.BNAF.data.bsds300 import BSDS300
from experiments.BNAF.data.hepmass import HEPMASS
from experiments.BNAF.data.miniboone import MINIBOONE
from experiments.BNAF.data.power import POWER

np.random.seed(0)
t.manual_seed(0)


def fit_UCI():
  h = fit.get_default_h()
  h.dataset = 'power'
  h.batch_size = 1000
  h.n_epochs = 4
  h.n = 1000
  h.m = 10
  h.L = 6
  h.lr = 5e-2
  h.lambda_l2 = 0
  h.marginals_first = False
  h.marginal_iters = 500
  h.alt_opt = False
  h.incremental = False
  h.add_variable_every = 300
  h.patience = 200
  # #
  # h.M = 300000
  # h.M_val = 50
  # d = 2
  # h.d = d
  # copula_type = 'gumbel'
  # copula_params = 1.67
  # marginal_type = 'gaussian'
  # marginal_params = [np.array([0] * d), np.array([1] * d)]
  # raw_data = utils.generate_data(h.d,
  #                                h.M,
  #                                h.M_val,
  #                                copula_params=copula_params,
  #                                marginal_params=marginal_params,
  #                                copula_type=copula_type,
  #                                marginal_type=marginal_type)
  # h.d = 2
  # raw_data = [
  #     np.random.randn(h.M, h.d),
  #     np.random.randn(h.M_val, h.d),
  #     np.random.randn(h.M_val, h.d)
  # ]
  # data = utils.create_loaders([raw_data[0], raw_data[1], raw_data[1]],
  #                             h.batch_size)
  data = load_dataset(h)

  #%%
  # !pip install pytorch_memlab
  from pytorch_memlab import MemReporter
  # from pytorch_memlab import LineProfiler
  # import models
  #
  # with LineProfiler(models.CDFNet.likelihood, models.CDFNet.phidots,
  #                   fit.eval_nll) as prof:
  outs = fit.fit_neural_copula(h,
                               data,
                               val_every=100,
                               checkpoint_every=20,
                               eval_validation=False,
                               eval_test=True,
                               use_tb=True,
                               tb_log_dir=TB_DIR,
                               save_checkpoints=True)

  #%%
  plt.plot(outs['nlls'])
  plt.show()
  #prof.display()

  #reporter = MemReporter()
  #reporter.report()


def load_dataset(h):
  if h.dataset == 'gas':
    dataset = GAS(DATA_DIR + '/gas/ethylene_CO.pickle')
  elif h.dataset == 'bsds300':
    dataset = BSDS300(DATA_DIR + '/BSDS300/BSDS300.hdf5')
  elif h.dataset == 'hepmass':
    dataset = HEPMASS(DATA_DIR + '/hepmass')
  elif h.dataset == 'miniboone':
    dataset = MINIBOONE(DATA_DIR + '/miniboone/data.npy')
  elif h.dataset == 'power':
    dataset = POWER(DATA_DIR + '/power/data.npy')
  else:
    raise RuntimeError()

  dataset_train = t.utils.data.TensorDataset(t.tensor(dataset.trn.x).float())
  data_loader_train = t.utils.data.DataLoader(dataset_train,
                                              batch_size=h.batch_size,
                                              shuffle=True)

  dataset_valid = t.utils.data.TensorDataset(t.tensor(dataset.val.x).float())
  data_loader_valid = t.utils.data.DataLoader(dataset_valid,
                                              batch_size=h.batch_size,
                                              shuffle=False)

  dataset_test = t.utils.data.TensorDataset(t.tensor(dataset.tst.x).float())
  data_loader_test = t.utils.data.DataLoader(dataset_test,
                                             batch_size=h.batch_size,
                                             shuffle=False)

  h.d = dataset.n_dims
  h.M = len(dataset_train)
  h.M_val = len(dataset_valid)

  return data_loader_train, data_loader_valid, data_loader_test


if __name__ == '__main__':
  fit_UCI()
