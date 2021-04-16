import fit
import torch as t
import numpy as np

from experiments.BNAF.data.gas import GAS
from experiments.BNAF.data.bsds300 import BSDS300
from experiments.BNAF.data.hepmass import HEPMASS
from experiments.BNAF.data.miniboone import MINIBOONE
from experiments.BNAF.data.power import POWER

np.random.seed(0)
t.manual_seed(0)


def fit_UCI():
  # dataset='power',
  # batch_size=2000,
  # n_epochs=8,
  # n=500,
  # m=10,
  # L=6,
  # lr=5e-2,
  # lambda_l2=0,
  # patience=200,
  # checkpoint_every=200):
  checkpoint_every = 200
  h = fit.get_default_h()
  # h.dataset = dataset
  # h.batch_size = batch_size
  # h.n_epochs = n_epochs
  # h.n = n
  # h.m = m
  # h.L = L
  # h.lr = lr
  # h.lambda_l2 = lambda_l2
  # h.patience = patience
  data = load_dataset(h)

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
                               checkpoint_every=checkpoint_every,
                               eval_validation=False,
                               eval_test=True,
                               save_checkpoints=True)

  #plt.plot(outs['nlls'])
  #plt.show()
  #prof.display()

  #reporter = MemReporter()
  #reporter.report()


def load_dataset(h):
  if h.dataset == 'gas':
    dataset = GAS(h.data_dir + '/gas/ethylene_CO.pickle')
  elif h.dataset == 'bsds300':
    dataset = BSDS300(h.data_dir + '/BSDS300/BSDS300.hdf5')
  elif h.dataset == 'hepmass':
    dataset = HEPMASS(h.data_dir + '/hepmass')
  elif h.dataset == 'miniboone':
    dataset = MINIBOONE(h.data_dir + '/miniboone/data.npy')
  elif h.dataset == 'power':
    dataset = POWER(h.data_dir + '/power/data.npy')
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
