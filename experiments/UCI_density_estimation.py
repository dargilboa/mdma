#%%
import fit
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import utils

from experiments.BNAF.data.gas import GAS
from experiments.BNAF.data.bsds300 import BSDS300
from experiments.BNAF.data.hepmass import HEPMASS
from experiments.BNAF.data.miniboone import MINIBOONE
from experiments.BNAF.data.power import POWER

np.random.seed(0)
t.manual_seed(0)


def load_dataset(h):
  data_dir = 'BNAF/data'
  if h.dataset == 'gas':
    dataset = GAS(data_dir + '/gas/ethylene_CO.pickle')
  elif h.dataset == 'bsds300':
    dataset = BSDS300(data_dir + '/BSDS300/BSDS300.hdf5')
  elif h.dataset == 'hepmass':
    dataset = HEPMASS(data_dir + '/hepmass')
  elif h.dataset == 'miniboone':
    dataset = MINIBOONE(data_dir + '/miniboone/data.npy')
  elif h.dataset == 'power':
    dataset = POWER(data_dir + '/power/data.npy')
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


#%%
h = fit.get_default_h()
h.dataset = 'power'
h.batch_size = 2000
h.n_epochs = 4
h.lr = 1e-2

data = load_dataset(h)

#%%
#!pip install pytorch_memlab
#from pytorch_memlab import MemReporter
#from pytorch_memlab import LineProfiler

#with LineProfiler(fit.fit_neural_copula, fit.eval_val_nll) as prof:
outs = fit.fit_neural_copula(
    h,
    data,
    val_every=100,
    checkpoint_every=4000,
)

#%%
plt.plot(outs['nlls'])
plt.show()
#prof.display()

#reporter = MemReporter()
#reporter.report()
