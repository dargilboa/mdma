#%%
import fit
import torch as t
import matplotlib.pyplot as plt

from experiments.BNAF.data.gas import GAS
from experiments.BNAF.data.bsds300 import BSDS300
from experiments.BNAF.data.hepmass import HEPMASS
from experiments.BNAF.data.miniboone import MINIBOONE
from experiments.BNAF.data.power import POWER


def load_dataset(h):
  data_dir = 'data'
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
h.batch_size = 1000
h.n_epochs = 5
h.lr_m = 5e-4
h.lr = 5e-3
h.fit_marginals_first = True
h.n_marginal_iters = 2000

train_data, val_data, _ = load_dataset(h)
data = [train_data, val_data]

#%%
#!pip install pytorch_memlab
#from pytorch_memlab import MemReporter
#from pytorch_memlab import LineProfiler

#with LineProfiler(fit.fit_neural_copula) as prof:
outs = fit.fit_neural_copula(
    h,
    data,
    val_every=1000,
    checkpoint_every=500,
)

#%%
plt.plot(outs['nlls'])
plt.show()
#prof.display()

#reporter = MemReporter()
#reporter.report()
