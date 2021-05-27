import numpy as np


class ci_data:
  class Data:
    def __init__(self, data):

      self.x = data.astype(np.float32)
      self.N = self.x.shape[0]

  def __init__(self, fn, n=20000, d=3):

    trn = self.sample_ci_data(fn, n, d)
    val = self.sample_ci_data(fn, n, d)
    tst = self.sample_ci_data(fn, n, d)

    self.trn = self.Data(trn)
    self.val = self.Data(val)
    self.tst = self.Data(tst)

    self.n_dims = self.trn.x.shape[1]

  def sample_ci_data(self, fn, n=1000, d=3):
    data_r = fn(n, d)
    np.concatenate([np.array(xx) for xx in data_r], axis=1)
    return np.concatenate([np.array(xx) for xx in data_r], axis=1)
