import torch as t
import numpy as np
from torch.distributions.normal import Normal
#from copulae import GumbelCopula
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import matplotlib.pyplot as plt
import time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Timer(object):
  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    if self.name:
      print('[%s]' % self.name, )
    print('Elapsed: %s' % (time.time() - self.tstart))


class linear_interp():
  # linear interpolator
  def __init__(self, x, y):
    # we assume y(x) is monotonic, x_i \neq x_j
    self.x = x  # t.sort(t.squeeze(x))[0]
    self.y = y  # t.sort(t.squeeze(y))[0]
    self.num_knots = len(x)
    self.dydx = t.div(self.y[1:] - self.y[:-1], self.x[1:] - self.x[:-1])

  def __call__(self, points):
    inds = t.searchsorted(self.x, points) - 1
    lb_inds = t.where(inds == -1)
    ub_inds = t.where(inds == self.num_knots - 1)

    # extrapolating using the values at the edges
    inds = t.clamp(inds, 0, self.num_knots - 1)
    diff_inds = t.clamp(inds, 0, self.num_knots - 2)

    interps = t.take(self.y, inds) \
           + t.take(self.dydx, diff_inds) * (points - t.take(self.x, inds))

    # adding in values at the edges
    interps[lb_inds] = self.y[0]
    interps[ub_inds] = self.y[-1]
    return interps


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def invsigmoid(x, eps=1e-15):
  return t.log(x + eps) - t.log(1 - x + eps)


@t.jit.script
def sigmoiddot(x):
  return t.sigmoid(x) * (1 - t.sigmoid(x))


@t.jit.script
def tanhdot(x):
  return 4 * t.exp(-2 * t.abs(x)) / (t.exp(-2 * t.abs(x)) + 1)**2


def ddsigmoid(x):
  return t.sigmoid(x) * (1 - t.sigmoid(x))**2 - t.sigmoid(x)**2 * (
      1 - t.sigmoid(x))


def dddsigmoid(x):
  return sigmoiddot(x) * (
      (1 - t.sigmoid(x))**2 + t.sigmoid(x)**2 - 4 * sigmoiddot(x))


def invsigmoiddot(x):
  return 1 / (x * (1 - x))


def bisect(f, x, lb, ub, n_iter=35):
  # inverts a scalar function f by bisection at x points
  xl = t.ones_like(x) * lb
  xh = t.ones_like(x) * ub
  x_temp = (xl + xh) / 2

  for _ in range(n_iter):
    fmx = f(x_temp) - x
    xl = t.where(fmx < 0, x_temp, xl)
    xh = t.where(fmx < 0, xh, x_temp)
    x_temp = (xl + xh) / 2

  return x_temp


def invert(f, r, lb=0, ub=1, n_bisect_iter=35):
  # return f^{-1}(r), assuming lb <= f^{-1}(r) <= ub
  return bisect(f, r, lb, ub, n_iter=n_bisect_iter)


def stabilize(u, stab_const=1e-5):
  return t.clamp(u, stab_const, 1 - stab_const)


def correlation_matrix(d, rho):
  return np.eye(d) * (1 - rho) + np.ones((d, d)) * rho


def random_correlation_matrix(d):
  S = np.random.randn(d, d)
  P = np.dot(S, S.transpose())
  P = P / np.sqrt(np.diag(P))[:, None] / np.sqrt(np.diag(P))[None, :]
  return P


def normalize(M):
  return (M - np.mean(M, axis=0)) / np.std(M, axis=0)


def load_dataset(args):
  if args.dataset == 'gas':
    dataset = GAS('data/gas/ethylene_CO.pickle')
  elif args.dataset == 'bsds300':
    dataset = BSDS300('data/BSDS300/BSDS300.hdf5')
  elif args.dataset == 'hepmass':
    dataset = HEPMASS('data/hepmass')
  elif args.dataset == 'miniboone':
    dataset = MINIBOONE('data/miniboone/data.npy')
  elif args.dataset == 'power':
    dataset = POWER('data/power/data.npy')
  else:
    raise RuntimeError()

  dataset_train = t.utils.data.TensorDataset(
      t.from_numpy(dataset.trn.x).float().to(args.device))
  data_loader_train = t.utils.data.DataLoader(dataset_train,
                                              batch_size=args.batch_dim,
                                              shuffle=True)

  dataset_valid = t.utils.data.TensorDataset(
      t.from_numpy(dataset.val.x).float().to(args.device))
  data_loader_valid = t.utils.data.DataLoader(dataset_valid,
                                              batch_size=args.batch_dim,
                                              shuffle=False)

  dataset_test = t.utils.data.TensorDataset(
      t.from_numpy(dataset.tst.x).float().to(args.device))
  data_loader_test = t.utils.data.DataLoader(dataset_test,
                                             batch_size=args.batch_dim,
                                             shuffle=False)

  args.d = dataset.n_dims

  return data_loader_train, data_loader_valid, data_loader_test


def create_loaders(data, batch_size):
  # create dataloaders from list of data arrays or tensors
  train_data, val_data, test_data = data
  if type(train_data) == np.ndarray:
    train_data = t.Tensor(np.expand_dims(train_data, 1))
  train_dataset = TensorDataset(train_data)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  if val_data is None:
    val_loader = None
  else:
    if type(val_data) == np.ndarray:
      val_data = t.Tensor(np.expand_dims(val_data, 1))
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

  if test_data is None:
    test_loader = None
  else:
    if type(test_data) == np.ndarray:
      test_data = t.Tensor(np.expand_dims(test_data, 1))
    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

  return train_loader, val_loader, test_loader


def shorten(s):
  return ''.join([w[0] for w in s.split('_')])


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def save_file(file_name):
  plt.savefig(file_name)
  os.system('pdfcrop "' + file_name + '" "' + file_name + '"')


class EarlyStopping(object):
  def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
    self.mode = mode
    self.min_delta = min_delta
    self.patience = patience
    self.best = None
    self.num_bad_epochs = 0
    self.is_better = None
    self._init_is_better(mode, min_delta, percentage)

    if patience == 0:
      self.is_better = lambda a, b: True
      self.step = lambda a: False

  def step(self, metrics):
    if self.best is None:
      self.best = metrics
      return False

    if np.isnan(metrics):
      return True

    if self.is_better(metrics, self.best):
      self.num_bad_epochs = 0
      self.best = metrics
    else:
      self.num_bad_epochs += 1

    if self.num_bad_epochs >= self.patience:
      return True

    return False

  def _init_is_better(self, mode, min_delta, percentage):
    if mode not in {'min', 'max'}:
      raise ValueError('mode ' + mode + ' is unknown!')
    if not percentage:
      if mode == 'min':
        self.is_better = lambda a, best: a < best - min_delta
      if mode == 'max':
        self.is_better = lambda a, best: a > best + min_delta
    else:
      if mode == 'min':
        self.is_better = lambda a, best: a < best - (best * min_delta / 100)
      if mode == 'max':
        self.is_better = lambda a, best: a > best + (best * min_delta / 100)
