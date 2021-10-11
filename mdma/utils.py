# Copyright © 2021 Dar Gilboa, Ari Pakman and Thibault Vatter
# This file is part of the mdma library and licensed under the terms of the MIT license.
# For a copy, see the LICENSE file in the root directory.

import torch as t
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import matplotlib.pyplot as plt
import time
from typing import Callable, Union, List


def invert(f: Callable,
           x: t.Tensor,
           lb: float,
           ub: float,
           n_bisect_iter: int = 35) -> t.Tensor:
  """ Invert a univariate function f at points r using the bisection method.

  Args:
    f: Function to invert.
    x: Tensor of points at which to invert f.
    lb: Lower bound for f.
    ub: Upper bound for f.
    n_bisect_iter: Number of iterations of bisection method.

  Returns:
    A tensor of points f^{-1}(x).
  """

  xl = t.ones_like(x) * lb
  xh = t.ones_like(x) * ub
  x_temp = (xl + xh) / 2

  for _ in range(n_bisect_iter):
    fmx = f(x_temp).reshape(x.shape) - x
    xl = t.where(fmx < 0, x_temp, xl)
    xh = t.where(fmx < 0, xh, x_temp)
    x_temp = (xl + xh) / 2
  return x_temp


def create_loaders(data: Union[List[t.Tensor], List[np.ndarray]],
                   batch_size: int) -> List[DataLoader]:
  """ Create data loaders from data arrays or tensors.

  Args:
    data: List of training, validation and test data.
    batch_size: The desired batch size.

  Returns:
    List of data loaders.
  """
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

  return [train_loader, val_loader, test_loader]


class Timer(object):
  """
    A simple timer class.
  """
  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    if self.name:
      print('[%s]' % self.name, )
    print('Elapsed: %s' % (time.time() - self.tstart))


@t.jit.script
def sigmoiddot(x: t.Tensor):
  return t.sigmoid(x) * (1 - t.sigmoid(x))


@t.jit.script
def tanhdot(x: t.Tensor):
  return 4 * t.exp(-2 * t.abs(x)) / (t.exp(-2 * t.abs(x)) + 1)**2


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


def eval_log_density_on_grid(model,
                             meshgrid,
                             inds=...,
                             grid_res=20,
                             batch_size=200):
  flat_grid_on_R = np.array([g.flatten() for g in meshgrid]).transpose()
  if inds == ...:
    final_shape = (grid_res, grid_res, grid_res)
  else:
    final_shape = (grid_res, grid_res)
  model_log_density = []
  for grid_part in np.split(flat_grid_on_R, len(flat_grid_on_R) // batch_size):
    model_log_density += [
        model.log_density(t.tensor(grid_part).float(),
                          inds=inds).cpu().detach().numpy()
    ]
  model_log_density = np.concatenate(model_log_density).reshape(final_shape)
  return model_log_density


def eval_cond_density_on_grid(
    model,
    meshgrid,
    cond_val,
    inds=...,
    grid_res=20,
    batch_size=200,
    cond_inds=...,
):
  flat_grid_on_R = np.array([g.flatten() for g in meshgrid]).transpose()
  if inds == ...:
    final_shape = (grid_res, grid_res, grid_res)
  else:
    final_shape = (grid_res, grid_res)
  model_cond_density = []
  split_grid = np.split(flat_grid_on_R, len(flat_grid_on_R) // batch_size)
  for grid_part in split_grid:
    cond_x = cond_val * t.ones((batch_size, 1)).float()
    model_cond_density += [
        model.cond_density(t.tensor(grid_part).float(),
                           inds=inds,
                           cond_X=cond_x,
                           cond_inds=cond_inds).cpu().detach().numpy()
    ]
  model_cond_density = np.concatenate(model_cond_density).reshape(final_shape)
  return model_cond_density