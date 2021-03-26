'''
Density estimation on the UCI power dataset

https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
'''

import fit
import csv
import numpy as np
import torch as t

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.DoubleTensor')

#%% load data

with open('experiments/UCI_power/household_power_consumption.txt',
          newline='') as csvfile:  # assumes data file is in this folder
  raw_data = list(csv.reader(csvfile))

data_array = np.zeros((len(raw_data), 7))
for nrow, row in enumerate(raw_data[1:]):
  if '?' not in row[0].split(';')[2:]:
    data_array[nrow] = np.array([float(n) for n in row[0].split(';')[2:]])

val_size = 2000
train_data, val_data = data_array[:-val_size], data_array[-val_size:]
data = [train_data, val_data]

#%% fit
M = len(train_data)
d = 7
h = {
    'M': M,
    'M_val': val_size,
    'd': d,
    'n_epochs': 1,
    'batch_size': 200,
    'n': 100,
    'lambda_l2': 1e-5,
    'lr': 5e-3,
    'fit_marginals': True,
}

np.random.seed(1)
t.manual_seed(1)

outs = fit.fit_neural_copula(data, h, val_every=100)
