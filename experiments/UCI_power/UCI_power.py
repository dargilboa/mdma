"""
Density estimation on the UCI power dataset

https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Fields in data:

1.date: Date in format dd/mm/yyyy (removed)
2.time: time in format hh:mm:ss
3.global_active_power: household global minute-averaged active power (in kilowatt)
4.global_reactive_power: household global minute-averaged reactive power (in kilowatt) (removed)
5.voltage: minute-averaged voltage (in volt)
6.global_intensity: household global minute-averaged current intensity (in ampere)
7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.
"""

import fit
import csv
import numpy as np
import torch as t
import urllib.request
import os
import utils
import matplotlib.pyplot as plt
import plots

#%% load data
data_dir = './experiments/UCI_power/data'
data_file_path = data_dir + '/household_power_consumption.txt'

if not os.path.exists(data_file_path):
  print('Downloading data')
  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
  urllib.request.urlretrieve(url, data_dir + '.zip')
  os.system(f'unzip {data_dir}.zip -d {data_dir} -o')
with open(data_file_path,
          newline='') as csvfile:  # assumes data file is in this folder
  raw_data = list(csv.reader(csvfile))

#%% preprocessing
data_list = []
for nrow, row in enumerate(raw_data[1:]):
  if '?' not in row[0]:
    row_list = row[0].split(';')
    # discarding date and global reactive power
    row_list = row_list[1:3] + row_list[4:]

    # convert time to number
    time = row_list[0].split(':')
    row_list[0] = float(time[0]) * 60 + float(time[1])
    data_list += [[float(i) for i in row_list]]

data_array = np.array(data_list)

# add noise
N = len(data_array)
voltage_noise = 0.01 * np.random.rand(N, 1)
# grp_noise = 0.001*rng.rand(N, 1)
gap_noise = 0.001 * np.random.rand(N, 1)
sm_noise = np.random.rand(N, 3)
tot_noise = np.random.rand(N, 1)
noise = np.concatenate([gap_noise, voltage_noise, tot_noise, sm_noise], axis=1)

data_array[:, 1:] += noise

# split test and val
split_ratio = 0.01

test_size = int(split_ratio * data_array.shape[0])
test_data = data_array[-test_size:]
data_array = data_array[0:-test_size]
val_size = int(split_ratio * data_array.shape[0])
val_data = data_array[-val_size:]
train_data = data_array[0:-val_size]

test_data = utils.normalize(test_data)
val_data = utils.normalize(val_data)
train_data = utils.normalize(train_data)

data = [train_data, val_data, test_data]

#%% remove variables
train_data = train_data[:, :2]
val_data = val_data[:, :2]

#%% fit
M, d = train_data.shape

h = {
    'M': M,
    'M_val': val_size,
    'd': d,
    'epochs': 1,
    'batch_size': 512,
    'n': 100,
    'l2': 1e-5,
    'lr': 5e-1,
    'fit_marginals': True,
    'marginal_smoothing_factor': 1,
}

np.random.seed(1)
t.manual_seed(1)

outs = fit.fit_neural_copula([train_data, val_data],
                             h,
                             val_every=50,
                             max_iters=100)
plots.plot_contours_ext(outs,
                        copula_params=None,
                        copula_type='data',
                        marginal_type='data',
                        model_includes_marginals=True,
                        data=val_data[:1000])
#%% debug
plt.scatter(train_data[:3000, 0], train_data[:3000, 1])
plt.show()
