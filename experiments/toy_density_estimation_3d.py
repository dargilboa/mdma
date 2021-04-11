import matplotlib.pyplot as plt
import numpy as np
import torch as t
from experiments.BNAF.data.generate2d import sample2d

# generate data
rng = np.random.RandomState()
dataset_size=200
scale = 4.
centers = [(1, 0, 1), (-1, 0, 1), (0, 1, -1), (0, -1, -1), (1. / np.sqrt(2), 1. / np.sqrt(2), 0),
           (1. / np.sqrt(2), -1. / np.sqrt(2), 0), (-1. / np.sqrt(2),
                                                 1. / np.sqrt(2), 0), (-1. / np.sqrt(2), -1. / np.sqrt(2)), 0]
centers = scale * np.array(centers)

dataset = []
for i in range(dataset_size):
  point = rng.randn(3) * 0.5
  idx = rng.randint(8)
  center = centers[idx]
  point += center
  dataset.append(point)
dataset = np.array(dataset, dtype='float32')
dataset /= 1.414



# plot fitted density
grid_res = 10
xlim = [-3, 3]
ylim = [-3, 3]
zlim = [-3, 3]
batch_size = 200

x_coords = np.linspace(xlim[0], xlim[1], grid_res)
y_coords = np.linspace(ylim[0], ylim[1], grid_res)
z_coords = np.linspace(zlim[0], zlim[1], grid_res)
iX, iY, iZ = np.meshgrid(x_coords, y_coords, z_coords)

flat_grid_on_R = t.tensor([g.flatten()
                           for g in [iX, iY, iZ]]).transpose(0, 1).float()
model_log_density = []
for grid_part in flat_grid_on_R.split(batch_size):
  #model_log_density += [model.log_density(grid_part).cpu().detach().numpy()]
  model_log_density += [grid_part.sum(1).cpu().detach().numpy()]
model_log_density = np.concatenate(model_log_density).reshape(
    (grid_res, grid_res, grid_res))
plt.
