# Copyright © 2021 Dar Gilboa, Ari Pakman and Thibault Vatter
# This file is part of the mdma library and licensed under the terms of the MIT license.
# For a copy, see the LICENSE file in the root directory.

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import mdma.fit as fit
import mdma.utils as utils

save_plots = False
h = fit.get_default_h()
M = 200000
save_dir = "./"

# generate data
rng = np.random.RandomState()
if h.dataset == 'gaussians':
  scale = 4.
  centers = [[1, 0, -1], [-1, 0, .14], [0, 1, -.43], [0, -1, .71],
             [1. / np.sqrt(2), 1. / np.sqrt(2), -.71],
             [1. / np.sqrt(2), -1. / np.sqrt(2), 1],
             [-1. / np.sqrt(2), 1. / np.sqrt(2), -.14],
             [-1. / np.sqrt(2), -1. / np.sqrt(2), .43]]
  centers = scale * np.array(centers)

  dataset = []
  for i in range(M):
    point = rng.randn(3) * 0.5
    idx = rng.randint(8)
    center = centers[idx]
    point += center
    dataset.append(point)
  dataset = np.array(dataset, dtype='float32')
  dataset /= 1.414
  zlim = [-4, 4]
elif h.dataset == 'spirals':
  n = np.sqrt(np.random.rand(M // 2, 1)) * 540 * (2 * np.pi) / 360
  d1x = -np.cos(n) * n + np.random.rand(M // 2, 1) * 0.5
  d1y = np.sin(n) * n + np.random.rand(M // 2, 1) * 0.5
  z = np.expand_dims(np.linalg.norm(np.hstack((d1x, d1y)), axis=1), -1) / 3
  dataset = np.vstack((np.hstack((d1x, d1y, z)), np.hstack(
      (-d1x, -d1y, -z)))) / 3
  dataset += np.random.randn(*dataset.shape) * 0.1
  zlim = [-2, 2]
  np.random.shuffle(dataset)
elif h.dataset == 'checkerboard':
  x = np.random.rand(M)
  dataset = np.zeros((M, 3))
  one_centers = [.125, .625]
  two_centers = [.375, .875]
  one_inds = np.where((x < .25) | ((x > .5) & (x < .75)))[0]
  two_inds = np.where((x > .75) | ((x > .25) & (x < .5)))[0]
  dataset[:, 0] = x
  for ind in [1, 2]:
    dataset[one_inds, ind] = np.take(one_centers,
                                     np.random.choice([0, 1], len(one_inds)))
    dataset[two_inds, ind] = np.take(two_centers,
                                     np.random.choice([0, 1], len(two_inds)))
    dataset[:, ind] += np.random.rand(M) / 4 - .125
  dataset = dataset * 8 - 4
  zlim = [-4, 4]

# Plot data 3d scatter
plt.figure()
n_pts_to_plot = 2000
ax = plt.axes(projection='3d')
lenX = len(dataset)
ax.scatter3D(dataset[:n_pts_to_plot, 0],
             dataset[:n_pts_to_plot, 1],
             dataset[:n_pts_to_plot, 2],
             s=1)
if h.dataset == 'gaussians':
  ax.view_init(elev=30., azim=55)
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=90)
elif h.dataset == 'spirals':
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=90)
  ax.view_init(elev=20., azim=10)
else:
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=270)

ax.set_xlabel('$x_1$', labelpad=-12)
ax.set_ylabel('$x_2$', labelpad=-12)
ax.set_title('Training data')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.locator_params(nbins=3)
if save_plots:
  plt.savefig(save_dir + '_'.join(['3d', h.dataset, 'data']) + '.pdf')
plt.show()

# Plot 2d histogram of data
ub = 4
lb = -4
grid_res = 60
lims = [[lb, ub], [lb, ub], zlim]
ax = plt.axes(projection=None)

for vars in [[0, 1], [1, 2], [0, 2]]:
  hist_data = np.histogram2d(dataset[:, vars[0]],
                             dataset[:, vars[1]],
                             grid_res,
                             range=[lims[vars[0]], lims[vars[1]]])
  lim0 = lims[vars[0]]
  lim1 = lims[vars[1]]
  plt.imshow(hist_data[0].transpose(),
             extent=[lim0[0], lim0[1], lim1[0], lim1[1]],
             aspect=(lim0[1] - lim0[0]) / (lim1[1] - lim1[0]))
  plt.xlabel('$x_' + str(vars[0] + 1) + '$')
  plt.ylabel('$x_' + str(vars[1] + 1) + '$')
  plt.xticks([])
  plt.yticks([])
  plt.title('Training data')
  if save_plots:
    plt.savefig(save_dir + '_'.join(
        ['3d', h.dataset, 'data_2d',
         str(vars[0] + 1),
         str(vars[1] + 1)]) + '.pdf')
  plt.show()

# Create model and fit
batch_size = 1000
h = fit.get_default_h()
h.batch_size = batch_size
h.d = 3
h.M = M
h.use_HT = True
if h.dataset == 'checkerboard':
  h.m = 250
else:
  h.m = 1000
h.model_to_load = ''
h.n_epochs = 10
h.patience = 150
h.save_checkpoints = False
loaders = utils.create_loaders([dataset, None, None], batch_size)
h.eval_validation = False
h.eval_test = False
model = fit.fit_mdma(h, loaders)

# Plot samples from model
plt.figure()
n_pts_to_plot = 2000
ax = plt.axes(projection='3d')
samples = model.sample(n_pts_to_plot)
ax.scatter3D(samples[:n_pts_to_plot, 0],
             samples[:n_pts_to_plot, 1],
             samples[:n_pts_to_plot, 2],
             s=1)
if h.dataset == 'gaussians':
  ax.view_init(elev=30., azim=55)
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=90)
elif h.dataset == 'spirals':
  ax.view_init(elev=20., azim=10)
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=90)
else:
  ax.zaxis.set_rotate_label(False)
  ax.set_zlabel('$x_3$', labelpad=-12, rotation=270)
ax.set_title('Samples from model')
ax.set_xlabel('$x_1$', labelpad=-12)
ax.set_ylabel('$x_2$', labelpad=-12)
ax.zaxis.set_rotate_label(False)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.locator_params(nbins=3)
if save_plots:
  plt.savefig(save_dir + '_'.join(['3d', h.dataset, 'model_samples_3d']) +
              '.pdf')
plt.show()

# 2d marginals
lims = [[lb, ub], [lb, ub], zlim]
ax = plt.axes(projection=None)
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
z_coords = np.linspace(zlim[0], zlim[1], grid_res)
coords = [x_coords, y_coords, z_coords]
for vars in [[0, 1], [1, 2], [0, 2]]:
  lim0 = lims[vars[0]]
  lim1 = lims[vars[1]]
  mg = np.meshgrid(coords[vars[0]], coords[vars[1]])
  model_log_density = utils.eval_log_density_on_grid(model,
                                                     mg,
                                                     inds=vars,
                                                     grid_res=grid_res)
  plt.figure()
  plt.imshow(np.exp(model_log_density),
             extent=[lim0[0], lim0[1], lim1[0], lim1[1]],
             aspect=(lim0[1] - lim0[0]) / (lim1[1] - lim1[0]))
  plt.xlabel('$x_' + str(vars[0] + 1) + '$')
  plt.ylabel('$x_' + str(vars[1] + 1) + '$')
  plt.xticks([])
  plt.yticks([])
  plt.title(f'$f(x_{vars[0] + 1}, x_{vars[1] + 1})$', fontsize=22)
  if save_plots:
    plt.savefig(save_dir + '_'.join(
        ['3d', h.dataset, '2d_marg',
         str(vars[0] + 1),
         str(vars[1] + 1)]) + '.pdf')
  plt.show()

# 2d conditionals
ub = 4
lb = -4
grid_res = 60
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords)
inds = [0, 1]
cond_vals = [-.5, 0, .5]
for cond_val in cond_vals:
  model_cond_density = utils.eval_cond_density_on_grid(model,
                                                       mg,
                                                       cond_val,
                                                       inds=inds,
                                                       cond_inds=[2],
                                                       grid_res=grid_res)

  plt.figure(figsize=(4, 4))
  plt.imshow(model_cond_density, extent=[lb, ub, lb, ub])
  plt.xlabel('$x_1$')
  plt.ylabel('$x_2$')
  plt.xticks([])
  plt.yticks([])
  plt.title('$f(x_1, x_2 | x_3 = ' + str(cond_val) + ')$', fontsize=22)
  if save_plots:
    plt.savefig(save_dir + '_'.join(
        ['3d', h.dataset, '2d_cond', str(cond_val)]) + '.pdf')
  plt.show()

# 1d marginals
plt.figure(figsize=(4, 4))
if h.dataset == 'checkerboard':
  rng = 5
else:
  rng = 4
xs = np.linspace(-rng, rng, 100)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(3):
  model_log_density = np.exp(
      model.log_density(t.tensor(xs).float(), inds=[i]).cpu().detach().numpy())
  plt.plot(xs, model_log_density, color=colors[i], label=f'$f(x_{i+1})$', lw=2)
  hist_data = np.histogram(dataset[:, i], 100, range=(-rng, rng), density=1)
  plt.plot((hist_data[1][1] - hist_data[1][0]) / 2 + hist_data[1][:-1],
           hist_data[0],
           'x',
           color=colors[i])
plt.plot(10, 0, 'x', color='k', label='Training \n data')
plt.xlim([-rng, rng])
if h.dataset == 'gaussians':
  plt.legend(handletextpad=0, handlelength=1, loc=3)
else:
  plt.legend(handletextpad=0, handlelength=1)
if save_plots:
  plt.savefig(save_dir + '_'.join(['3d', h.dataset, '1d_marg']) + '.pdf')
plt.show()
