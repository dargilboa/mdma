#%%
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import cdfnet.fit as fit
import cdfnet.utils as utils
import cdfnet.plots as plots
import os
from mpl_toolkits import mplot3d
from experiments.BNAF.data.generate2d import sample2d

save_plots = True
dataset_name = 'spirals'
M = 20000
os.chdir(utils.ROOT_DIR)
save_dir = "../Copula Estimation with Neural Networks/figs/"

#%% generate data
rng = np.random.RandomState()
if dataset_name == 'gaussians':
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
elif dataset_name == 'spirals':
  n = np.sqrt(np.random.rand(M // 2, 1)) * 540 * (2 * np.pi) / 360
  d1x = -np.cos(n) * n + np.random.rand(M // 2, 1) * 0.5
  d1y = np.sin(n) * n + np.random.rand(M // 2, 1) * 0.5
  z = np.expand_dims(np.linalg.norm(np.hstack((d1x, d1y)), axis=1), -1) / 3
  dataset = np.vstack((np.hstack((d1x, d1y, z)), np.hstack(
      (-d1x, -d1y, -z)))) / 3
  dataset += np.random.randn(*dataset.shape) * 0.1

#%% plot data
plt.figure()
n_pts_to_plot = 2000
ax = plt.axes(projection='3d')
np.random.shuffle(dataset)
lenX = len(dataset)
colors = np.array([[0., 0., 0.]] * n_pts_to_plot)
colors[:, 2] = 10 - (np.max(dataset[:n_pts_to_plot, 2]) -
                     dataset[:n_pts_to_plot, 2]) + 2
colors[:, 2] = colors[:, 2] / np.max(colors[:, 2])
ax.scatter3D(dataset[:n_pts_to_plot, 0],
             dataset[:n_pts_to_plot, 1],
             dataset[:n_pts_to_plot, 2],
             c=colors)
if dataset_name == 'gaussians':
  ax.view_init(elev=30., azim=55)
elif dataset_name == 'spirals':
  ax.view_init(elev=15., azim=10)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
if save_plots:
  utils.save_file(save_dir + '_'.join(['3d', dataset_name, 'data']) + '.pdf')
plt.show()

#%% create model and fit
batch_size = 100
h = fit.get_default_h()
h.d = 3
h.M = M
h.use_HT = True
h.n = 1000
h.model_to_load = ''
h.save_checkpoints = False
loaders = utils.create_loaders([dataset, 0, 0], batch_size)
h.eval_validation = False
h.eval_test = False
model = fit.fit_neural_copula(h, loaders)

#%% loading model
# checkpoint = t.load('saved_models/3d_gaussians_model.pt')
# model.load_state_dict(checkpoint['model'])

# #%% plot combined
# plt.figure()
# ax = plt.axes(projection='3d')
# from matplotlib import cm
# ub = 4
# lb = -4
# # full
# x_coords = np.linspace(lb, ub, grid_res)
# y_coords = np.linspace(lb, ub, grid_res)
# z_coords = np.linspace(lb, ub, grid_res)
# mg = np.meshgrid(x_coords, y_coords, z_coords)
# iX, iY, iZ = mg
# model_log_density = eval_log_density_on_grid(
#     model,
#     mg,
# )
#
# #lenX = len(iX.flatten())
# #colors = np.array([[0., 0., 0.]] * lenX)
# #colors[:, 2] = np.abs(iX.flatten()) / np.max(np.abs(iX))
# ax.scatter(iX, iY, iZ, s=10000 * np.exp(model_log_density), alpha=0.5, c='r')
#
# # marginals
# x_coords = np.linspace(lb, ub, grid_res)
# y_coords = np.linspace(lb, ub, grid_res)
# mg = np.meshgrid(x_coords, y_coords)
# iX, iY = mg
# Z = lb * np.ones_like(iX)
# model_log_density = eval_log_density_on_grid(model, mg, inds=[0, 1])
# fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
# ax.plot_surface(iX, iY, Z, facecolors=fc)
# model_log_density = eval_log_density_on_grid(model, mg, inds=[1, 2])
# fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
# ax.plot_surface(Z, iX, iY, facecolors=fc)
# model_log_density = eval_log_density_on_grid(model, mg, inds=[0, 2])
# fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
# ax.plot_surface(iX, Z, iY, facecolors=fc)
#
# ax.set_xlabel('x_1')
# ax.set_ylabel('x_2')
# ax.set_zlabel('x_3')
# ax.view_init(elev=30., azim=45)
# plt.show()

#%% plot separate - 3d density
plt.figure()
ax = plt.axes(projection='3d')
ub = 4
lb = -4
grid_res = 20
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
z_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords, z_coords)
iX, iY, iZ = mg
model_log_density = plots.eval_log_density_on_grid(model,
                                                   mg,
                                                   grid_res=grid_res)

lenX = len(iX.flatten())
colors = np.array([[0., 0., 0.]] * lenX)

colors[:, 2] = 10 - (np.max(iZ) - iZ.flatten()) + 2
colors[:, 2] = colors[:, 2] / np.max(colors[:, 2])
ax.scatter(iX, iY, iZ, s=1000 * np.exp(model_log_density), alpha=0.5, c=colors)
if dataset_name == 'gaussians':
  ax.view_init(elev=30., azim=55)
elif dataset_name == 'spirals':
  ax.view_init(elev=15., azim=10)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
if save_plots:
  utils.save_file(save_dir + '_'.join(['3d', dataset_name, '3d_density']) +
                  '.pdf')
plt.show()

#%% 2d marginals
ub = 4
lb = -4
grid_res = 60
# marginals
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords)
iX, iY = mg
Z = lb * np.ones_like(iX)
for vars in [[0, 1], [1, 2], [0, 2]]:
  model_log_density = plots.eval_log_density_on_grid(model,
                                                     mg,
                                                     inds=vars,
                                                     grid_res=grid_res)
  plt.figure()
  plt.imshow(model_log_density, extent=[lb, ub, lb, ub])
  plt.xlabel('$x_' + str(vars[0] + 1) + '$')
  plt.ylabel('$x_' + str(vars[1] + 1) + '$')
  if save_plots:
    utils.save_file(save_dir + '_'.join(
        ['3d', dataset_name, '2d_marg',
         str(vars[0] + 1),
         str(vars[1] + 1)]) + '.pdf')
  plt.show()

#%% 2d conditionals
ub = 4
lb = -4
grid_res = 20
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords)
iX, iY = mg
Z = lb * np.ones_like(iX)
inds = [0, 1]
cond_vals = [-2.5, 2.5]
for cond_val in cond_vals:
  model_cond_density = plots.eval_cond_density_on_grid(model,
                                                       mg,
                                                       cond_val,
                                                       inds=inds,
                                                       cond_inds=[2],
                                                       grid_res=grid_res)

  plt.figure()
  plt.imshow(model_cond_density, extent=[lb, ub, lb, ub])
  plt.xlabel('$x_' + str(vars[0] + 1) + '$')
  plt.ylabel('$x_' + str(vars[1] + 1) + '$')
  if save_plots:
    utils.save_file(save_dir +
                    '_'.join(['3d', dataset_name, '2d_cond',
                              str(cond_val)]) + '.pdf')
  plt.show()

#%% 1d marginals
plt.figure()
xs = np.linspace(-4, 4, 100)
model_log_density = np.exp(
    model.log_density(t.tensor(xs).float(), inds=[0]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
model_log_density = np.exp(
    model.log_density(t.tensor(xs).float(), inds=[1]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
model_log_density = np.exp(
    model.log_density(t.tensor(xs).float(), inds=[2]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
plt.legend(['$X_1$', '$X_2$', '$X_3$'])
if save_plots:
  utils.save_file(save_dir + '_'.join(['3d', dataset_name, '1d_marg']) +
                  '.pdf')
plt.show()
