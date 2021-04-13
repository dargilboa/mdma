#%%
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import models
import fit
import utils
from mpl_toolkits import mplot3d
from experiments.BNAF.data.generate2d import sample2d

#%% generate data
rng = np.random.RandomState()
M = 20000
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

#%% plot data
plt.figure()
ax = plt.axes(projection='3d')
lenX = len(dataset)
colors = np.array([[0., 0., 0.]] * lenX)
colors[:, 2] = (np.max(dataset[:, 1]) - dataset[:, 1]) + 2
colors[:, 2] = colors[:, 2] / np.max(colors[:, 2])
ax.scatter3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colors)
ax.view_init(elev=30., azim=45)
plt.show()

#%% create model and fit
batch_size = 100
h = fit.get_default_h()
h.d = 3
h.M = M
loaders = utils.create_loaders([dataset, 0, 0], batch_size)
outs = fit.fit_neural_copula(h, loaders, eval_validation=False)

#%% plot fitted density
grid_res = 20
xlim = [-3.5, 3.5]
ylim = [-3.5, 3.5]
zlim = [-3.5, 3.5]
batch_size = 200


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


#%% plot combined
plt.figure()
ax = plt.axes(projection='3d')
from matplotlib import cm
ub = 4
lb = -4
# full
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
z_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords, z_coords)
iX, iY, iZ = mg
model_log_density = eval_log_density_on_grid(
    outs['model'],
    mg,
)

#lenX = len(iX.flatten())
#colors = np.array([[0., 0., 0.]] * lenX)
#colors[:, 2] = np.abs(iX.flatten()) / np.max(np.abs(iX))
ax.scatter(iX, iY, iZ, s=10000 * np.exp(model_log_density), alpha=0.5, c='r')

# marginals
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords)
iX, iY = mg
Z = lb * np.ones_like(iX)
model_log_density = eval_log_density_on_grid(outs['model'], mg, inds=[0, 1])
fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
ax.plot_surface(iX, iY, Z, facecolors=fc)
model_log_density = eval_log_density_on_grid(outs['model'], mg, inds=[1, 2])
fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
ax.plot_surface(Z, iX, iY, facecolors=fc)
model_log_density = eval_log_density_on_grid(outs['model'], mg, inds=[0, 2])
fc = cm.viridis(np.exp(model_log_density) / np.max(np.exp(model_log_density)))
ax.plot_surface(iX, Z, iY, facecolors=fc)

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('x_3')
ax.view_init(elev=30., azim=45)
plt.show()

#%% plot separate - 3d density
plt.figure()
ax = plt.axes(projection='3d')
from matplotlib import cm
ub = 4
lb = -4
# full
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
z_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords, z_coords)
iX, iY, iZ = mg
model_log_density = eval_log_density_on_grid(
    outs['model'],
    mg,
)

lenX = len(iX.flatten())
colors = np.array([[0., 0., 0.]] * lenX)
colors[:, 2] = (np.max(iY) - iY.flatten()) + 2
colors[:, 2] = colors[:, 2] / np.max(colors[:, 2])
ax.scatter(iX,
           iY,
           iZ,
           s=10000 * np.exp(model_log_density),
           alpha=0.5,
           c=colors)
ax.view_init(elev=30., azim=45)
plt.show()

#%% separate - 2d marginals
ub = 4
lb = -4
grid_res = 60
# marginals
x_coords = np.linspace(lb, ub, grid_res)
y_coords = np.linspace(lb, ub, grid_res)
mg = np.meshgrid(x_coords, y_coords)
iX, iY = mg
Z = lb * np.ones_like(iX)
model_log_density = eval_log_density_on_grid(outs['model'],
                                             mg,
                                             inds=[0, 1],
                                             grid_res=grid_res)
plt.figure()
plt.imshow(model_log_density)
plt.show()
model_log_density = eval_log_density_on_grid(outs['model'],
                                             mg,
                                             inds=[1, 2],
                                             grid_res=grid_res)
plt.figure()
plt.imshow(model_log_density)
plt.show()
model_log_density = eval_log_density_on_grid(outs['model'],
                                             mg,
                                             inds=[0, 2],
                                             grid_res=grid_res)
plt.figure()
plt.imshow(model_log_density)
plt.show()

#%% 1d marginals
plt.figure()
xs = np.linspace(-4, 4, 100)
model_log_density = np.exp(outs['model'].log_density(
    t.tensor(xs).float(), inds=[0]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
model_log_density = np.exp(outs['model'].log_density(
    t.tensor(xs).float(), inds=[1]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
model_log_density = np.exp(outs['model'].log_density(
    t.tensor(xs).float(), inds=[2]).cpu().detach().numpy())
plt.plot(xs, model_log_density)
plt.show()

#%% spiral
M = 20000
n = np.sqrt(np.random.rand(M // 2, 1)) * 540 * (2 * np.pi) / 360
d1x = -np.cos(n) * n + np.random.rand(M // 2, 1) * 0.5
d1y = np.sin(n) * n + np.random.rand(M // 2, 1) * 0.5
x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
x += np.random.randn(*x.shape) * 0.1