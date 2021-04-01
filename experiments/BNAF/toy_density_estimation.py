import torch as t
import numpy as np
import matplotlib.pyplot as plt
import time
import plots
import fit
from data.generate2d import sample2d

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.DoubleTensor')
"""
settings for 8gaussians dataset
n = 100
n_m = 10
L_m = 6
lr = 0.1
batch_size = 2000
n_iters = 500

settings for checkerboard
lr = 5e-2
n_iters = 500
'w_m_std': 0.01, (before new normalization)
'a_m_std': 0.01,

"""

#%% fit density
d = 2
dataset = '2spirals'
batch_size = 2000
n_iters = 500
M = n_iters * batch_size
h = {
    'M': M,
    'M_val': 500,
    'd': d,
    'n_epochs': 1,
    'batch_size': batch_size,
    'n': 500,
    'lambda_l2': 0,
    'lambda_l2_m': 0,
    'lr': 5e-3,
    'fit_marginals': True,
    'dataset': dataset,
    'n_m': 50,
    'L_m': 6,
    'w_m_std': 0.01,
    'a_m_std': 0.01,
    'decrease_lr_time': 0.5,
}

np.random.seed(1)
t.manual_seed(1)

data = [sample2d(h['dataset'], h['M']), sample2d(h['dataset'], h['M_val'])]
start_time = time.time()
outs = fit.fit_neural_copula(data, h)
run_time = (time.time() - start_time) / 60
print(f'Runtime: {run_time:.3g} mins')

plots.plot_contours_ext(
    outs,
    model_includes_marginals=True,
    copula_type='data',
    marginal_type='data',
    data=sample2d(h['dataset'], h['M_val']),
    add_nll_plot=False,
)
plt.plot(outs['nlls'])
plt.show()

plots.plot_heatmap(outs['model'], outs, xlim=[-4, 4], ylim=[-4, 4])

#%% plot marginal density
x_rng = 5
lx = 100
xs = t.unsqueeze(t.linspace(-x_rng, x_rng, lx), -1)

plt.figure()
plt.plot(
    xs.cpu().detach().numpy(),
    outs['model'].marginal_likelihood(xs, inds=[0]).cpu().detach().numpy())
plt.hist(data[0][:, 0], 50, alpha=0.5, density=True)
plt.show()

#%% checkerboard on hypercube
d = 2
dataset = 'checkerboard'
batch_size = 2000
n_iters = 1000
M = n_iters * batch_size
h = {
    'M': M,
    'M_val': 500,
    'd': d,
    'n_epochs': 1,
    'batch_size': batch_size,
    'n': 500,
    'lambda_l2': 0,
    'lambda_l2_m': 0,
    'lr': 1e-2,
    'fit_marginals': False,
    'dataset': dataset,
}

np.random.seed(1)
t.manual_seed(1)

data = [sample2d(h['dataset'], h['M']), sample2d(h['dataset'], h['M_val'])]
data[0] = (data[0] + 4) / 8
data[1] = (data[1] + 4) / 8

start_time = time.time()
outs = fit.fit_neural_copula(data, h)
run_time = (time.time() - start_time) / 60
print(f'Runtime: {run_time:.3g} mins')

plots.plot_contours_ext(
    outs,
    model_includes_marginals=False,
    copula_type='data',
    marginal_type='data',
    data=data[1],
    add_nll_plot=False,
)

plt.plot(outs['nlls'])
plt.show()

plots.plot_copula_density(outs['model'], data[1])
