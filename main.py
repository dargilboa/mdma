# %%
import torch as t
import numpy as np
import utils
import plots
import fit
import matplotlib.pyplot as plt

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.DoubleTensor')

#%% fit copula
d = 50
h = {
  'M': 500,
  'M_val': 500,
  'd': d,
  'n_iters': 600,
  'n': 100,
  'lambda_l2': 1e-5,
  'lambda_H_diag': 1e-5,
  'lambda_H_full': 0,
  #'opt': 'sgd',
  'lr': 5e-4,
  }

np.random.seed(1)
t.manual_seed(1)
P = utils.random_correlation_matrix(d)
data = utils.generate_data(h['d'], h['M'], h['M_val'], P)
outs = fit.fit_neural_copula(data, h)
plots.plot_contours_ext(outs, P)

#%% IAE plot
n_iters = 600
n_samples = 1000
rho = 0.6
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
n_reps = 1
n_sam_reps = 100
np.random.seed(1)
t.manual_seed(1)
file_name = 'IAEs_multisample'

IAEs = t.zeros(n_reps, len(ds), len(Ms), 2)
all_outs = []
for r in range(n_reps):
  for nd, d in enumerate(ds):
    for nM, M in enumerate(Ms):
      h = {
        'M': M,
        'M_val': 500,
        'd': d,
        'n_iters': n_iters,
        'n': 100,
        'lambda_l2': 1e-5,
        'lambda_H_diag': 1e-5,
        'checkpoint_every': n_iters,
      }
      P = np.eye(d) * (1 - rho) + np.ones((d,d)) * rho
      data = utils.generate_data(h['d'], h['M'], h['M_val'], P)
      outs = fit.fit_neural_copula(data, h)
      C = outs['best_val_NLL_model']

      curr_IAE = []
      for rs in range(n_sam_reps):
        us, _ = utils.generate_data(d, n_samples, 0, P)
        gd = utils.gaussian_copula_density(us, P)
        cd = t.exp(C.log_density(us))
        IAE = t.mean(t.abs(cd - gd) / gd)
        curr_IAE.append(IAE.cpu().detach().numpy())
      IAEs[r, nd, nM] = np.mean(curr_IAE), np.std(curr_IAE)

      outs['IAEs'] = [np.mean(curr_IAE), np.std(curr_IAE)]
      outs.pop('model')
      outs.pop('checkpoints')
      outs.pop('best_val_NLL_model')
      outs['best_val_NLL_model_params'] = [p for p in C.parameters()]
      all_outs += [outs]
      #np.save(file_name, [all_outs, IAEs, IAE_ms])
      #print('Outputs saved to '.format(file_name))


#%% IAEs plot
IAEs = np.squeeze(IAEs)
fig, axs = plt.subplots(1,3, figsize=(9,3))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  m, s = IAEs[nd][:,0], IAEs[nd][:,1]
  axs[nd].plot(Ms, m, label='WO marginal densities')
  axs[nd].fill_between(Ms, m - s, m + s, alpha=0.3)
  axs[nd].set_title(f'd={d}')

for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
plt.tight_layout()
plt.show()

#%% generate plots from saved data
all_outs, IAEs, IAE_ms = np.load('IAEs_multisample.npy', allow_pickle=True)
IAEs = t.squeeze(IAEs).detach().numpy()
IAE_ms = t.squeeze(IAE_ms).detach().numpy()

import models
n_iters = 600
n_samples = 10000
rho = 0.6
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
n_reps = 1
n_sam_reps = 100
np.random.seed(1)
t.manual_seed(1)

IAEs = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
IAE_ms = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
r = 0

for outs in all_outs:
  d = outs['h']['d']
  M = outs['h']['M']
  C = models.CopNet(100, d)
  saved_params = outs['best_val_NLL_model_params']
  C.W, C.b, C.a = saved_params
  P = np.eye(d) * (1 - rho) + np.ones((d,d)) * rho
  nd = ds.index(d)
  nM = Ms.index(M)
  for rs in range(n_sam_reps):
    us = t.rand(n_samples, d)
    gd = utils.gaussian_copula_density(us, P)
    cd = t.exp(C.log_density(us))
    IAE = t.mean(t.abs(cd - gd))
    IAE_m = 0
    # print(IAE, IAE_m)

    IAEs[r, nd, nM, rs] = IAE
    IAE_ms[r, nd, nM, rs] = IAE_m
  print('.')

IAEs = t.squeeze(IAEs).detach().numpy()
IAE_ms = t.squeeze(IAE_ms).detach().numpy()

#%% IAE plot
fig, axs = plt.subplots(1,3, figsize=(9,3))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  m, s = np.mean(IAEs[nd], axis=-1), np.std(IAEs[nd], axis=-1)
  axs[nd].plot(Ms, m, label='WO marginal densities')
  axs[nd].fill_between(Ms, m - s, m + s, alpha=0.3)
  axs[nd].set_title(f'd={d}')

for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
plt.tight_layout()
plt.show()