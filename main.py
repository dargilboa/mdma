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

IAEs = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
IAE_ms = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
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

      for rs in range(n_sam_reps):
        us = t.rand(n_samples, d)
        gd = utils.gaussian_copula_density(us, P)
        cd = t.exp(C.log_density(us))
        IAE = t.mean(t.abs(cd - gd) * gd)
        IAE_m = t.mean(t.abs(cd - gd) * utils.gauss_marginals(us) * gd)
        #print(IAE, IAE_m)
        IAEs[r, nd, nM, rs] = IAE
        IAE_ms[r, nd, nM, rs] = IAE_m

      outs['IAEs'] = [IAE, IAE_m]
      outs.pop('model')
      outs.pop('checkpoints')
      outs.pop('best_val_NLL_model')
      outs['best_val_NLL_model_params'] = [p for p in C.parameters()]
      all_outs += [outs]
      #np.save(file_name, [all_outs, IAEs, IAE_ms])
      #print('Outputs saved to '.format(file_name))

#%%
all_outs, IAEs, IAE_ms = np.load('IAEs_multisample.npy', allow_pickle=True)
IAEs = t.squeeze(IAEs).detach().numpy()
IAE_ms = t.squeeze(IAE_ms).detach().numpy()

#%%
fig, axs = plt.subplots(3,2, figsize=(6,9))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  m, s = np.mean(IAEs[nd], axis=-1), np.std(IAEs[nd], axis=-1)
  axs[nd, 0].plot(Ms, m, label='WO marginal densities')
  axs[nd, 0].fill_between(Ms, m - s, m + s, alpha=0.3)
  axs[nd, 0].set_title(f'd={d}')
  m, s = np.mean(IAE_ms[nd], axis=-1), np.std(IAE_ms[nd], axis=-1)
  axs[nd, 1].plot(Ms, m, label='with marginal densities')
  axs[nd, 1].fill_between(Ms, m - s, m + s, alpha=0.3)
for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
axs[0,0].set_title('no marginal densities')
axs[0,1].set_title('with marginal densities')
plt.tight_layout()
plt.show()

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

#%%
all_outs, IAEs, IAE_ms = np.load('IAEs.npy', allow_pickle=True)
IAEs = t.squeeze(IAEs).detach().numpy()
IAE_ms = t.squeeze(IAE_ms).detach().numpy()

#%%
fig, axs = plt.subplots(3,2, figsize=(6,9))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  axs[nd, 0].plot(Ms, IAEs[nd], label='WO marginal densities')
  axs[nd, 1].plot(Ms, IAE_ms[nd], label='with marginal densities')
for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
axs[0,0].set_title('d=3, no marginal densities')
axs[0,1].set_title('d=3, with marginal densities')
plt.tight_layout()
plt.show()

#%%
fig, axs = plt.subplots(1,2, figsize=(6,3))
axs[0].plot(Ms, t.squeeze(MISEs).detach().numpy(), label='WO marginal densities')
axs[1].plot(Ms, t.squeeze(MISE2s).detach().numpy(), label='with marginal densities')
for ax in axs:
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
axs[0].set_title('d=3, no marginal densities')
axs[1].set_title('d=3, with marginal densities')
plt.tight_layout()
plt.show()