# %%
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pyvinecopulib as pv
import time

import utils
import plots
import fit

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.DoubleTensor')

#%% fit copula density
d = 5
h = {
    'M': 2000,
    'batch_size': 2000,
    'M_val': 500,
    'd': d,
    'n_epochs': 100,
    'n': 100,
    'lambda_l2': 1e-5,
    'lambda_hess_diag': 0,
    'lambda_hess_full': 0,
    #'opt': 'sgd',
    'lr': 5e-3,
}

np.random.seed(1)
t.manual_seed(1)

copula_type = 'gumbel'
copula_params = 1.67
data = utils.generate_c_data(
    h['d'],
    h['M'],
    h['M_val'],
    copula_params=copula_params,
    copula_type=copula_type,
)
outs = fit.fit_neural_copula(data, h)
plots.plot_contours_ext(outs,
                        copula_params=copula_params,
                        copula_type=copula_type)

#%% fit full density (copula + marginals)
d = 3
h = {
    'M': 2000,
    'M_val': 500,
    'd': d,
    'n_epochs': 1000,
    'batch_size': 2000,
    'n': 100,
    'lambda_l2': 1e-5,
    'lr': 5e-3,
    'fit_marginals': True,
}

np.random.seed(1)
t.manual_seed(1)

copula_type = 'gumbel'
copula_params = 1.67
marginal_type = 'gaussian'
marginal_params = [np.array([0] * d), np.array([1] * d)]
data = utils.generate_data(h['d'],
                           h['M'],
                           h['M_val'],
                           copula_params=copula_params,
                           marginal_params=marginal_params,
                           copula_type=copula_type,
                           marginal_type=marginal_type)
start_time = time.time()
outs = fit.fit_neural_copula(data, h)
run_time = (time.time() - start_time) / 60
print(f'Runtime: {run_time:.3g} mins')
plots.plot_contours_ext(
    outs,
    copula_params=copula_params,
    copula_type=copula_type,
    marginal_type=marginal_type,
    marginal_params=marginal_params,
    model_includes_marginals=True,
)

#%% fit copula density
d = 5
h = {
    'M': 2000,
    'batch_size': 2000,
    'M_val': 500,
    'd': d,
    'n_epochs': 600,
    'n': 100,
    'lambda_l2': 1e-5,
    'lambda_hess_diag': 0,
    'lambda_hess_full': 0,
    #'opt': 'sgd',
    'lr': 5e-3,
}

np.random.seed(1)
t.manual_seed(1)

copula_type = 'gumbel'
copula_params = 1.67
data = utils.generate_c_data(
    h['d'],
    h['M'],
    h['M_val'],
    copula_params=copula_params,
    copula_type=copula_type,
)
outs = fit.fit_neural_copula(data, h)
plots.plot_contours_ext(outs,
                        copula_params=copula_params,
                        copula_type=copula_type)

train_data, val_data = data
# if too slow, then use trunc_lvl
vine_controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll])
vine_fit = pv.Vinecop(train_data.cpu().detach().numpy(),
                      controls=vine_controls)
neural_train_ll = -outs['best_val_nll_model'].nll(
    train_data).cpu().detach().numpy()
vine_train_ll = vine_fit.loglik(
    train_data.cpu().detach().numpy()) / train_data.shape[0]
neural_val_ll = -outs['best_val_nll_model'].nll(
    val_data).cpu().detach().numpy()
vine_val_ll = vine_fit.loglik(
    val_data.cpu().detach().numpy()) / val_data.shape[0]
print(
    'train LL(neural/vine): {:.4f}/{:.4f}, val LL(neural/vine): {:.4f}/{:.4f}'.
    format(neural_train_ll, vine_train_ll, neural_val_ll, vine_val_ll))

#%% iae estimation
n_iters = 600
n_samples = 1000
rho = 0.6
theta = 1.67
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
n_reps = 1
n_sam_reps = 100
np.random.seed(1)
t.manual_seed(1)
file_name = 'iaes_gumbel'
copula_types = []
# copula_types += [{'type': 'gaussian',
#                   'd': d,
#                   'copula_params': np.eye(d) * (1 - rho) + np.ones((d,d)) * rho} for d in ds]
copula_types += [{
    'type': 'gumbel',
    'd': d,
    'copula_params': theta
} for d in ds]

iaes = np.zeros((n_reps, len(copula_types), len(Ms), 4))
all_outs = []
for r in range(n_reps):
  for nd, cop in enumerate(copula_types):
    d = cop['d']
    print(cop)
    for nM, M in enumerate(Ms):
      h = {
          'M': M,
          'M_val': 500,
          'd': d,
          'n_iters': n_iters,
          'n': 100,
          'lambda_l2': 1e-5,
          'lambda_hess_diag': 0,
          'checkpoint_every': n_iters,
      }
      P = np.eye(d) * (1 - rho) + np.ones((d, d)) * rho
      data = utils.generate_c_data(d,
                                   M,
                                   h['M_val'],
                                   copula_type=cop['type'],
                                   copula_params=cop['copula_params'])
      outs = fit.fit_neural_copula(data, h)
      neural_fit = outs['best_val_nll_model']
      vine_fit = pv.Vinecop(data[0].cpu().detach().numpy(),
                            controls=vine_controls)
      curr_iae_neural = []
      curr_iae_vine = []
      for rs in range(n_sam_reps):
        us, _ = utils.generate_c_data(d,
                                      n_samples,
                                      0,
                                      copula_type=cop['type'],
                                      copula_params=cop['copula_params'])
        gd = utils.copula_density(us.cpu(),
                                  copula_type=cop['type'],
                                  copula_params=cop['copula_params'])
        gd_np = gd.cpu().detach().numpy()
        cd = t.exp(neural_fit.log_density(us))
        vd = vine_fit.pdf(us.cpu().detach().numpy())
        iae_neural = t.mean(t.abs(cd - gd) / gd)
        iae_vine = np.mean(abs(vd - gd_np) / gd_np)
        curr_iae_neural.append(iae_neural.cpu().detach().numpy())
        curr_iae_vine.append(iae_vine)
      iaes[r, nd, nM] = np.mean(curr_iae_neural), np.std(
          curr_iae_neural), np.mean(curr_iae_vine), np.std(curr_iae_vine)
      all_outs += [outs]
      np.save(file_name, iaes)
      print(f'Outputs saved to {file_name}')
      # outs['iaes'] = [np.mean(curr_iae), np.std(curr_iae)]
      # outs.pop('model')
      # outs.pop('checkpoints')
      # outs.pop('best_val_nll_model')
      # outs['best_val_nll_model_params'] = [p for p in neural_fit.parameters()]

#%% IAEs plot
iaes = np.squeeze(iaes)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  m_neural, s_neural = iaes[nd][:, 0], iaes[nd][:, 1]
  m_vine, s_vine = iaes[nd][:, 2], iaes[nd][:, 3]
  axs[nd].plot(Ms, m_neural, label='neural')
  axs[nd].plot(Ms, m_vine, label='vine')
  axs[nd].fill_between(Ms, m_neural - s_neural, m_neural + s_neural, alpha=0.3)
  axs[nd].fill_between(Ms, m_vine - s_vine, m_vine + s_vine, alpha=0.3)
  axs[nd].set_title(f'd={d}')

for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')

plt.tight_layout()
plt.show()

#%% generate plots from saved data
all_outs, iaes, iae_ms = np.load('iaes_multisample.npy', allow_pickle=True)
iaes = t.squeeze(iaes).detach().numpy()
iae_ms = t.squeeze(iae_ms).detach().numpy()

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

iaes = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
iae_ms = t.zeros(n_reps, len(ds), len(Ms), n_sam_reps)
r = 0

for outs in all_outs:
  d = outs['h']['d']
  M = outs['h']['M']
  neural_fit = models.CopNet(d, 100)
  saved_params = outs['best_val_nll_model_params']
  neural_fit.w, neural_fit.b, neural_fit.a = saved_params
  P = np.eye(d) * (1 - rho) + np.ones((d, d)) * rho
  nd = ds.index(d)
  nM = Ms.index(M)
  for rs in range(n_sam_reps):
    us = t.rand(n_samples, d)
    gd = utils.gaussian_copula_density(us, P)
    cd = t.exp(neural_fit.log_density(us))
    iae = t.mean(t.abs(cd - gd))
    iae_m = 0
    # print(iae, iae_m)

    iaes[r, nd, nM, rs] = iae
    iae_ms[r, nd, nM, rs] = iae_m
  print('.')

iaes = t.squeeze(iaes).detach().numpy()
iae_ms = t.squeeze(iae_ms).detach().numpy()

#%% IAE plot
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
Ms = [200, 500, 1000, 2500, 5000]
ds = [3, 5, 10]
for nd, d in enumerate(ds):
  m, s = np.mean(iaes[nd], axis=-1), np.std(iaes[nd], axis=-1)
  axs[nd].plot(Ms, m, label='WO marginal densities')
  axs[nd].fill_between(Ms, m - s, m + s, alpha=0.3)
  axs[nd].set_title(f'd={d}')

for ax in axs.flatten():
  ax.set_xlabel('M')
  ax.set_ylabel('$\\widehat{IAE}$')
plt.tight_layout()
plt.show()
