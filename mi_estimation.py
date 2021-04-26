#%%
import numpy as np
import matplotlib.pyplot as plt
import cdfnet.fit as fit
import cdfnet.utils as utils
import torch as t

d = 5
Sigma = np.eye(d)
for i in range(d):
  for j in range(d):
    if i != j:
      Sigma[i, j] = (i + j) / (5 * d)

# plt.imshow(Sigma)
# plt.colorbar()
# plt.show()
print(np.linalg.det(Sigma))

#%%
M = 10000
A = np.linalg.cholesky(Sigma)
Z = np.random.randn(d, M)
data = np.dot(A, Z).transpose()
h = fit.get_default_h()
h.batch_size = 1000
h.d = d
h.eval_validation = False
h.eval_test = False
h.n = 500
h.use_HT = 1
h.m = 5
h.L = 4
h.n_epochs = 2
h.model_to_load = ''
h.save_path = '.'
h.M = M
h.patience = 200
loaders = utils.create_loaders([data, 0, 0], h.batch_size)
model = fit.fit_neural_copula(h, loaders)

#%%
n_samples = 1000
# samples = np.array([])
# for i in range(n_samples // h.batch_size):
#   samples = np.concatenate((samples, model.sample(h.batch_size)))
# from pytorch_memlab import LineProfiler
# import cdfnet.models
#
# with LineProfiler(cdfnet.models.CDFNet.sample,
#                   cdfnet.models.CDFNet.condCDF) as prof:
samples = model.sample(n_samples)

#prof.display()

samples = t.Tensor(samples)
ind_rng = range(1, d)
mis = []
mi_ests = []
for i in ind_rng:
  mis += [(1 / 2) * np.log(
      np.linalg.det(Sigma[:i, :i]) * np.linalg.det(Sigma[i:, i:]) /
      np.linalg.det(Sigma))]
  mi_ests += [
      t.mean(
          model.log_density(samples) -
          model.log_density(samples[:, range(i)], inds=range(i)) -
          model.log_density(samples[:, range(i, d)], inds=range(i, d))).cpu().
      detach().numpy()
  ]
# mis = []
# mi_ests = []
# for i in ind_rng:
#   mis += [
#       -(1 / 2) * np.log((2 * np.pi)**i * np.linalg.det(Sigma[:i, :i])) - i / 2
#   ]
#   mi_ests += [
#       t.mean(model.log_density(samples[:, range(i)],
#                                inds=range(i))).cpu().detach().numpy()
#   ]
# mis = []
# mi_ests = []
# for i in ind_rng:
#   mis += [
#       -(1 / 2) * np.log(
#           (2 * np.pi)**(d - i) * np.linalg.det(Sigma[i:, i:])) - (d - i) / 2
#   ]
#   mi_ests += [
#       t.mean(model.log_density(samples[:, range(i, d)],
#                                inds=range(i, d))).cpu().detach().numpy()
#   ]
plt.figure()
plt.scatter(ind_rng, mis, label='ground truth')
plt.scatter(ind_rng, mi_ests, label='estimator')
plt.ylabel('$I(X_1;X_2)$')
plt.xticks(range(1, d))
plt.xlabel('$\mathrm{dim}(X_1)$')
plt.legend()
plt.show()

#%%
