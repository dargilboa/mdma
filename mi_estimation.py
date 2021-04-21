#%%
import numpy as np
import matplotlib.pyplot as plt
import cdfnet.fit as fit
import cdfnet.utils as utils
import torch as t

d = 10
Sigma = np.eye(10)
for i in range(d):
  for j in range(d):
    if i != j:
      Sigma[i, j] = (i + j)**1.2 / 40

# plt.imshow(Sigma)
# plt.colorbar()
# plt.show()

np.linalg.det(Sigma)

mis = []
ind_rng = range(1, d)
for i in ind_rng:
  mis += [(1 / 2) * np.log(
      np.linalg.det(Sigma[:i, :i]) * np.linalg.det(Sigma[i:, i:]) /
      np.linalg.det(Sigma))]

plt.bar(ind_rng, mis)
plt.show()

#%%
M = 100000
A = np.linalg.cholesky(Sigma)
Z = np.random.randn(d, M)
data = np.dot(A, Z).transpose()
h = fit.get_default_h()
h.batch_size = 1000
h.d = d
h.eval_validation = False
h.eval_test = False
h.n = 100
h.m = 5
h.L = 4
h.n_epochs = 1
h.model_to_load = ''
h.M = M
loaders = utils.create_loaders([data, 0, 0], h.batch_size)
model = fit.fit_neural_copula(h, loaders)

#%%
n_samples = 1000
samples = model.sample(n_samples)
samples = t.Tensor(samples)
mis = []
mi_ests = []
for i in ind_rng:
  mis += [(1 / 2) * np.log(
      np.linalg.det(Sigma[:i, :i]) * np.linalg.det(Sigma[i:, i:]) /
      np.linalg.det(Sigma))]
  mi_ests += [(1 / 2) * t.mean(
      model.log_density(samples[:, range(i)], inds=range(i)) +
      model.log_density(samples[:, range(i + 1, d)], inds=range(i + 1, d)) -
      model.log_density(samples)).cpu().detach().numpy()]
