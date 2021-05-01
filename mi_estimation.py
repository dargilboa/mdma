#%%
import numpy as np
import matplotlib.pyplot as plt
import cdfnet.fit as fit
import cdfnet.utils as utils
import torch as t


def run_mi_estimation(d=5, n_samples=10000, batch_size=1000, M=100000, n=500):
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
  A = np.linalg.cholesky(Sigma)
  Z = np.random.randn(d, M)
  data = np.dot(A, Z).transpose()
  h = fit.get_default_h()
  h.batch_size = batch_size
  h.d = d
  h.eval_validation = False
  h.save_checkpoints = False
  h.eval_test = False
  h.n = n
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
  # samples = np.array([])
  # for i in range(n_samples // h.batch_size):
  #   samples = np.concatenate((samples, model.sample(h.batch_size)))
  # from pytorch_memlab import LineProfiler
  # import cdfnet.models
  #
  # with LineProfiler(cdfnet.models.CDFNet.sample,
  #                   cdfnet.models.CDFNet.condCDF) as prof:
  print('Sampling')
  samples = model.sample(n_samples, batch_size=h.batch_size)
  samples = t.Tensor(samples)

  #prof.display()
  print('Computing mutual information')
  ind_rng = range(1, d)
  mis = []
  for i in ind_rng:
    mis += [(1 / 2) * np.log(
        np.linalg.det(Sigma[:i, :i]) * np.linalg.det(Sigma[i:, i:]) /
        np.linalg.det(Sigma))]
  all_mi_ests = []
  samples_dataloader = t.utils.data.DataLoader(samples,
                                               batch_size=h.batch_size,
                                               shuffle=False)
  for batch_idx, batch in enumerate(samples_dataloader):
    mi_ests = []
    for i in ind_rng:
      mi_ests += [
          t.mean(
              model.log_density(batch) -
              model.log_density(batch[:, range(i)], inds=range(i)) -
              model.log_density(batch[:, range(i, d)], inds=range(i, d))).cpu(
              ).detach().numpy()
      ]
    all_mi_ests.append(mi_ests)
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
  all_mi_ests = np.array(all_mi_ests)
  plt.figure()
  plt.scatter(ind_rng, mis, label='ground truth')
  plt.scatter(ind_rng, all_mi_ests.mean(axis=0), label='estimator')
  m, s = all_mi_ests.mean(
      axis=0), all_mi_ests.std(axis=0) / np.sqrt(len(all_mi_ests))
  plt.errorbar(ind_rng,
               m,
               yerr=[m - s, m + s],
               color='orange',
               ls='none',
               capsize=5)
  plt.ylabel('$I(X_1;X_2)$')
  plt.xticks(range(1, d))
  plt.xlabel('$\mathrm{dim}(X_1)$')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  run_mi_estimation()
