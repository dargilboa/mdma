import numpy as np
import torch as t
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
import utils


def plot_contours(outs):

  model = outs['model']

  grid_res = 200
  x = np.linspace(0.001, .999, grid_res)
  y = np.linspace(0.001, .999, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0, 1)
  log_densities = model.log_density(flat_grid).cpu().detach().numpy().reshape(
      (grid_res, grid_res))
  gauss_log_densities = utils.gaussian_copula_log_density(
      flat_grid, rho=0.8).cpu().detach().numpy().reshape((grid_res, grid_res))

  iX, iY = norm.ppf(grid)
  contours = [-4.5, -3.4, -2.8, -2.2, -1.6]
  colors_k = ['k'] * len(contours)
  colors_r = ['r'] * len(contours)
  plt.contour(iX,
              iY,
              gauss_log_densities + norm.logpdf(iX) + norm.logpdf(iY),
              contours,
              colors=colors_k)
  plt.contour(iX,
              iY,
              log_densities + norm.logpdf(iX) + norm.logpdf(iY),
              contours,
              colors=colors_r)
  plt.title(
      "n: {}, M: {}, n_iters: {}, $\lambda_{{L^2}}$: {}, $\lambda_H$: {}, \n b_std: {}, a_std: {}, W_std: {}, bptz: {}"
      .format(outs['n'], outs['M'], outs['n_iters'], outs['lambda_l2'],
              outs['lambda_hess'], outs['b_std'], outs['a_std'], outs['_std'],
              outs['bp_through_z_update']))
  #plt.scatter(norm.ppf(data.cpu()[:,0]), norm.ppf(data.cpu()[:,1]))
  plt.xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
             ',  final nll: {:.3f}'.format(outs['final_nll']))
  plt.show()


def plot_contours_single(model,
                         axs,
                         copula_params,
                         copula_type='gaussian',
                         title=None,
                         eps=1e-3,
                         grid_res=200,
                         i=0,
                         j=1):
  x = np.linspace(eps, 1 - eps, grid_res)
  y = np.linspace(eps, 1 - eps, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0, 1)
  log_densities = model.log_density(flat_grid, bivariate=True, bv_i=i,
                                    bv_j=j).cpu().detach().numpy().reshape(
                                        (grid_res, grid_res))
  true_log_densities = utils.copula_log_density(
      flat_grid.cpu(), copula_type=copula_type,
      copula_params=copula_params).cpu().detach().numpy().reshape(
          (grid_res, grid_res))

  iX, iY = norm.ppf(grid)
  contours = [-4.5, -3.4, -2.8, -2.2, -1.6]
  colors_k = ['k'] * len(contours)
  colors_r = ['r'] * len(contours)
  axs.contour(iX,
              iY,
              true_log_densities + norm.logpdf(iX) + norm.logpdf(iY),
              contours,
              colors=colors_k)
  axs.contour(iX,
              iY,
              log_densities + norm.logpdf(iX) + norm.logpdf(iY),
              contours,
              colors=colors_r)
  if title is not None:
    axs.title.set_text(title)


def plot_contours_ext(outs,
                      copula_params,
                      copula_type='gaussian',
                      final_only=False):

  d = outs['h']['d']
  if final_only:
    models = [[len(outs['nlls']) - 1, outs['model']]]
  else:
    models = zip(outs['checkpoint_iters'], outs['checkpoints'])

  for iter, model in models:
    fig, axs = plt.subplots(d - 1, d - 1, figsize=(d * 3, d * 3))
    for i in range(d - 1):
      for j in range(i, d - 1):
        if copula_type == 'gaussian':
          # create correlation matrix
          c_cop_params = np.array([[1, copula_params[i, j + 1]],
                                   [copula_params[i, j + 1], 1]])
        elif copula_type == 'gumbel':
          c_cop_params = copula_params
        plot_contours_single(model,
                             axs[i, j],
                             copula_params=c_cop_params,
                             copula_type=copula_type,
                             i=i,
                             j=j + 1)
    axs[0, 0].set_ylabel('u_1')
    axs[1, 0].set_ylabel('u_2')
    axs[0, 0].set_title('u_2')
    axs[0, 1].set_title('u_3')
    if d > 3:
      ind_nll_plot = 1
      alpha = 1
    else:
      ind_nll_plot = 0
      alpha = 0.5
    axs[-1, ind_nll_plot].plot(outs['nlls'], alpha=alpha, label='train')
    axs[-1, ind_nll_plot].plot(outs['val_nlls'], alpha=alpha, label='val')
    axs[-1, ind_nll_plot].legend()
    axs[-1, ind_nll_plot].grid()
    axs[-1, ind_nll_plot].scatter(iter,
                                  outs['nlls'][iter],
                                  color='red',
                                  alpha=alpha)
    axs[-1, 0].text(
        0, 0, '\n'.join(
            [key + ' : ' + str(value) for key, value in outs['h'].items()]))
    nll_val = outs['nlls'][iter]
    axs[-1,
        0].set_xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      ',  nll: {:.3f}, iter: {}'.format(nll_val, iter))
    fig.show()