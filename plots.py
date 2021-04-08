import numpy as np
import torch as t
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
import utils
import pprint
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def plot_contours_single(
    model,
    axs,
    copula_params,
    marginal_params=None,
    copula_type='gaussian',
    marginal_type='gaussian',
    model_includes_marginals=False,
    title=None,
    eps=1e-3,
    grid_res=200,
    i=0,
    j=1,
    data_ij=None,
    contours=(-4.5, -3.4, -2.8, -2.2, -1.6),
):

  colors_k = ['k'] * len(contours)
  colors_r = ['r'] * len(contours)

  # create grid on hypercube and evaluate ground truth copula density
  x = np.linspace(eps, 1 - eps, grid_res)
  y = np.linspace(eps, 1 - eps, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0, 1)
  if copula_type != 'data':
    true_copula_log_density = utils.copula_log_density(
        flat_grid.cpu().detach().numpy(),
        copula_type=copula_type,
        copula_params=copula_params).cpu().detach().numpy().reshape(
            (grid_res, grid_res))

  if model_includes_marginals:
    # include the marginal densities
    if marginal_type == 'gaussian':
      iX, iY = norm.ppf(grid)
      mus, sigmas = marginal_params

      # modify iX, iY to match the marginals
      iX = iX * sigmas[0] + mus[0]
      iY = iY * sigmas[1] + mus[1]
    elif marginal_type == 'data':
      # get boundaries of plot from the samples
      margin = 0.05
      x_coords = np.linspace(min(data_ij[0]), max(data_ij[0]), grid_res)
      y_coords = np.linspace(min(data_ij[1]), max(data_ij[1]), grid_res)
      iX, iY = np.meshgrid(x_coords, y_coords)
    else:
      raise Exception('Unknown marginal type')

    # evaluate the density of the model, assuming it contains marginals
    flat_grid_on_R = t.tensor([g.flatten() for g in [iX, iY]]).transpose(0, 1)
    model_log_density = model.log_density(
        flat_grid_on_R, inds=[i, j]).cpu().detach().numpy().reshape(
            (grid_res, grid_res))

  else:
    # compute full densities assuming gaussian marginals
    iX, iY = norm.ppf(grid)

    # model doesn't include marginals, so evaluate log copula density
    model_log_copula_density = model.log_copula_density(
        flat_grid, inds=[i, j]).cpu().detach().numpy().reshape(
            (grid_res, grid_res))
    model_log_density = model_log_copula_density + norm.logpdf(
        iX) + norm.logpdf(iY)

  # plot the true density contours or ground truth samples
  if copula_type != 'data':
    true_log_density = true_copula_log_density + norm.logpdf(iX) + norm.logpdf(
        iY)
    axs.contour(iX, iY, true_log_density, contours, colors=colors_k)
  else:
    axs.scatter(data_ij[0], data_ij[1])
  axs.contour(iX, iY, model_log_density, contours, colors=colors_r)
  if title is not None:
    axs.title.set_text(title)


def plot_contours_ext(outs,
                      copula_params=None,
                      marginal_params=None,
                      copula_type=None,
                      marginal_type=None,
                      model_includes_marginals=False,
                      final_only=False,
                      data=None,
                      add_nll_plot=True):

  d = outs['h'].d
  if final_only:
    models = [[len(outs['nlls']) - 1, outs['model']]]
  else:
    models = zip(outs['checkpoint_iters'], outs['checkpoints'])

  for iter, model in models:
    fig, axs = plt.subplots(d - 1, d - 1, figsize=(d * 3, d * 3))
    if d == 2:
      axs = np.expand_dims(np.array(axs), [0, 1])
    for i in range(d - 1):
      for j in range(i, d - 1):
        if copula_type == 'gaussian':
          # create correlation matrix
          c_cop_params = np.array([[1, copula_params[i, j + 1]],
                                   [copula_params[i, j + 1], 1]])
        elif copula_type == 'gumbel':
          c_cop_params = copula_params
        else:
          c_cop_params = None

        # get marginal params for the plotted variables
        if model_includes_marginals:
          if marginal_type == 'gaussian':
            marginal_params_ij = [mp[[i, j]] for mp in marginal_params]
            data_ij = None
          elif marginal_type == 'data':
            marginal_params_ij = None
            data_ij = [data[:, i], data[:, j + 1]]
          else:
            raise Exception('Unknown marginal type')
        else:
          marginal_params_ij = None
          if copula_type == 'data':
            data_ij = [data[:, i], data[:, j + 1]]
          else:
            data_ij = None

        # plot contours
        plot_contours_single(model,
                             axs[i, j],
                             copula_params=c_cop_params,
                             copula_type=copula_type,
                             marginal_type=marginal_type,
                             marginal_params=marginal_params_ij,
                             model_includes_marginals=model_includes_marginals,
                             i=i,
                             j=j + 1,
                             data_ij=data_ij)
    for i in range(d - 1):
      axs[i, 0].set_ylabel(f'u_{i+1}')
      axs[0, i].set_title(f'u_{i+2}')

    # plot NLL
    if d > 3:
      ind_nll_plot = 1
      alpha = 1
    else:
      ind_nll_plot = 0
      alpha = 0.5
    if add_nll_plot:
      plot_nll(axs[-1, ind_nll_plot], outs, alpha, iter)

    axs[-1, 0].text(
        0, 0, '\n'.join([
            key + ' : ' + str(value)
            for key, value in outs['h'].__dict__.items()
        ]))
    nll_val = outs['nlls'][iter]
    axs[-1,
        0].set_xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      ',  nll: {:.3f}, iter: {}'.format(nll_val, iter))
    fig.show()


def plot_nll(ax, outs, alpha=1, iter_to_plot=-1):
  ax.plot(outs['nlls'], alpha=alpha, label='train')
  ax.plot(outs['val_nll_iters'], outs['val_nlls'], alpha=alpha, label='val')
  ax.legend()
  ax.grid()
  ax.scatter(iter_to_plot,
             outs['nlls'][iter_to_plot],
             color='red',
             alpha=alpha)


def plot_heatmap(model, outs, xlim=[-1, 1], ylim=[-1, 1], grid_res=100):
  fig, axs = plt.subplots(1, 3, figsize=(9, 3))
  x_coords = np.linspace(xlim[0], xlim[1], grid_res)
  y_coords = np.linspace(ylim[0], ylim[1], grid_res)
  iX, iY = np.meshgrid(x_coords, y_coords)
  flat_grid_on_R = t.tensor([g.flatten() for g in [iX, iY]]).transpose(0, 1)
  model_log_density = model.log_density(
      flat_grid_on_R).cpu().detach().numpy().reshape((grid_res, grid_res))
  fplot = axs[0].imshow(np.exp(model_log_density),
                        extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
  #fig.colorbar(fplot, ax=axs[0])

  plot_nll(axs[1], outs)

  axs[-1].text(
      0, 0, '\n'.join([
          key + ' : ' + str(value)
          for key, value in outs['h'].__dict__.items()
      ]))
  nll_val = outs['nlls'][-1]
  axs[-1].set_xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                     ',  final nll: {:.3f}'.format(nll_val))

  plt.show()


def plot_copula_density(
    model,
    data=None,
    eps=1e-3,
    grid_res=200,
):
  x = np.linspace(eps, 1 - eps, grid_res)
  y = np.linspace(eps, 1 - eps, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0, 1)
  model_log_copula_density = model.log_copula_density(
      flat_grid).cpu().detach().numpy().reshape((grid_res, grid_res))
  iX, iY = grid
  contours = [-4.5, -3.4, -2.8, -2.2, -1.6]
  colors_r = ['r'] * len(contours)
  plt.contour(iX, iY, model_log_copula_density, contours, colors=colors_r)
  if data is not None:
    plt.scatter(data[:, 0], data[:, 1])
  plt.show()


def plot_all_heatmaps(outs, xlim=[-1, 1], ylim=[-1, 1]):
  fig, axs = plt.subplots(1,
                          len(outs['checkpoints']),
                          figsize=(3 * len(outs['checkpoints']), 3))
  grid_res = 100
  x_coords = np.linspace(xlim[0], xlim[1], grid_res)
  y_coords = np.linspace(ylim[0], ylim[1], grid_res)
  iX, iY = np.meshgrid(x_coords, y_coords)
  flat_grid_on_R = t.tensor([g.flatten() for g in [iX, iY]]).transpose(0, 1)
  for ax, model, iter in zip(axs, outs['checkpoints'],
                             outs['checkpoint_iters']):
    model_log_density = model.log_density(
        flat_grid_on_R).cpu().detach().numpy().reshape((grid_res, grid_res))
    ax.imshow(np.exp(model_log_density),
              extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    ax.set_title(iter)

  plt.show()
