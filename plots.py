import numpy as np
import torch as t
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
import utils

def plot_contours(outs):
  C = outs['model']
  grid_res = 200
  x = np.linspace(0.001, .999, grid_res)
  y = np.linspace(0.001, .999, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0,1)
  log_densities = C.log_density(flat_grid).cpu().detach().numpy().reshape((grid_res,grid_res))
  gauss_log_densities = utils.gaussian_copula_log_density(flat_grid, rho=0.8).cpu().detach().numpy().reshape((grid_res,grid_res))

  iX, iY = norm.ppf(grid)
  contours = [-4.5, -3.4, -2.8, -2.2, -1.6]
  colors_k = ['k'] * len(contours)
  colors_r = ['r'] * len(contours)
  plt.contour(iX, iY, gauss_log_densities + norm.logpdf(iX) + norm.logpdf(iY), contours,
              colors=colors_k)
  plt.contour(iX, iY, log_densities + norm.logpdf(iX) + norm.logpdf(iY), contours,
              colors=colors_r)
  plt.title("n: {}, M: {}, n_iters: {}, $\lambda_{{L^2}}$: {}, $\lambda_H$: {}, \n b_std: {}, a_std: {}, W_std: {}, bptz: {}"
            .format(outs['n'], outs['M'], outs['n_iters'], outs['lambda_l2'], outs['lambda_H'], outs['b_std'],
                    outs['a_std'], outs['W_std'], outs['bp_through_z_update']))
  #plt.scatter(norm.ppf(data.cpu()[:,0]), norm.ppf(data.cpu()[:,1]))
  plt.xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ',  final NLL: {:.3f}'.format(outs['final_NLL']))
  plt.show()

def plot_contours_single(C, axs, rho, title=None, eps=1e-3, grid_res=200, i=0, j=1):
  # grid_points = (1 - eps) * np.ones(d, grid_res)
  # grid_points[i] = np.linspace(eps, 1 - eps, grid_res)
  # grid_points[j] = np.linspace(eps, 1 - eps, grid_res)
  x = np.linspace(eps, 1 - eps, grid_res)
  y = np.linspace(eps, 1 - eps, grid_res)
  grid = np.meshgrid(x, y)
  flat_grid = t.tensor([g.flatten() for g in grid]).transpose(0,1)
  log_densities = C.log_density(flat_grid, bivariate=True, bv_i=i, bv_j=j).cpu().detach().numpy().reshape((grid_res,
                                                                                                           grid_res))
  gauss_log_densities = utils.gaussian_copula_log_density(flat_grid, rho=rho).cpu().detach().numpy().reshape((grid_res,
                                                                                                              grid_res))

  iX, iY = norm.ppf(grid)
  contours = [-4.5, -3.4, -2.8, -2.2, -1.6]
  colors_k = ['k'] * len(contours)
  colors_r = ['r'] * len(contours)
  axs.contour(iX, iY, gauss_log_densities + norm.logpdf(iX) + norm.logpdf(iY), contours,
              colors=colors_k)
  axs.contour(iX, iY, log_densities + norm.logpdf(iX) + norm.logpdf(iY), contours,
              colors=colors_r)
  if title is not None:
    axs.title.set_text(title)

def plot_contours_ext(outs, P, final_only=False):
  d = outs['h']['d']
  if final_only:
    models = [[len(outs['NLLs'])-1, outs['model']]]
  else:
    models = zip(outs['checkpoint_iters'], outs['checkpoints'])

  for iter, model in models:
    fig, axs = plt.subplots(d - 1, d - 1, figsize=(d * 3, d * 3))
    for i in range(d - 1):
      for j in range(i, d - 1):
        plot_contours_single(model, axs[i, j], rho=P[i, j + 1], i=i, j=j + 1)
    axs[0, 0].set_ylabel('u_1')
    axs[1, 0].set_ylabel('u_2')
    axs[0, 0].set_title('u_2')
    axs[0, 1].set_title('u_3')
    axs[d - 2, 0].plot(outs['NLLs'], alpha=0.3)
    axs[d - 2, 0].scatter(iter, outs['NLLs'][iter], color='red', alpha=0.3)
    axs[d - 2, 0].text(0, min(outs['NLLs']), '\n'.join([key + ' : ' + str(value) for key, value in outs['h'].items()]))
    NLL_val = outs['NLLs'][iter]
    axs[1, 0].set_xlabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               + ',  NLL: {:.3f}, iter: {}'.format(NLL_val, iter))
    fig.show()