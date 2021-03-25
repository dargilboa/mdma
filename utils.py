import torch as t
import numpy as np
from scipy.stats import norm
from torch.distributions.normal import Normal
from copulae import GumbelCopula


class d_interpolator():  # old, replaced with xitorch interpolation
  # interpolate first derivative
  def __init__(self, x, y):
    # we assume the function is monotonic
    self.x = np.sort(np.squeeze(x))
    self.y = np.sort(np.squeeze(y))
    self.dydx = np.diff(self.y) / np.diff(self.x)

  def interpolate_derivative(self, t):
    inds = np.searchsorted(self.x, t)
    # extrapolating using the values at the edges
    max_diff_ind = len(self.x) - 2
    inds = [min(max(ind - 1, 0), max_diff_ind) for ind in inds]

    return np.take(self.dydx, inds)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def invsigmoid(x, eps=1e-15):
  return t.log(x + eps) - t.log(1 - x + eps)


def sigmoiddot(x):
  return t.sigmoid(x) * (1 - t.sigmoid(x))


def ddsigmoid(x):
  return t.sigmoid(x) * (1 - t.sigmoid(x))**2 - t.sigmoid(x)**2 * (
      1 - t.sigmoid(x))


def dddsigmoid(x):
  return sigmoiddot(x) * (
      (1 - t.sigmoid(x))**2 + t.sigmoid(x)**2 - 4 * sigmoiddot(x))


def invsigmoiddot(x):
  return 1 / (x * (1 - x))


def generate_data(d,
                  M,
                  M_val,
                  copula_params,
                  marginal_params,
                  copula_type='gaussian',
                  marginal_type='gaussian'):
  copula_data = generate_c_data(d,
                                M,
                                M_val,
                                copula_params,
                                copula_type=copula_type)
  if marginal_type == 'gaussian':
    mus, sigmas = marginal_params
    data = [norm.ppf(cd) * sigmas + mus for cd in copula_data]
  else:
    raise ('Unknown marginal type')

  return data


def generate_c_data(d, M, M_val, copula_params, copula_type='gaussian'):
  # copula_params is either a d x d correlation matrix for the gaussian case or a theta parameter
  # for the gumbel case
  # returns [train_data, val_data]
  # train_data : M x d
  # val_data : M_val x d

  if copula_type == 'gaussian':
    P = copula_params
    A = np.linalg.cholesky(P)
    Z = np.random.randn(d, M + M_val)
    Z = np.dot(A, Z)
    U = norm.cdf(Z).transpose()
    return [t.Tensor(U[:M]), t.Tensor(U[M:])]
  elif copula_type == 'gumbel':
    theta = copula_params
    gumbel_copula = GumbelCopula(theta=theta, dim=d)
    U = gumbel_copula.random(M + M_val)
    return [t.Tensor(U[:M]), t.Tensor(U[M:])]


def gaussian_copula_log_density(u, rho):
  rho = t.tensor([rho])
  normal = Normal(loc=0, scale=1, validate_args=None)
  icdfu = normal.icdf(u)
  exponent = (rho**2 * (icdfu[:, 0]**2 + icdfu[:, 1]**2) -
              2 * rho * icdfu[:, 0] * icdfu[:, 1]) / (2 * (1 - rho**2))
  return -t.log(t.sqrt(1 - rho**2)) - exponent


def copula_log_density(u, copula_params, copula_type='gaussian'):
  return t.log(copula_density(u, copula_params, copula_type=copula_type))


def copula_density(us, copula_params, copula_type='gaussian'):
  # us : M x d tensor of points to sample
  d = us.shape[1]

  if copula_type == 'gaussian':
    P = t.tensor(copula_params)
    normal = Normal(loc=0, scale=1, validate_args=None)
    ius = normal.icdf(us)
    exponent = t.exp(-(1 / 2) *
                     t.sum(ius @ (t.inverse(P) - t.eye(d)) * ius, dim=1))
    return exponent * 1 / t.sqrt(t.det(P))
  elif copula_type == 'gumbel':
    theta = copula_params
    gumbel_copula = GumbelCopula(theta=theta, dim=d)
    return t.tensor(gumbel_copula.pdf(us))


def density_w_gauss_marginals(us, copula_density):
  # us : M x d tensor of points on the hypercube
  # copula_density : the copula density at us
  # returns the density assuming gaussian marginals
  normal = Normal(loc=0, scale=1, validate_args=None)
  ius = normal.icdf(us)
  return copula_density * t.exp(t.sum(normal.log_prob(ius), dim=1))


def gauss_marginals(us):
  # us : M x d tensor of points on the hypercube
  # returns product of gaussian densities
  normal = Normal(loc=0, scale=1, validate_args=None)
  ius = normal.icdf(us)
  return t.exp(t.sum(normal.log_prob(ius), dim=1))


def bisect(f, x, lb, ub, n_iter=35):
  # inverts a scalar function f by bisection at x points
  xl = t.ones_like(x) * lb
  xh = t.ones_like(x) * ub
  x_temp = (xl + xh) / 2

  for _ in range(n_iter):
    fmx = f(x_temp) - x
    xl = t.where(fmx < 0, x_temp, xl)
    xh = t.where(fmx < 0, xh, x_temp)
    x_temp = (xl + xh) / 2

  return x_temp


def invert(f, r, lb=0, ub=1, n_bisect_iter=35):
  # return f^{-1}(r), assuming lb <= f^{-1}(r) <= ub
  return bisect(f, r, lb, ub, n_iter=n_bisect_iter)


def stabilize(u, stab_const=1e-5):
  return t.clamp(u, stab_const, 1 - stab_const)


def correlation_matrix(d, rho):
  return np.eye(d) * (1 - rho) + np.ones((d, d)) * rho


def random_correlation_matrix(d):
  S = np.random.randn(d, d)
  P = np.dot(S, S.transpose())
  P = P / np.sqrt(np.diag(P))[:, None] / np.sqrt(np.diag(P))[None, :]
  return P
