import torch as t
import torch.nn as nn
from xitorch.interpolate import Interp1D
import utils
from operator import itemgetter


# %%
class CopNet(nn.Module):
  def __init__(
      self,
      d,
      n=100,
      z_update_samples_scale=3,
      z_update_samples=1000,
      w_std=0.1,
      b_bias=0,
      b_std=0.1,
      a_std=1.0,
  ):
    super(CopNet, self).__init__()
    self.d = d
    self.n = n
    self.z_update_samples_scale = z_update_samples_scale
    self.z_update_samples = z_update_samples
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    self.nonneg = t.nn.Softplus()

    # initialize trianable parameters
    self.w = nn.Parameter(t.Tensor(w_std * t.randn(n, d)))
    self.b = nn.Parameter(t.Tensor(b_std * t.randn(n, d) + b_bias))
    self.a = nn.Parameter(t.Tensor(a_std * t.randn(n, )))

    # initialize z using normal samples
    self.update_zs()

  def update_zs(self, data=None, bp_through_z_update=False):
    if data is None:
      # generate data if not provided
      zdata = self.z_update_samples_scale * t.randn(self.z_update_samples,
                                                    self.d)
    else:
      # assuming data is samples from the hypercube, transforming with z
      with t.no_grad():
        zdata = t.stack([
            z_j(u_j) for z_j, u_j in zip(self.z, data.transpose(0, 1))
        ]).transpose(0, 1)

    if bp_through_z_update:
      g = self.g(zdata)
      self.z, self.zdot = self.z_zdot(zdata, g)
    else:
      with t.no_grad():
        g = self.g(zdata)
        self.z, self.zdot = self.z_zdot(zdata, g)

  def c_sample(self, M, R=None, n_bisect_iter=35):
    # return M x d tensor of samples from the copula using the inverse Rosenblatt formula

    # take random uniform samples if not provided:
    R = t.rand(self.d, M)
    samples = t.zeros_like(R)
    samples[0] = R[0]
    for k in range(1, self.d):
      rosenblatt_current = self.rosenblatt(k, samples[:k])
      samples[k] = utils.invert(rosenblatt_current,
                                R[k],
                                n_bisect_iter=n_bisect_iter)

    return samples.transpose(0, 1)

  def rosenblatt(self, k, us):
    # returns a function C(u_k|u_1,...,u_{k-1}) for some 1 <= k < d
    # this function takes an M-dimensional vector as input
    # us: k-1 x M tensor of conditional values for u_1 ... u_{k-1} at M sampled points

    M = us.shape[1]
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z[:k], us)])
    zdu = t.stack([zdot_j(u_j)[0] for zdot_j, u_j in zip(self.zdot[:k], us)
                   ])  # check for d > 2
    A = t.einsum('ij,jm->ijm', self.nonneg(self.w[:, :k]),
                 zu) + self.b[:, :k].unsqueeze(-1).expand(self.n, k,
                                                          M)  # n x k-1 x M
    AA = self.phidot(A) * self.nonneg(self.w[:, :k]).unsqueeze(-1).expand(
        self.n, k, M) * zdu.unsqueeze(0).expand(self.n, k, M)
    AAA = t.prod(AA, 1)  # n x M
    iZ = 1 / (t.sum(self.nonneg(self.a)))
    cond_cdf = lambda u: iZ * t.einsum(
        'i,im,im->m', self.nonneg(self.a), AAA,
        self.phi(
            self.nonneg(self.w[:, k]).unsqueeze(-1).expand(self.n, M) * self.z[
                k](u).unsqueeze(0).expand(self.n, M) + self.b[:, k].unsqueeze(
                    -1).expand(self.n, M)))  # M
    return lambda u: cond_cdf(u) / cond_cdf(t.ones(M))

  def nll(self, u):
    # compute negative log copula likelihood (per datapoint)
    # u: M x d tensor

    nll = -t.mean(self.log_copula_density(u))
    return nll

  def diag_hess(self, us):
    # returns M x d tensor of diagonal Hessian elements
    # us: M x d tensor of conditional values for u_1 ... u_{k-1} at M sampled points
    us = us.transpose(0, 1)
    M = us.shape[1]
    e = self.nonneg(self.w)
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, us)])  # d x M
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, us)])  # d x M
    p = t.einsum('ij,jm->ijm', e, zu) + self.b.unsqueeze(-1).expand(
        self.n, self.d, M)  # n x d x M
    AA = self.phidot(p) * e.unsqueeze(-1).expand(
        self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d,
                                                     M)  # n x d x M
    T = utils.dddsigmoid(p) * t.pow(e, 2).unsqueeze(-1).expand(
        self.n, self.d, M) * t.pow(zdu, 3).unsqueeze(0).expand(
            self.n, self.d, M)  # n x d x M
    diag_hess = t.zeros(M, self.d)
    for k in range(self.d):
      prod = t.prod(t.cat([AA[:, :k, :], AA[:, k + 1:, :]], 1), 1)  # n x M
      AAA = t.einsum('i,im,im->im', self.nonneg(self.a), prod, T[:, k, :])
      AAA = t.sum(AAA, 0)  # M
      diag_hess[:, k] = AAA
    return diag_hess

  def hess(self, us):
    # returns M x d x d tensor of Hessians at every datapoint
    # us: M x d tensor of conditional values for u_1 ... u_{k-1} at M sampled points
    us = us.transpose(0, 1)
    M = us.shape[1]
    e = self.nonneg(self.w)
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, us)])  # d x M
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, us)])  # d x M
    p = t.einsum('ij,jm->ijm', e, zu) + self.b.unsqueeze(-1).expand(
        self.n, self.d, M)  # n x d x M
    AA = self.phidot(p) * e.unsqueeze(-1).expand(
        self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d,
                                                     M)  # n x d x M
    T = utils.ddsigmoid(p) * t.pow(e, 2).unsqueeze(-1).expand(
        self.n, self.d, M) * t.pow(zdu, 2).unsqueeze(0).expand(
            self.n, self.d, M)  # n x d x M
    hess = t.zeros(M, self.d, self.d)
    for k in range(self.d):
      for l in range(k + 1):
        prod = t.prod(
            t.cat([AA[:, :l, :], AA[:, l + 1:k, :], AA[:, k + 1:, :]], 1),
            1)  # n x M
        AAA = t.einsum('i,im,im,im->im', self.nonneg(self.a), prod, T[:, k, :],
                       T[:, l, :])
        AAA = t.sum(AAA, 0)  # M
        hess[:, k, l] = hess[:, l, k] = AAA
    return hess

  def log_copula_density(self, u, inds=...):
    # compute log copula density
    # u: M x n_var tensor (n_var=d in the full case and 2 in the bivariate case)

    M = u.shape[0]

    if inds == ...:
      # compute the density for all variables
      n_vars = self.d
      z = self.z
      zdot = self.zdot
      e = self.nonneg(self.w)
      b = self.b
    else:
      # pick a subset of variables
      n_vars = len(inds)
      z = itemgetter(*inds)(self.z)
      zdot = itemgetter(*inds)(self.zdot)
    e = self.nonneg(self.w[:, inds])
    b = self.b[:, inds]
    assert u.shape[1] == n_vars

    zu = t.stack([z_j(u_j) for z_j, u_j in zip(z, u.transpose(0, 1))])
    zdu = t.stack(
        [zdot_j(u_j) for zdot_j, u_j in zip(zdot, u.transpose(0, 1))])
    A = t.einsum('ij,jm->ijm', e, zu) + b.unsqueeze(-1).expand(
        self.n, n_vars, M)
    AA = self.phidot(A) * e.unsqueeze(-1).expand(
        self.n, n_vars, M) * zdu.unsqueeze(0).expand(self.n, n_vars, M)
    AAA = t.prod(AA, 1)  # n x M
    log_copula_density = t.log(t.einsum(
        'i,im->m', self.nonneg(self.a), AAA)) - t.log(
            t.sum(self.nonneg(self.a))).unsqueeze(-1).expand(M)
    return log_copula_density

  def g(self, u):
    # u: M x d tensor of points
    M = u.size(0)
    A = t.einsum('ik,ka->ika', self.nonneg(self.w), u.transpose(
        0, 1)) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    return t.einsum('i,ika->ka', self.nonneg(self.a), self.phi(A)).transpose(
        0, 1) / t.sum(self.nonneg(self.a))

  def z_zdot(self, s, g):
    # returns a list of d functions z_j = g_j^{-1} by using spline interpolation
    # s: M x d tensor
    # g: M x d tensor such that s = z(g), where g : R -> [0,1]

    s = s.transpose(0, 1)
    g = g.transpose(0, 1)

    # reparameterize in terms of \tilde{z}_j: [0,1] -> [0,1], so that z = sigma^1 o \tilde{z}
    s = self.phi(s)
    # add endpoints
    endpoints = t.tensor([0, 1]).unsqueeze(0).expand(self.d, 2)
    g = t.cat((g, endpoints), dim=1)
    s = t.cat((s, endpoints), dim=1)

    # interpolate \tilde{z}_j and \dot{\tilde{z}}_j, then use the results to construct z_j and \dot{z}_j
    g, _ = t.sort(g, dim=1)
    s, _ = t.sort(s, dim=1)
    tilde_splines = [
        Interp1D(g_k, s_k, method='linear', extrap='bound')
        for g_k, s_k in zip(g, s)
    ]

    dsdg = t.div(s[:, 1:] - s[:, :-1], g[:, 1:] - g[:, :-1])
    mg = (g[:, :-1] + g[:, 1:]) / 2
    tilde_dsplines = [
        Interp1D(mg_k, ds_k, method='linear', extrap='bound')
        for mg_k, ds_k in zip(mg, dsdg)
    ]

    z = [lambda x: utils.invsigmoid(ztilde(x)) for ztilde in tilde_splines]
    zdot = [
        lambda x: utils.invsigmoiddot(ztilde(x)) * ztildedot(x)
        for ztilde, ztildedot in zip(tilde_splines, tilde_dsplines)
    ]
    return z, zdot


class SklarNet(CopNet):
  def __init__(self, d, **kwargs):
    self.d = d
    self.L_m = kwargs.pop('L_m', 4)
    self.n_m = kwargs.pop('n_m', 5)
    self.w_m_std = kwargs.pop('w_m_std', 0.1)
    self.b_m_std = kwargs.pop('b_m_std', 0)
    self.a_m_std = kwargs.pop('a_m_std', 0.1)
    self.phi_m = t.sigmoid
    self.phi_mdot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    super(SklarNet, self).__init__(d, **kwargs)

    # initialize parameters for marginal CDFs
    assert self.L_m >= 2
    self.w_ms = t.nn.ParameterList(
        [nn.Parameter(t.Tensor(self.w_m_std * t.randn(1, self.n_m, d)))])
    self.b_ms = t.nn.ParameterList(
        [nn.Parameter(t.Tensor(self.b_m_std * t.randn(1, self.n_m, d)))])
    self.a_ms = t.nn.ParameterList(
        [nn.Parameter(t.Tensor(self.a_m_std * t.randn(1, d)))])
    for _ in range(self.L_m - 2):
      self.w_ms += [
          nn.Parameter(t.Tensor(self.w_m_std * t.randn(self.n_m, self.n_m, d)))
      ]
      self.b_ms += [
          nn.Parameter(t.Tensor(self.b_m_std * t.randn(self.n_m, d)))
      ]
      self.a_ms += [
          nn.Parameter(t.Tensor(self.a_m_std * t.randn(self.n_m, d)))
      ]
    self.w_ms += [
        nn.Parameter(t.Tensor(self.w_m_std * t.randn(self.n_m, 1, d)))
    ]
    self.b_ms += [nn.Parameter(t.Tensor(self.b_m_std * t.randn(1, 1, d)))]
    self.marginal_params = [self.w_ms, self.b_ms, self.a_ms]

  def marginal_CDF(self, X, smoothing_factor=4, inds=...):
    # marginal CDF of samples
    # X : M x d tensor of sample points
    X.requires_grad = True
    F = t.unsqueeze(X, 1)

    # keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[..., inds] for w in self.w_ms]
    sliced_bs = [b[..., inds] for b in self.b_ms]
    sliced_as = [a[..., inds] for a in self.a_ms]

    # compute CDF using a feed-forward network
    for w, b, a in zip(sliced_ws, sliced_bs, sliced_as):
      F = t.einsum('mij,ikj->mkj', F, self.nonneg(w)) + b
      F = F + t.tanh(F) * t.tanh(a)

    F = t.einsum('mij,ikj->mkj', F, self.nonneg(sliced_ws[-1])) + sliced_bs[-1]
    F = self.phi_m(F / smoothing_factor)

    return t.squeeze(F)

  def marginal_likelihood(self, X, inds=...):
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)

    # compute the marginal CDF F(X)
    F = self.marginal_CDF(X, inds=inds)
    F = t.sum(F)
    f = t.autograd.grad(F, X, create_graph=True)[0]
    return f

  def log_density(self, X, inds=...):
    # full density (copula + marginals)
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)
    F = self.marginal_CDF(X, inds=inds)
    log_copula_density = self.log_copula_density(F, inds=inds)
    log_marginal_density = t.sum(t.log(self.marginal_likelihood(X, inds=inds)),
                                 dim=1)
    log_density = log_copula_density + log_marginal_density
    return log_density

  def nll(self, X):
    # negative log likelihood (copula + marginals, average over datapoints)
    # X : M x d tensor of sample points

    X = t.Tensor(X)
    M = X.shape[0]
    F = self.marginal_CDF(X)
    return - t.sum(t.log(self.marginal_likelihood(X))) / M \
           + super(SklarNet, self).nll(F)

  def sample(self, M, n_bisect_iter=35):
    # return M x d tensor of samples from the full density

    copula_samples = self.c_sample(M)
    # samples = utils.invert(self.marginal_CDF,
    #                        copula_samples,
    #                        n_bisect_iter=n_bisect_iter)
    raise NotImplementedError
