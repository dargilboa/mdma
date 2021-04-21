import torch as t
import torch.nn as nn
from cdfnet import utils
from operator import itemgetter
from opt_einsum import contract, contract_expression
from numpy import concatenate

einsum = t.einsum
# einsum = contract


class CP1Net(nn.Module):
  def __init__(self,
               d,
               n=100,
               r=100,
               l=4,
               m=5,
               w_std=0.1,
               b_bias=0,
               b_std=0.1,
               a_std=1.0):
    super(CP1Net, self).__init__()
    self.d = d
    self.n = n
    self.r = r
    self.m = m
    self.l = l
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    self.nonneg = t.nn.Softplus()
    self.nonneg_m = t.nn.Softplus(10)  # try beta=1?
    self.w_std = w_std
    self.a_std = a_std
    self.b_std = b_std
    self.b_bias = b_bias
    self.phidot_expr = [[None, None] for w in range(0, l)]

    # initialize parameters for marginal CDFs
    assert self.l >= 2
    w_scale = self.w_std / t.sqrt(t.Tensor([self.m]))
    self.w_s = t.nn.ParameterList(
        [nn.Parameter(self.w_std * t.randn(self.r, self.m, 1))])
    self.b_s = t.nn.ParameterList(
        [nn.Parameter(self.b_std * t.randn(self.r, self.m))])
    self.a_s = t.nn.ParameterList(
        [nn.Parameter(self.a_std * t.randn(self.r, self.m))])
    for i in range(self.l - 1):
      self.w_s += [nn.Parameter(w_scale * t.randn(self.r, self.m, self.m))]
      self.b_s += [nn.Parameter(self.b_std * t.randn(self.r, self.m))]
      self.a_s += [nn.Parameter(self.a_std * t.randn(self.r, self.m))]

    self.w_s += [nn.Parameter(w_scale * t.randn(self.r, self.m))]
    self.b_s += [nn.Parameter(self.b_std * t.randn(self.r))]
    # for the a^{z,i}
    self.a_s += [nn.Parameter(self.a_std * t.randn(self.n, self.d, self.r))]
    # for the a_z
    self.a_s += [nn.Parameter(self.a_std * t.randn(self.n))]

  def phidots(self, X, inds=...):
    # X is a m x dim(inds) tensor
    # returns m x dim(inds) x r tensor
    phis = self.expand_X(X)  # m x dim(inds) x r x 1

    phidots = t.ones_like(phis)
    for w, b, a in zip(self.w_s[:-1], self.b_s[:-1], self.a_s[:-2]):
      w = self.nonneg_m(w)
      a = t.tanh(a)
      phis = t.einsum('ijlk,lmk->ijlm', phis, w) + b
      phidots = t.einsum('ijlk,lmk->ijlm', phidots,
                         w) * (1 + a * utils.tanhdot(phis))

      phis = phis + a * t.tanh(phis)

    w = self.nonneg_m(self.w_s[-1])
    phis = t.einsum('ijlk,lk->ijl', phis, w) + self.b_s[-1]
    phidots = t.einsum('ijlk,lk->ijl', phidots, w) * utils.sigmoiddot(phis)

    fm = -t.log(phidots + 1e-10).detach()
    fm = fm.mean(1, True)  # mean over d
    fm = fm.min(2, True)[0]  # min over r

    return phidots, fm.squeeze()  # m x dim(inds) x r x 1

  def expand_X(self, X):
    if len(X.shape) == 1:
      X = X.unsqueeze(-1)
    X = X.unsqueeze(-1).unsqueeze(-1)
    Xr = X.expand(-1, -1, self.r, 1)  # m x d x r x 1
    return Xr

  def likelihood(self, X, inds=...):
    # Evaluate joint likelihood at X
    # X : m x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)

    phidots, fm = self.phidots(X, inds)

    # those are the a^{z,i} from the paper
    a = self.nonneg(self.a_s[-2])[:, inds, :]
    a = a / a.sum(1, keepdims=True)  # normalize for each d

    # tensor x representation functions
    phidots = phidots.unsqueeze(1).expand(-1, self.n, -1, -1) * a
    # this is the a_z from the paper
    a = self.nonneg(self.a_s[-1])
    # sum over representation functions, prod over d, sum over nodes
    f = einsum('mi,i->m', phidots.sum(3).prod(2), a / a.sum())
    return f, fm

  def log_density(self, X, inds=...):
    lk, fm = self.likelihood(X, inds)
    if inds == ...:
      n_vars = self.d
    else:
      n_vars = len(inds)
    return t.log(lk + 1e-10)

  def nll(self, X, inds=...):
    # negative log likelihood
    # X : m x d tensor of sample points
    return -t.mean(self.log_density(X, inds=inds))


class CDFNet(nn.Module):
  def __init__(self,
               d,
               n=100,
               l=4,
               m=5,
               w_std=0.1,
               b_bias=0,
               b_std=0.1,
               a_std=1.0):
    super(CDFNet, self).__init__()
    self.d = d
    self.n = n
    self.m = m
    self.l = l
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    self.nonneg = t.nn.Softplus()
    self.nonneg_m = t.nn.Softplus(10)  # try beta=1?
    self.w_std = w_std
    self.a_std = a_std
    self.b_std = b_std
    self.b_bias = b_bias
    self.phidot_expr = [[None, None] for w in range(0, l)]

    # initialize parameters for marginal CDFs
    assert self.l >= 2
    w_scale = self.w_std / t.sqrt(t.Tensor([self.m]))
    self.w_s = t.nn.ParameterList(
        [nn.Parameter(self.w_std * t.randn(self.d, self.n, 1, self.m))])
    self.b_s = t.nn.ParameterList(
        [nn.Parameter(self.b_std * t.randn(self.d, self.n, self.m))])
    self.a_s = t.nn.ParameterList(
        [nn.Parameter(self.a_std * t.randn(self.d, self.n, self.m))])
    for _ in range(self.l - 2):
      self.w_s += [
          nn.Parameter(w_scale * t.randn(self.d, self.n, self.m, self.m))
      ]
      self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.n, self.m))]
      self.a_s += [nn.Parameter(self.a_std * t.randn(self.d, self.n, self.m))]

    self.w_s += [nn.Parameter(w_scale * t.randn(self.d, self.n, self.m, 1))]
    self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.n, 1))]
    self.a_s += [nn.Parameter(self.a_std * t.randn(self.n))]

  def phis(self, Xn, inds=...):
    # Xn is a M x 1 x n x dim(inds) tensor
    # returns M x n x dim(inds) tensor
    phis = Xn  # cleaner to move expand_X inside and take M x dim(inds) as input, like phidots

    # keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[inds, ...] for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [a[inds, ...] for a in self.a_s]

    # compute CDF using a feed-forward network
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as[:-1]):
      phis = einsum('mjik,jikl->mjil', phis, self.nonneg_m(w)) + b
      phis = phis + t.tanh(phis) * t.tanh(a)

    phis = einsum('mjik,jikl->mjil', phis, self.nonneg_m(
        sliced_ws[-1])) + sliced_bs[-1]

    phis = self.phi(phis)
    return t.squeeze(phis)

  def phidots(self, X, inds=...):
    # X is a M x dim(inds) tensor
    # returns M x n tensor
    phis = self.expand_X(X)  # M x dim(inds) x n x 1

    # keep only the parameters relevant for the subset of variables specified by inds and apply the appropriate transformations
    sliced_ws = [self.nonneg_m(w[inds, ...]) for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [t.tanh(a[inds, ...]) for a in self.a_s]

    compute_expr = False
    if all(concatenate(self.phidot_expr) == None):  # for the first evaluation
      compute_expr = True

    # compute CDF using a feed-forward network
    phidots = t.ones_like(phis)
    for w, b, a, ix in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as[:-1],
                           range(0, len(sliced_ws[:-1]))):

      if compute_expr:
        self.phidot_expr[ix][0] = contract_expression('...ik,...ikl->...il',
                                                      phis.shape, w.shape)
      phis = self.phidot_expr[ix][0](phis, w) + b

      if compute_expr == True:
        self.phidot_expr[ix][1] = contract_expression(
            '...ik,...ikl,...il->...il', phidots.shape, w.shape, phis.shape)
      phidots = self.phidot_expr[ix][1](phidots, w,
                                        (1 + a * utils.tanhdot(phis)))

      phis = phis + t.tanh(phis) * a

    if compute_expr == True:
      self.phidot_expr[-1][0] = contract_expression('...ik,...ikl->...il',
                                                    phis.shape,
                                                    sliced_ws[-1].shape)
    phis = self.phidot_expr[-1][0](phis, sliced_ws[-1]) + sliced_bs[-1]
    if compute_expr == True:
      self.phidot_expr[-1][1] = contract_expression(
          '...ik,...ikl,...il->...il', phidots.shape, sliced_ws[-1].shape,
          phis.shape)
      # pdb.set_trace()
    phidots = self.phidot_expr[-1][1](phidots, sliced_ws[-1],
                                      utils.sigmoiddot(phis))

    fm = -t.log(phidots + 1e-10).detach()
    fm = fm.mean(1, True)  # mean over d
    fm = fm.min(2, True)[0]  # min over n

    phidots = t.prod(t.exp(fm) * phidots, 1)  # prod over d
    return t.squeeze(phidots), fm.squeeze()  # M x n

  def expand_X(self, X):
    if len(X.shape) == 1:
      X = X.unsqueeze(-1)
    X = X.unsqueeze(-1).unsqueeze(-1)
    Xn = X.expand(-1, -1, self.n, 1)  # M x d x n x 1
    return Xn

  def CDF(self, X, inds=...):
    # Evaluate joint CDF at X
    # X : M x len(inds) tensor of sample points
    # returns M dim tensor

    Xn = self.expand_X(X)
    phis = self.phis(Xn, inds)
    phis = t.prod(phis, 1)
    a = self.nonneg(self.a_s[-1])
    F = einsum('mi,i->m', phis, a / a.sum())
    return F

  def likelihood(self, X, inds=...):
    # Evaluate joint likelihood at X
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)

    phidots, fm = self.phidots(X, inds)
    a = self.nonneg(self.a_s[-1])
    f = einsum('mi,i->m', phidots, a / a.sum())
    return f, fm

  def marginal_likelihood(self, X):
    marg_l = t.prod(t.stack(
        [self.likelihood(X[:, i], inds=[i]) for i in range(self.d)]),
                    dim=0)
    return marg_l

  def marginal_nll(self, X):
    log_marginal_density = t.log(self.marginal_likelihood(X))
    return -t.mean(log_marginal_density)

  def log_density(self, X, inds=...):
    lk, fm = self.likelihood(X, inds)
    if inds == ...:
      n_vars = self.d
    else:
      n_vars = len(inds)
    return t.log(lk + 1e-10) - fm * n_vars

  def nll(self, X, inds=...):
    # negative log likelihood
    # X : M x d tensor of sample points
    return -t.mean(self.log_density(X, inds=inds))

  def sample(self,
             S,
             inds=None,
             n_bisect_iter=35,
             upper_bound=1e3,
             lower_bound=-1e3):
    # if inds is not specified, sample all variables
    if inds is None:
      inds = range(self.d)

    dim_samples = len(inds)
    U = t.rand(S, dim_samples)

    samples = t.zeros_like(U)
    for k in range(dim_samples):
      curr_condCDF = self.condCDF(k, samples[:, :k], inds)
      samples[:, k] = utils.invert(curr_condCDF,
                                   U[:, k],
                                   n_bisect_iter=n_bisect_iter,
                                   ub=upper_bound,
                                   lb=lower_bound)

    return samples.cpu().detach().numpy()

  def condCDF(self, k, prev_samples, inds):
    # compute the conditional CDF F(x_{inds[k]}|x_{inds[0]},...,x_{inds[k-1]})
    # prev_samples: S x (k-1) tensor of variables to condition over
    # returns a function R^S -> [0,1]^S

    a = self.nonneg(self.a_s[-1])

    if k == 0:
      # there is nothing to condition on
      phidots = 1
      denom = 1
    else:
      phidots = self.phidots(prev_samples, inds=inds[:k])
      denom = einsum('mi,i->m', phidots, a / a.sum())

    def curr_condCDF(u):
      un = self.expand_X(u)
      phis = self.phis(un, inds=[inds[k]])
      phis = t.prod(phis, 1)
      prod = phis * phidots
      CCDF = einsum('mi,i->m', prod, a / a.sum())
      return CCDF / denom

    return curr_condCDF


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
    a_scale = a_std / t.sqrt(t.Tensor([n]))
    self.w = nn.Parameter(t.Tensor(w_std * t.randn(n, d)))
    self.b = nn.Parameter(t.Tensor(b_std * t.randn(n, d) + b_bias))
    self.a = nn.Parameter(t.Tensor(a_scale * t.randn(n, )))
    self.copula_params = [self.w, self.b, self.a]

    # initialize z using normal samples
    self.update_zs()

  def update_zs(self,
                data=None,
                bp_through_z_update=False,
                fit_marginals=True):
    if data is None:
      # generate data if not provided
      zdata = self.z_update_samples_scale * t.randn(self.z_update_samples,
                                                    self.d)
    else:
      if fit_marginals:
        # the data is sampled from a full density
        zdata = data
      else:
        # the data is from the hypercube, transforming with z first
        with t.no_grad():
          zdata = t.stack([
              z_j(u_j) for z_j, u_j in zip(self.z, data.transpose(0, 1))
          ]).transpose(0, 1)

      # adding noise
      zdata = zdata + t.randn_like(data) * 0.01 * t.std(data)

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
    A = einsum('ij,jm->ijm', self.nonneg(self.w[:, :k]),
               zu) + self.b[:, :k].unsqueeze(-1).expand(self.n, k,
                                                        M)  # n x k-1 x M
    AA = self.phidot(A) * self.nonneg(self.w[:, :k]).unsqueeze(-1).expand(
        self.n, k, M) * zdu.unsqueeze(0).expand(self.n, k, M)
    AAA = t.prod(AA, 1)  # n x M
    iZ = 1 / (t.sum(self.nonneg(self.a)))
    cond_cdf = lambda u: iZ * einsum(
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
    p = einsum('ij,jm->ijm', e, zu) + self.b.unsqueeze(-1).expand(
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
      AAA = einsum('i,im,im->im', self.nonneg(self.a), prod, T[:, k, :])
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
    p = einsum('ij,jm->ijm', e, zu) + self.b.unsqueeze(-1).expand(
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
        AAA = einsum('i,im,im,im->im', self.nonneg(self.a), prod, T[:, k, :],
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
    A = einsum('ij,jm->ijm', e, zu) + b.unsqueeze(-1).expand(self.n, n_vars, M)
    A = self.phidot(A) * e.unsqueeze(-1).expand(
        self.n, n_vars, M) * zdu.unsqueeze(0).expand(self.n, n_vars, M)
    AA = t.prod(A, 1)  # n x M
    log_copula_density = t.log(
        einsum('i,im->m', self.nonneg(self.a), AA)) - t.log(
            t.sum(self.nonneg(self.a))).unsqueeze(-1).expand(M)
    return log_copula_density

  def g(self, u):
    # u: M x d tensor of points
    M = u.size(0)
    A = einsum('ik,ka->ika', self.nonneg(self.w), u.transpose(
        0, 1)) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    return einsum('i,ika->ka', self.nonneg(self.a), self.phi(A)).transpose(
        0, 1) / t.sum(self.nonneg(self.a))

  def z_zdot(self, s, g):
    # returns a list of d functions z_j \approx g_j^{-1} by using linear interpolation
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
    tilde_splines = [utils.linear_interp(g_k, s_k) for g_k, s_k in zip(g, s)]

    dsdg = t.div(s[:, 1:] - s[:, :-1], g[:, 1:] - g[:, :-1])
    mg = (g[:, :-1] + g[:, 1:]) / 2
    tilde_dsplines = [
        utils.linear_interp(mg_k, ds_k) for mg_k, ds_k in zip(mg, dsdg)
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
    self.w_m_bias = kwargs.pop('w_m_bias', 0)
    self.b_m_std = kwargs.pop('b_m_std', 0)
    self.a_m_std = kwargs.pop('a_m_std', 0.1)
    self.marginal_smoothing_factor = kwargs.pop('marginal_smoothing_factor', 4)
    super(SklarNet, self).__init__(d, **kwargs)
    self.nonneg_m = t.nn.Softplus(10)
    self.phi_m = t.sigmoid
    self.phi_mdot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))

    # initialize parameters for marginal CDFs
    assert self.L_m >= 2
    w_scale = self.w_m_std / t.sqrt(t.Tensor([self.n_m]))
    self.w_ms = t.nn.ParameterList([
        nn.Parameter(
            t.Tensor(self.w_m_std * t.randn(1, self.n_m, d) + self.w_m_bias))
    ])
    self.b_ms = t.nn.ParameterList(
        [nn.Parameter(t.Tensor(self.b_m_std * t.randn(1, self.n_m, d)))])
    self.a_ms = t.nn.ParameterList(
        [nn.Parameter(t.Tensor(self.a_m_std * t.randn(1, self.n_m, d)))])
    for _ in range(self.L_m - 2):
      self.w_ms += [
          nn.Parameter(
              t.Tensor(w_scale * t.randn(self.n_m, self.n_m, d) +
                       self.w_m_bias))
      ]
      self.b_ms += [
          nn.Parameter(t.Tensor(self.b_m_std * t.randn(self.n_m, d)))
      ]
      self.a_ms += [
          nn.Parameter(t.Tensor(self.a_m_std * t.randn(self.n_m, d)))
      ]
    self.w_ms += [
        nn.Parameter(
            t.Tensor(w_scale * t.randn(self.n_m, 1, d) + self.w_m_bias))
    ]
    self.b_ms += [nn.Parameter(t.Tensor(self.b_m_std * t.randn(1, 1, d)))]
    self.marginal_params = [self.w_ms, self.b_ms, self.a_ms]
    self.sf = t.Tensor([self.marginal_smoothing_factor] * self.d)

  def set_marginal_scales(self, data):
    # scale the marginals by the std of the marginal data
    self.sf = t.std(t.Tensor(data), dim=0)

  def marginal_CDF(self, X, inds=...):
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
      F = einsum('mij,ikj->mkj', F, self.nonneg_m(w)) + b
      F = F + t.tanh(F) * t.tanh(a)

    F = einsum('mij,ikj->mkj', F, self.nonneg_m(sliced_ws[-1])) + sliced_bs[-1]
    F = self.phi_m(F)

    return t.squeeze(F)

  def marginal_likelihood(self, X, inds=..., eps=1e-15):
    # compute all marginal likelihoods f(X)
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)
    with t.enable_grad():
      F = self.marginal_CDF(X, inds=inds)
      F = t.sum(F)
      f = t.autograd.grad(F, X, create_graph=True)[0] + eps
    return f

  def log_marginal_density(self, X, inds=...):
    log_marginal_density = t.sum(t.log(self.marginal_likelihood(X, inds=inds)),
                                 dim=1)
    return log_marginal_density

  def log_density(self, X, inds=...):
    # full density (copula + marginals)
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)
    # returns M dimensional vector of log densities
    F = self.marginal_CDF(X, inds=inds)
    log_copula_density = self.log_copula_density(F, inds=inds)
    log_marginal_density = t.sum(t.log(self.marginal_likelihood(X, inds=inds)),
                                 dim=1)
    log_density = log_copula_density + log_marginal_density
    return log_density

  def nll(self, X):
    # negative log likelihood (copula + marginals, average over datapoints)
    # X : M x d tensor of sample points

    return -t.mean(self.log_density(X))

  def marginal_nll(self, X):
    # negative log marginal likelihood (copula + marginals, average over datapoints)
    # X : M x d tensor of sample points

    return -t.mean(self.log_marginal_density(X))

  def sample(self, M, n_bisect_iter=35):
    # return M x d tensor of samples from the full density

    copula_samples = self.c_sample(M)
    # samples = utils.invert(self.marginal_CDF,
    #                        copula_samples,
    #                        n_bisect_iter=n_bisect_iter)
    raise NotImplementedError