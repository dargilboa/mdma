import torch as t
import torch.nn as nn
import numpy as np
from cdfnet import utils
from operator import itemgetter
from opt_einsum import contract, contract_expression
from numpy import concatenate
from torch.nn import AvgPool1d, AvgPool2d

einsum = t.einsum
# einsum = contract


class CDFNet(nn.Module):
  def __init__(
      self,
      d,
      n=100,
      L=4,
      m=5,
      w_std=0.1,
      b_bias=0,
      b_std=0,
      a_std=1.0,
      HT_poolsize=2,
      use_HT=False,
      adaptive_coupling=False,
  ):
    super(CDFNet, self).__init__()
    self.d = d
    self.n = n
    self.m = m
    self.L = L
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    self.nonneg = t.nn.Softplus(10)
    self.nonneg_m = t.nn.Softplus(10)  # try beta=1?
    self.w_std = w_std
    self.a_std = a_std
    self.b_std = b_std
    self.b_bias = b_bias
    #self.phidot_expr = [[None, None]] * self.L
    self.use_HT = use_HT
    self.adaptive_coupling = adaptive_coupling

    # initialize parameters for marginal CDFs
    assert self.L >= 2
    w_scale = self.w_std / t.sqrt(t.Tensor([self.m]))
    self.w_s = t.nn.ParameterList(
        [nn.Parameter(self.w_std * t.randn(self.d, self.n, 1, self.m))])
    self.b_s = t.nn.ParameterList(
        [nn.Parameter(self.b_std * t.randn(self.d, self.n, self.m))])
    self.a_s = t.nn.ParameterList(
        [nn.Parameter(self.a_std * t.randn(self.d, self.n, self.m))])
    for _ in range(self.L - 2):
      self.w_s += [
          nn.Parameter(w_scale * t.randn(self.d, self.n, self.m, self.m))
      ]
      self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.n, self.m))]
      self.a_s += [nn.Parameter(self.a_std * t.randn(self.d, self.n, self.m))]

    self.w_s += [nn.Parameter(w_scale * t.randn(self.d, self.n, self.m, 1))]
    self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.n, 1))]
    self.a_s += [nn.Parameter(self.a_std * t.randn(self.n))]

    # HT parameters
    if self.use_HT:
      self.HT_poolsize = HT_poolsize
      self.a_HT_std = .1
      a_scale = self.a_HT_std / np.sqrt(self.n)
      self.L_HT = int(np.ceil(np.log(self.d) / np.log(self.HT_poolsize)))
      self.a_HTs = t.nn.ParameterList()
      dim_l = self.d
      for _ in range(self.L_HT - 1):
        dim_l = int(np.ceil(dim_l / self.HT_poolsize))
        # at layer l we require dim_l * n * n parameters
        # self.a_HTs += [nn.Parameter(a_scale * t.randn(dim_l, self.n, self.n))]
        self.a_HTs += [
            nn.Parameter(self.n * t.eye(self.n).repeat(dim_l, 1, 1))
        ]
      self.a_HTs += [nn.Parameter(a_scale * t.randn(1, self.n, 1))]

      # pooling layer for stabilizing nll
      self.pool_layer = AvgPool1d(kernel_size=self.HT_poolsize,
                                  stride=self.HT_poolsize,
                                  ceil_mode=True)

      # create couplings
      self.all_couplings = self.create_default_couplings()

  def phis(self, X, inds=...):
    # X is a M x dim(inds) tensor
    # returns M x n x dim(inds) tensor
    phis = self.expand_X(X)

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
    return t.squeeze(phis, -1)

  # def phidots(self, X, inds=...):
  #   # X is a M x dim(inds) tensor
  #   # returns M x n tensor
  #
  #   phidots, fm = self.phidots(X, inds=inds)
  #   phidots = t.prod(t.exp(fm).unsqueeze(1).unsqueeze(1) * phidots,
  #                    1)  # prod over d
  #   return t.squeeze(phidots), fm.squeeze()  # M x n
  #
  # def phidots_opt_einsum(self, X, inds=...):
  #   # X is a M x dim(inds) tensor
  #   # returns M x n tensor
  #   phis = self.expand_X(X)  # M x dim(inds) x n x 1
  #
  #   # keep only the parameters relevant for the subset of variables specified by inds and apply the appropriate transformations
  #   sliced_ws = [self.nonneg_m(w[inds, ...]) for w in self.w_s]
  #   sliced_bs = [b[inds, ...] for b in self.b_s]
  #   sliced_as = [t.tanh(a[inds, ...]) for a in self.a_s]
  #
  #   compute_expr = False
  #   if all(concatenate(self.phidot_expr) == None):  # for the first evaluation
  #     compute_expr = True
  #
  #   # compute CDF using a feed-forward network
  #   phidots = t.ones_like(phis)
  #   for ix, (w, b, a) in enumerate(
  #       zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as[:-1])):
  #     if compute_expr:
  #       self.phidot_expr[ix][0] = contract_expression('...ik,...ikl->...il',
  #                                                     phis.shape, w.shape)
  #     phis = self.phidot_expr[ix][0](phis, w) + b
  #
  #     if compute_expr == True:
  #       self.phidot_expr[ix][1] = contract_expression(
  #           '...ik,...ikl,...il->...il', phidots.shape, w.shape, phis.shape)
  #     phidots = self.phidot_expr[ix][1](phidots, w,
  #                                       (1 + a * utils.tanhdot(phis)))
  #
  #     phis = phis + t.tanh(phis) * a
  #
  #   if compute_expr == True:
  #     self.phidot_expr[-1][0] = contract_expression('...ik,...ikl->...il',
  #                                                   phis.shape,
  #                                                   sliced_ws[-1].shape)
  #   phis = self.phidot_expr[-1][0](phis, sliced_ws[-1]) + sliced_bs[-1]
  #   if compute_expr == True:
  #     self.phidot_expr[-1][1] = contract_expression(
  #         '...ik,...ikl,...il->...il', phidots.shape, sliced_ws[-1].shape,
  #         phis.shape)
  #     # pdb.set_trace()
  #   phidots = self.phidot_expr[-1][1](phidots, sliced_ws[-1],
  #                                     utils.sigmoiddot(phis))
  #
  #   fm = -t.log(phidots + 1e-10).detach()
  #   fm = fm.mean(1, True)  # mean over d
  #   fm = fm.min(2, True)[0]  # min over n
  #
  #   phidots = t.prod(t.exp(fm) * phidots, 1)  # prod over d
  #   return t.squeeze(phidots), fm.squeeze()  # M x n

  def phidots(self, X, inds=...):
    # X is a M x dim(inds) tensor
    # returns M x d x n tensor
    phis = self.expand_X(X)  # M x dim(inds) x n x 1

    # keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[inds, ...] for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [a[inds, ...] for a in self.a_s]

    # compute CDF using a feed-forward network
    phidots = t.ones_like(phis)
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as[:-1]):
      phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(w)) + b
      phidots = t.einsum('mjik,jikl,mjil->mjil', phidots, self.nonneg_m(w),
                         (1 + t.tanh(a) * utils.tanhdot(phis)))
      phis = phis + t.tanh(phis) * t.tanh(a)
    phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(
        sliced_ws[-1])) + sliced_bs[-1]
    phidots = t.einsum('mjik,jikl,mjil->mjil', phidots,
                       self.nonneg_m(sliced_ws[-1]), utils.sigmoiddot(phis))

    return t.squeeze(phidots, -1)

  def get_stabilizer(self, phidots, eps=1e-40):
    if self.use_HT:
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.transpose(1, 2).squeeze(-1)  # put d last
      fm = self.pool_layer(fm)  # mean over pooled variables
      fm = fm.min(1, True)[0]  # min over n, fm is M x d / p x 1
      fm = fm.mean(2, True)  # mean over d / p, f is M x 1 x 1
    else:
      # assuming CP
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.mean(1, True)  # mean over d
      fm = fm.min(2, True)[0]  # min over n (fm is M dimensional)
    fm = fm.squeeze()
    return fm

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

    phis = self.phis(X, inds)
    F = self.contract(phis, inds)
    return F

  def contract(self, T, inds=...):
    if self.use_HT:
      return self.HT_contraction(T, inds=inds)
    else:
      # assuming CP
      return self.CP_contraction(T)

  def likelihood(self, X, inds=..., stabilize=False):
    # Evaluate joint likelihood at X
    # X : M x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)

    phidots = self.phidots(X, inds)
    fm = None
    # phidots is M x d x n

    # stabilize if required
    if stabilize:
      fm = self.get_stabilizer(phidots)
      phidots = t.exp(fm).unsqueeze(-1).unsqueeze(-1) * phidots

    lk = self.contract(phidots, inds)
    return lk, fm

  def CP_contraction(self, T):
    # T is a M x len(inds) x n tensor
    T = t.prod(T, 1)  # prod over inds
    a = self.nonneg(self.a_s[-1])
    return einsum('mi,i->m', T, a / a.sum())

  def HT_contraction(self, T, inds=...):
    # T is a M x len(inds) x n tensor (some combination of phis and phidots) that is contracted with a
    # tensor of parameters to obtain a scalar
    # returns an M dimensional tensor

    # add ones for marginalized variables (can be avoided to speed up and save memory)
    if inds is not ...:
      T_full = t.ones((T.shape[0], self.d, self.n))
      T_full[:, inds, :] = T
      T = T_full

    for a_s, couplings in zip(self.a_HTs, self.all_couplings):

      T = [t.prod(T[:, coupling, :], dim=1) for coupling in couplings]
      # normalize sum of a_s across second dimension
      a_s = self.nonneg(a_s)
      a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
      T = t.stack([t.matmul(phid, a) for phid, a in zip(T, a_s)], dim=1)

    return t.squeeze(T)

  def marginal_likelihood(self, X):
    marg_l = t.prod(t.stack(
        [self.likelihood(X[:, i], inds=[i])[0] for i in range(self.d)]),
                    dim=0)
    return marg_l

  def marginal_nll(self, X):
    log_marginal_density = t.log(self.marginal_likelihood(X))
    return -t.mean(log_marginal_density)

  def log_density(self, X, inds=..., eps=1e-40, stabilize=False):
    lk, fm = self.likelihood(X, inds, stabilize=stabilize)
    if stabilize:
      if inds is ...:
        n_vars = self.d
      else:
        n_vars = len(inds)
      return t.log(lk + eps) - n_vars * fm
    else:
      return t.log(lk + eps)

  def nll(self, X, inds=..., stabilize=False):
    # negative log likelihood
    # X : M x d tensor of sample points
    return -t.mean(self.log_density(X, inds=inds, stabilize=stabilize))

  def sample(self,
             S,
             batch_size=None,
             inds=None,
             n_bisect_iter=35,
             upper_bound=1e3,
             lower_bound=-1e3):
    # if inds is not specified, sample all variables
    if batch_size is None:
      batch_size = S

    all_samples = []
    with t.no_grad():
      if inds is None:
        inds = range(self.d)

      dim_samples = len(inds)
      U = t.rand(S, dim_samples)
      U_dataloader = t.utils.data.DataLoader(U,
                                             batch_size=batch_size,
                                             shuffle=False)
      for batch_idx, batch in enumerate(U_dataloader):
        samples = t.zeros((batch_size, self.d))
        for k in range(dim_samples):
          # can this be done in log space for stability?
          curr_condCDF = self.condCDF(k, samples[:, :k], inds)
          samples[:, k] = utils.invert(curr_condCDF,
                                       batch[:, k],
                                       n_bisect_iter=n_bisect_iter,
                                       ub=upper_bound,
                                       lb=lower_bound)

        all_samples.append(samples.cpu().detach().numpy())
      return np.concatenate(all_samples)

  def condCDF(self, k, prev_samples, inds):
    # compute the conditional CDF F(x_{inds[k]}|x_{inds[0]},...,x_{inds[k-1]})
    # prev_samples: S x (k-1) tensor of variables to condition over
    # returns a function R^S -> [0,1]^S

    if k == 0:
      # there is nothing to condition on
      phidots = 1
      denom = 1
    else:
      phidots = self.phidots(prev_samples, inds=inds[:k])
      denom = self.contract(phidots, inds=inds[:k])

    def curr_condCDF(u):
      phis = self.phis(u, inds=[inds[k]])
      if k > 0:
        phiphidots = t.cat([phidots, phis], dim=1)
      else:
        # there are no phidots if k == 0
        phiphidots = phis
      CCDF = self.contract(phiphidots, inds=inds[:k + 1])
      return CCDF / denom

    return curr_condCDF

  def cond_density(self, X, inds, cond_X, cond_inds):
    # compute the conditional density of the variables in inds conditioned on those in cond_inds
    # X: S x len(inds) tensor of points at which to evaluate the density
    # cond_X: S x len(cond_inds) tensor of values to condition on
    # returns a length S tensor of densities

    cond_phidots = self.phidots(cond_X, inds=cond_inds)
    denom = self.contract(cond_phidots, inds=cond_inds)
    joint_X = t.cat([X, cond_X], dim=1)
    joint_inds = inds + cond_inds
    phidots = self.phidots(joint_X, inds=joint_inds)
    numerator = self.contract(phidots, inds=joint_inds)
    return numerator / denom

  def create_default_couplings(self):
    # create default coupling of variables used in the HT decomposition
    all_couplings = []
    dim_l = self.d
    for _ in range(self.L_HT):
      coupling_ranges = [(j, min(dim_l, j + self.HT_poolsize))
                         for j in range(0, dim_l, self.HT_poolsize)]
      couplings = [list(range(rng[0], rng[1])) for rng in coupling_ranges]
      dim_l = int(np.ceil(dim_l / self.HT_poolsize))
      all_couplings.append(couplings)
    return all_couplings

  def create_adaptive_couplings(self, X):
    # Assumes the data is standardized
    all_couplings = []
    if self.HT_poolsize > 2:
      raise ValueError("adaptive coupling not supported with HT_poolsize > 2")

    with t.no_grad():
      pool_layer = AvgPool2d(kernel_size=self.HT_poolsize,
                             stride=self.HT_poolsize,
                             ceil_mode=True)
      Sigma = t.matmul(X.transpose(0, 1), X)
      dim_l = self.d
      for _ in range(self.L_HT):
        # add couples at layer l
        couplings = []
        coupled_inds = []
        n_couples = int(np.ceil(dim_l / self.HT_poolsize))

        # sort correlation matrix if this isn't the last layer
        if n_couples > 1:
          sorted_flat_corrs = t.sort(Sigma.flatten(), descending=True)
          sorted_flat_inds = sorted_flat_corrs.indices[::
                                                       2]  # discard symmetric entries
          sorted_inds = [[ind // dim_l, ind % dim_l]
                         for ind in sorted_flat_inds]

        for _ in range(n_couples):
          if dim_l - len(coupled_inds) <= self.HT_poolsize:
            # we added all the couples, there are <= HT_poolsize variables left
            remainder = []
            for i in range(dim_l):
              if i not in coupled_inds:
                remainder.append(t.tensor(i))
                coupled_inds.append(i)
            couplings.append(remainder)
          else:
            # find the next couple to add
            while sorted_inds[0][0] in coupled_inds \
                or sorted_inds[0][1] in coupled_inds \
                or sorted_inds[0][0] == sorted_inds[0][1]:
              # one of the variables is already coupled or this is diagonal entry, pop
              sorted_inds.pop(0)
            # a new couple to add has been found
            couplings.append(sorted_inds[0])
            coupled_inds.append(sorted_inds[0][0].item())
            coupled_inds.append(sorted_inds[0][1].item())

        # coarse-grain Sigma and store
        perm_Sigma = Sigma[coupled_inds][:, coupled_inds]
        Sigma = pool_layer(perm_Sigma.unsqueeze(0)).squeeze()
        dim_l = n_couples
        all_couplings.append(couplings)

    self.all_couplings = all_couplings
