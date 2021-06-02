import torch as t
import torch.nn as nn
import numpy as np
from MDMA import utils
from torch.nn import AvgPool1d, AvgPool2d


class MDMA(nn.Module):
  def __init__(
      self,
      d,
      m=100,
      l=2,
      r=3,
      w_std=1.0,
      b_bias=0,
      b_std=0,
      a_std=1.0,
      HT_poolsize=2,
      use_HT=True,
      use_MERA=False,
      adaptive_coupling=False,
      random_coupling=True,
      mix_vars=False,
      n_mix_terms=1,
  ):
    super(MDMA, self).__init__()
    self.d = d
    self.m = m
    self.r = r
    self.l = l
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))
    self.nonneg = t.nn.Softplus(10)
    self.nonneg_m = t.nn.Softplus(10)
    self.w_std = w_std
    self.a_std = a_std
    self.b_std = b_std
    self.b_bias = b_bias
    self.use_HT = use_HT
    self.use_MERA = use_MERA
    self.adaptive_coupling = adaptive_coupling
    self.random_coupling = random_coupling

    # initialize parameters for marginal CDFs
    assert self.L >= 2
    w_scale = self.w_std / t.sqrt(t.Tensor([self.r]))
    self.w_s = t.nn.ParameterList(
        [nn.Parameter(self.w_std * t.randn(self.d, self.m, 1, self.r))])
    self.b_s = t.nn.ParameterList(
        [nn.Parameter(self.b_std * t.randn(self.d, self.m, self.r))])
    self.a_s = t.nn.ParameterList(
        [nn.Parameter(self.a_std * t.randn(self.d, self.m, self.r))])
    for _ in range(self.L - 2):
      self.w_s += [
          nn.Parameter(w_scale * t.randn(self.d, self.m, self.r, self.r))
      ]
      self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.m, self.r))]
      self.a_s += [nn.Parameter(self.a_std * t.randn(self.d, self.m, self.r))]

    self.w_s += [nn.Parameter(w_scale * t.randn(self.d, self.m, self.r, 1))]
    self.b_s += [nn.Parameter(self.b_std * t.randn(self.d, self.m, 1))]
    self.a_s += [nn.Parameter(self.a_std * t.randn(self.m))]

    # HT parameters
    if self.use_HT:
      self.HT_poolsize = HT_poolsize
      self.a_HT_std = .1
      a_scale = self.a_HT_std / np.sqrt(self.m)
      self.L_HT = int(np.ceil(np.log(self.d) / np.log(self.HT_poolsize)))
      self.a_HTs = t.nn.ParameterList()
      dim_l = self.d
      if self.use_MERA:
        self.a_MERAs = t.nn.ParameterList()

      for _ in range(self.L_HT - 1):
        dim_l = int(np.ceil(dim_l / self.HT_poolsize))
        self.a_HTs += [
            nn.Parameter(self.m * t.eye(self.m).repeat(dim_l, 1, 1))
        ]
        if self.use_MERA:
          self.a_MERAs += [nn.Parameter(5 * t.ones((dim_l, self.m, self.m)))]

      self.a_HTs += [nn.Parameter(a_scale * t.randn(1, self.m, 1))]

      # pooling layer for stabilizing nll
      self.pool_layer = AvgPool1d(kernel_size=self.HT_poolsize,
                                  stride=self.HT_poolsize,
                                  ceil_mode=True)

      # create couplings
      self.all_couplings = self.create_default_couplings()

    # add parameters for mixing variables
    self.mix_vars = mix_vars
    self.n_mix_terms = n_mix_terms
    if self.mix_vars:
      # permute the parameters
      nonzero_inds = [[i, i + j] for j in range(1, self.n_mix_terms + 1)
                      for i in range(self.d - j)]
      perms = [np.random.permutation(self.d) for _ in range(self.m)]
      permuted_nonzero_inds = []
      for n_perm, perm in enumerate(perms):
        permuted_nonzero_inds += [[n_perm, perm[inds[0]], perm[inds[1]]]
                                  for inds in nonzero_inds]

      params = (1 / np.sqrt(self.n_mix_terms)) * t.randn(
          len(permuted_nonzero_inds))
      self.mix_params = nn.Parameter(
          t.sparse_coo_tensor(list(zip(*permuted_nonzero_inds)),
                              params,
                              size=(self.m, self.d, self.d)).to_dense())

      # add gradient mask to prevent the zero entries from changing
      gradient_mask = t.zeros((self.m, self.d, self.d))
      gradient_mask[list(zip(*permuted_nonzero_inds))] = 1.0
      self.mix_params.register_hook(lambda grad: grad.mul_(gradient_mask))

  def phis(self, X, inds=...):
    # X is a B x dim(inds) tensor
    # returns B x n x dim(inds) tensor
    phis = self.expand_X(X)

    # keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[inds, ...] for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [a[inds, ...] for a in self.a_s[:-1]]

    # compute CDF using a feed-forward network
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as):
      phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(w)) + b
      phis = phis + t.tanh(phis) * t.tanh(a)

    phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(
        sliced_ws[-1])) + sliced_bs[-1]

    phis = self.phi(phis)
    return t.squeeze(phis, -1)

  def phidots(self, X, inds=..., missing_data_mask=None):
    # X is a B x dim(inds) tensor
    # returns B x d x m tensor
    X = self.expand_X(X)  # B x dim(inds) x m x 1

    if self.mix_vars:
      X = self.mix_X(X)

    # keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[inds, ...] for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [a[inds, ...] for a in self.a_s[:-1]]

    # compute CDF using a feed-forward network
    phis = X
    phidots = t.ones_like(phis)
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as):
      phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(w)) + b
      phidots = t.einsum('mjik,jikl,mjil->mjil', phidots, self.nonneg_m(w),
                         (1 + t.tanh(a) * utils.tanhdot(phis)))
      phis = phis + t.tanh(phis) * t.tanh(a)
    phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg_m(
        sliced_ws[-1])) + sliced_bs[-1]
    phidots = t.einsum('mjik,jikl,mjil->mjil', phidots,
                       self.nonneg_m(sliced_ws[-1]), utils.sigmoiddot(phis))
    phidots = t.squeeze(phidots, -1)

    # mask out missing data
    if missing_data_mask is not None:
      assert self.mix_vars is False  # we can't marginalize if variables are mixed
      # mask is zero at missing data entries
      phidots = t.einsum('mj,mji->mji', missing_data_mask, phidots) + \
                t.einsum('mj,mji->mji', 1 - missing_data_mask, t.ones_like(phidots))

    return phidots

  def get_stabilizer(self, phidots, eps=1e-40):
    if self.use_HT:
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.transpose(1, 2).squeeze(-1)  # put d last
      fm = self.pool_layer(fm)  # mean over pooled variables
      fm = fm.min(1, True)[0]  # min over m, fm is B x d / p x 1
      fm = fm.mean(2, True)  # mean over d / p, f is B x 1 x 1
    else:
      # assuming CP
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.mean(1, True)  # mean over d
      fm = fm.min(2, True)[0]  # min over m (fm is B dimensional)
    fm = fm.squeeze()
    return fm

  def expand_X(self, X):
    if len(X.shape) == 1:
      X = X.unsqueeze(-1)
    X = X.unsqueeze(-1).unsqueeze(-1)
    Xn = X.expand(-1, -1, self.m, 1)  # B x d x m x 1

    return Xn

  def CDF(self, X, inds=...):
    # Evaluate joint CDF at X
    # X : B x len(inds) tensor of sample points
    # returns a B dim tensor

    phis = self.phis(X, inds)
    F = self.contract(phis, inds)
    return F

  def contract(self, T, inds=...):
    if self.use_HT and self.use_MERA:
      return self.MERA_contraction(T, inds=inds)
    elif self.use_HT and not self.use_MERA:
      return self.HT_contraction(T, inds=inds)
    else:
      # assuming CP
      return self.CP_contraction(T)

  def likelihood(self, X, inds=..., stabilize=False, missing_data_mask=None):
    # Evaluate joint likelihood at X
    # X : B x d tensor of sample points
    # inds : list of indices to restrict to (if interested in a subset of variables)

    phidots = self.phidots(X, inds, missing_data_mask=missing_data_mask)
    fm = None
    # phidots is B x d x m

    # stabilize if required
    if stabilize:
      fm = self.get_stabilizer(phidots)
      phidots = t.exp(fm).unsqueeze(-1).unsqueeze(-1) * phidots

    lk = self.contract(phidots, inds)
    return lk, fm

  def CP_contraction(self, T):
    # T is a B x len(inds) x m tensor
    T = t.prod(T, 1)  # prod over inds
    a = self.nonneg(self.a_s[-1])
    return t.einsum('mi,i->m', T, a / a.sum())

  def HT_contraction(self, T, inds=...):
    # T is a B x len(inds) x m tensor (some combination of phis and phidots) that is contracted with a
    # tensor of parameters to obtain a scalar
    # returns a B dimensional tensor

    # add ones for marginalized variables (can be avoided to speed up and save memory)
    if inds is not ...:
      T_full = t.ones((T.shape[0], self.d, self.m))
      T_full[:, inds, :] = T
      T = T_full

    for a_s, couplings in zip(self.a_HTs, self.all_couplings):
      T = [t.prod(T[:, coupling, :], dim=1) for coupling in couplings]
      # normalize sum of a_s across second dimension
      a_s = self.nonneg(a_s)
      a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
      T = t.stack([t.matmul(phid, a) for phid, a in zip(T, a_s)], dim=1)

    return t.squeeze(T)

  def MERA_contraction(self, T, inds=...):
    # T is a B x len(inds) x m tensor (some combination of phis and phidots) that is contracted with a
    # tensor of parameters to obtain a scalar
    # returns a B dimensional tensor

    from opt_einsum import contract

    # add ones for marginalized variables (can be avoided to speed up and save memory)
    if inds is not ...:
      T_full = t.ones((T.shape[0], self.d, self.m))
      T_full[:, inds, :] = T
      T = T_full

    for a_s, a2_s, couplings in zip(self.a_HTs[:-1], self.a_MERAs,
                                    self.all_couplings):
      T = t.stack([
          t.stack([
              t.prod(T[:, coupling, :], dim=1), T[:, coupling[0], :] *
              t.roll(T[:, coupling[1], :], shifts=1, dims=1)
          ]) if len(coupling) == 2 else
          T[:, coupling, :].squeeze().unsqueeze(0).expand(2, -1, -1)
          for coupling in couplings
      ])
      a_s = self.nonneg(a_s)
      a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
      a2_s = t.sigmoid(a2_s).unsqueeze(1)
      a2_s = t.cat((a2_s, 1 - a2_s), dim=1)
      T = contract('jklm,jkmi,jmi->lji', T, a2_s, a_s)
      # T = t.einsum('jklm,jkpm,jpi->lji', T, a2_s, a_s)

    T = t.prod(T[:, self.all_couplings[-1][0], :], dim=1)
    a_s = self.nonneg(self.a_HTs[-1])
    a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
    T = t.matmul(T, a_s).squeeze()
    return T

  def marginal_likelihood(self, X):
    marg_l = t.prod(t.stack(
        [self.likelihood(X[:, i], inds=[i])[0] for i in range(self.d)]),
                    dim=0)
    return marg_l

  def marginal_nll(self, X):
    log_marginal_density = t.log(self.marginal_likelihood(X))
    return -t.mean(log_marginal_density)

  def log_density(self,
                  X,
                  inds=...,
                  eps=1e-40,
                  stabilize=False,
                  missing_data_mask=None):
    lk, fm = self.likelihood(X,
                             inds,
                             stabilize=stabilize,
                             missing_data_mask=missing_data_mask)
    if stabilize:
      if inds is ...:
        n_vars = self.d
      else:
        n_vars = len(inds)
      return t.log(lk + eps) - n_vars * fm
    else:
      return t.log(lk + eps)

  def nll(self, X, inds=..., stabilize=False, missing_data_mask=None):
    # negative log likelihood
    # X : B x d tensor of sample points
    return -t.mean(
        self.log_density(X,
                         inds=inds,
                         stabilize=stabilize,
                         missing_data_mask=missing_data_mask))

  def sample(self,
             S,
             batch_size=None,
             inds=None,
             n_bisect_iter=35,
             upper_bound=1e3,
             lower_bound=-1e3,
             eps=1e-10):
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
          curr_condCDF = self.condCDF(k, samples[:, :k], inds, eps=eps)
          samples[:, k] = utils.invert(curr_condCDF,
                                       batch[:, k],
                                       n_bisect_iter=n_bisect_iter,
                                       ub=upper_bound,
                                       lb=lower_bound)
        all_samples.append(samples.cpu().detach().numpy())
      return np.concatenate(all_samples)

  def condCDF(self, k, prev_samples, inds, eps=1e-10):
    # compute the conditional CDF F(x_{inds[k]}|x_{inds[0]},...,x_{inds[k-1]})
    # prev_samples: S x (k-1) tensor of variables to condition over
    # returns a function R^S -> [0,1]^S

    if k == 0:
      # there is nothing to condition on
      phidots = 1
      denom = 1
    else:
      phidots = self.phidots(prev_samples, inds=inds[:k])
      denom = self.contract(phidots, inds=inds[:k]) + eps

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

  def create_adaptive_couplings(self, batches):
    # Assumes the data is standardized
    all_couplings = []
    if self.HT_poolsize > 2:
      raise ValueError("adaptive coupling not supported with HT_poolsize > 2")

    with t.no_grad():
      pool_layer = AvgPool2d(kernel_size=self.HT_poolsize,
                             stride=self.HT_poolsize,
                             ceil_mode=True)
      Sigma = t.zeros((self.d, self.d))
      for X in batches:
        Sigma += t.matmul(X.transpose(0, 1), X)

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

  def mix_X(self, X):
    # X is B x dim(inds) x m x 1

    X = X + t.einsum('ijk,bkil->bjil', self.nonneg(self.mix_params),
                     t.sigmoid(X))

    return X
