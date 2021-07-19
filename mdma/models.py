# Copyright © 2021 Dar Gilboa, Ari Pakman and Thibault Vatter
# This file is part of the mdma library and licensed under the terms of the MIT license.
# For a copy, see the LICENSE file in the root directory.

from typing import List, Callable
import torch as t
import torch.nn as nn
import numpy as np
from mdma import utils
from torch.nn import AvgPool1d, AvgPool2d


class MDMA(nn.Module):
  """ MDMA Density estimator

  Attributes:
     d: Dimension of data
     m: Width of MDMA model
     l: Depth of univariate density networks
     r: Width of univariate density networks
     w_std: Standard deviation of univariate weight matrices
     w_bias: Bias of univariate weight matrices
     b_std: Standard deviation of univariate biases
     b_bias: Bias of univariate biases
     a_std: Standard deviation of univariate weight vectors
     a_HT_std: Standard deviation of Hierachical Tucker tensor parameters
     HT_poolsize: Hierachical Tucker tensor pool size
     use_HT: Use HT tensor
     use_MERA: Use MERA tensor
     adaptive_coupling: Couple variables in hierarchical tensor based on correlations
     mix_vars: Transform inputs by an invertible transformation that prevents marginalizability
     n_mix_terms: Number of variables to mix when mix_vars == True (between 1 and self.d - 1)
  """
  def __init__(
      self,
      d: int,
      m: int = 100,
      l: int = 2,
      r: int = 3,
      w_std: float = 1.0,
      w_bias: float = 1.0,
      b_std: float = 0,
      b_bias: float = 0,
      a_std: float = 1.0,
      a_HT_std: float = 0.1,
      HT_poolsize: int = 2,
      use_HT: bool = True,
      use_MERA: bool = False,
      adaptive_coupling: bool = True,
      mix_vars: bool = False,
      n_mix_terms: int = 1,
  ):
    """ Initialize an instance of the MDMA class.
    """
    super(MDMA, self).__init__()
    self.d = d
    self.m = m
    self.r = r
    self.l = l
    self.nonneg = t.nn.Softplus(10)
    self.w_std = w_std
    self.w_bias = w_bias
    self.a_std = a_std
    self.a_HT_std = a_HT_std
    self.b_std = b_std
    self.b_bias = b_bias
    self.use_HT = use_HT
    self.use_MERA = use_MERA
    self.adaptive_coupling = adaptive_coupling
    self.mix_vars = mix_vars
    self.n_mix_terms = n_mix_terms

    # Initialize parameters for univariate CDFs
    assert self.l >= 2
    w_scale = self.w_std / t.sqrt(t.Tensor([self.r]))
    self.w_s = t.nn.ParameterList([
        nn.Parameter(self.w_std * t.randn(self.d, self.m, 1, self.r) + w_bias)
    ])
    self.b_s = t.nn.ParameterList(
        [nn.Parameter(self.b_std * t.randn(self.d, self.m, self.r) + b_bias)])
    self.a_s = t.nn.ParameterList(
        [nn.Parameter(self.a_std * t.randn(self.d, self.m, self.r))])
    for _ in range(self.l - 2):
      self.w_s += [
          nn.Parameter(w_scale * t.randn(self.d, self.m, self.r, self.r) +
                       w_bias)
      ]
      self.b_s += [
          nn.Parameter(self.b_std * t.randn(self.d, self.m, self.r) + b_bias)
      ]
      self.a_s += [nn.Parameter(self.a_std * t.randn(self.d, self.m, self.r))]

    self.w_s += [
        nn.Parameter(w_scale * t.randn(self.d, self.m, self.r, 1) + w_bias)
    ]
    self.b_s += [
        nn.Parameter(self.b_std * t.randn(self.d, self.m, 1) + b_bias)
    ]

    # HT parameters
    if self.use_HT:
      self.HT_poolsize = HT_poolsize
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

      # Pooling layer for stabilizing nll
      self.pool_layer = AvgPool1d(kernel_size=self.HT_poolsize,
                                  stride=self.HT_poolsize,
                                  ceil_mode=True)

      # Create couplings
      self.all_couplings = self.create_default_couplings()
    else:
      self.a_CP = nn.Parameter(self.a_std * t.randn(self.m))

    # Add parameters for mixing variables
    if self.mix_vars:
      # Permute the parameters
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

      # Add gradient mask to prevent the zero entries from changing
      gradient_mask = t.zeros((self.m, self.d, self.d))
      gradient_mask[list(zip(*permuted_nonzero_inds))] = 1.0
      self.mix_params.register_hook(lambda grad: grad.mul_(gradient_mask))

  def phis(self,
           X: t.Tensor,
           inds: List[int] = ...,
           single_phi_per_point: bool = False,
           ks: t.Tensor = None) -> t.Tensor:
    """ Evaluate all univariate CDFs at points X.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for computing marginal CDFs. Defaults to all variables.
      single_phi_per_point: Indicated whether a single CDF is evaluated for each datapoint, used for fast sampling
      ks: [B x len(inds)] matrix of integers between 0 and self.m - 1 indicating which univariate CDFs to evaluate for each datapoint.

    Returns:
      A [B x len(inds) x self.m] tensor of CDF values if fast_sample == False, otherwise a [B x len(inds)] tensor.
    """

    if single_phi_per_point:
      phis = X.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
      ks = ks.transpose(0, 1).unsqueeze(-1)
      # Pick only the weights for variables in inds and neurons specified by ks
      sliced_ws = [
          w[inds, ...].gather(
              1,
              ks.unsqueeze(-1).expand(*([-1, -1] + list(w.shape[2:]))))
          for w in self.w_s
      ]
      sliced_bs = [
          b[inds, ...].gather(1, ks.expand(*([-1, -1] + list(b.shape[2:]))))
          for b in self.b_s
      ]
      sliced_as = [
          a[inds, ...].gather(1, ks.expand(*([-1, -1] + list(a.shape[2:]))))
          for a in self.a_s
      ]
    else:
      phis = self.expand_X(X)
      # Keep only the parameters relevant for the subset of variables specified by inds
      sliced_ws = [w[inds, ...] for w in self.w_s]
      sliced_bs = [b[inds, ...] for b in self.b_s]
      sliced_as = [a[inds, ...] for a in self.a_s]

    # Compute CDF using a feed-forward network
    # If fast_sample==True, i indexes the datapoints and m=1, otherwise m indexes the datapoints
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as):
      phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg(w)) + b
      phis = phis + t.tanh(phis) * t.tanh(a)

    phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg(
        sliced_ws[-1])) + sliced_bs[-1]

    phis = t.sigmoid(phis)
    return t.squeeze(phis, -1)

  def phidots(self,
              X: t.Tensor,
              inds: List[int] = ...,
              missing_data_mask: t.Tensor = None) -> t.Tensor:
    """ Evaluate all univariate PDFs at points X.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for computing marginal CDFs. Defaults to all variables.
      missing_data_mask: A tensor of the same shape as X indicating which values are considered missing and are to be ignored.
    
    Returns:
      A [B x len(inds) x self.m] tensor of PDF values.
    """
    X = self.expand_X(X)  # B x dim(inds) x m x 1

    if self.mix_vars:
      X = self.mix_X(X)

    # Keep only the parameters relevant for the subset of variables specified by inds
    sliced_ws = [w[inds, ...] for w in self.w_s]
    sliced_bs = [b[inds, ...] for b in self.b_s]
    sliced_as = [a[inds, ...] for a in self.a_s]

    # Compute CDF using a feed-forward network
    phis = X
    phidots = t.ones_like(phis)
    for w, b, a in zip(sliced_ws[:-1], sliced_bs[:-1], sliced_as):
      phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg(w)) + b
      phidots = t.einsum('mjik,jikl,mjil->mjil', phidots, self.nonneg(w),
                         (1 + t.tanh(a) * utils.tanhdot(phis)))
      phis = phis + t.tanh(phis) * t.tanh(a)
    phis = t.einsum('mjik,jikl->mjil', phis, self.nonneg(
        sliced_ws[-1])) + sliced_bs[-1]
    phidots = t.einsum('mjik,jikl,mjil->mjil', phidots,
                       self.nonneg(sliced_ws[-1]), utils.sigmoiddot(phis))
    phidots = t.squeeze(phidots, -1)

    # Mask out missing data
    if missing_data_mask is not None:
      assert self.mix_vars is False
      # Mask is zero at missing data entries
      phidots = t.einsum('mj,mji->mji', missing_data_mask, phidots) + \
                t.einsum('mj,mji->mji', 1 - missing_data_mask, t.ones_like(phidots))

    return phidots

  def get_stabilizer(self, phidots: t.Tensor, eps: float = 1e-40) -> t.Tensor:
    """ Compute tensor that is used to prevent underflow when contracting univariate PDFs.

    Args:
      phidots: [B x len(inds) x self.m] tensor of PDF values.
      eps: A small constant to prevent division by zero.

    Returns:
      A tensor of length B.
    """
    if self.use_HT:
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.transpose(1, 2).squeeze(-1)
      fm = self.pool_layer(fm)  # mean over pooled variables
      fm = fm.min(1, True)[0]  # min over m, fm is B x d / p x 1
      fm = fm.mean(2, True)  # mean over d / p, f is B x 1 x 1
    else:
      # Assuming CP
      fm = -t.log(t.clamp_min(phidots, eps)).detach()
      fm = fm.mean(1, True)  # mean over d
      fm = fm.min(2, True)[0]  # min over m
    fm = fm.squeeze()
    return fm

  def expand_X(self, X: t.Tensor) -> t.Tensor:
    if len(X.shape) == 1:
      X = X.unsqueeze(-1)
    X = X.unsqueeze(-1).unsqueeze(-1)
    Xn = X.expand(-1, -1, self.m, 1)  # B x d x m x 1
    return Xn

  def CDF(self, X: t.Tensor, inds: List[int] = ...) -> t.Tensor:
    """ Evaluate joint CDF at points X.
    
    Args:
      X: X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for computing marginal PDF. Defaults to all variables.

    Returns:
      A tensor of length B of CDF values at the points X.
    """

    phis = self.phis(X, inds)
    F = self.contract(phis, inds)
    return F

  def contract(self, T: t.Tensor, inds: List[int] = ...) -> t.Tensor:
    """ Contract a given tensor with a tensor of variables.  
    
    Args:
      T: [B x len(inds) x self.m] Tensor to contract.
      inds: List of indices between 0 and self.d for computing marginal PDF.

    Returns:
      A tensor of length B. 
    """
    if self.use_HT and self.use_MERA:
      return self.MERA_contraction(T, inds=inds)
    elif self.use_HT and not self.use_MERA:
      return self.HT_contraction(T, inds=inds)
    else:
      # assuming CP
      return self.CP_contraction(T)

  def likelihood(self,
                 X: t.Tensor,
                 inds: List[int] = ...,
                 stabilize: bool = False,
                 missing_data_mask: t.Tensor = None) -> t.Tensor:
    """ Compute likelihood at points X.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for computing marginal likelihoods. Defaults to all variables.
      stabilize: Use stabilized likelihood to avoid underflow in high dimensions.
      missing_data_mask: A tensor of the same shape as X indicating which values are considered missing and are to be ignored.

    Returns:
      A tensor of length B of likelihood values.
    """

    phidots = self.phidots(X, inds, missing_data_mask=missing_data_mask)
    fm = None
    # phidots is B x d x m

    # stabilize if required
    if stabilize:
      fm = self.get_stabilizer(phidots)
      phidots = t.exp(fm).unsqueeze(-1).unsqueeze(-1) * phidots

    lk = self.contract(phidots, inds)
    return lk, fm

  def CP_contraction(self, T: t.Tensor) -> t.Tensor:
    """ Compute CP contraction.

    Args:
      T: [B x len(inds) x self.m] Tensor to contract.

    Returns:
      A tensor of length B.
    """
    # T is a B x len(inds) x m tensor
    T = t.prod(T, 1)  # prod over inds
    a = self.nonneg(self.a_CP)
    return t.einsum('mi,i->m', T, a / a.sum())

  def HT_contraction(self, T: t.Tensor, inds: List[int] = ...) -> t.Tensor:
    """ Compute diagonal hierarchical Tucker contraction.

    Args:
      T: [B x len(inds) x self.m] Tensor to contract.
      inds: List of indices between 0 and self.d. Defaults to all variables.

    Returns:
      A tensor of length B.
    """

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

  def MERA_contraction(self, T, inds: List[int] = ...) -> t.Tensor:
    """ Compute MERA contraction.

    Args:
      T: [B x len(inds) x self.m] Tensor to contract.
      inds: List of indices between 0 and self.d. Defaults to all variables.

    Returns:
      A tensor of length B.
    """

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

    T = t.prod(T[:, self.all_couplings[-1][0], :], dim=1)
    a_s = self.nonneg(self.a_HTs[-1])
    a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
    T = t.matmul(T, a_s).squeeze()
    return T

  def log_density(self,
                  X: t.Tensor,
                  inds: List[int] = ...,
                  eps: float = 1e-40,
                  stabilize: bool = False,
                  missing_data_mask: t.Tensor = None) -> t.Tensor:
    """ Compute log density at points X.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for computing marginal log density. Defaults to all variables.
      eps: Small numerical constant added for stability.
      stabilize: Use stabilized likelihood to avoid underflow.
      missing_data_mask: A tensor of the same shape as X indicating which values are considered missing and are to be ignored.

    Returns:
      A tensor of length B of log densities.
    """

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

  def nll(self,
          X: t.Tensor,
          inds: List[int] = ...,
          stabilize: bool = False,
          missing_data_mask: t.Tensor = None) -> t.Tensor:
    """ Compute mean negative log likelihood at points X.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d for marginalization. Defaults to all variables.
      stabilize: Use stabilized likelihood to avoid underflow.
      missing_data_mask: A tensor of the same shape as X indicating which values are considered missing and are to be ignored.

    Returns:
      A scalar mean negative log likelihood over the data.
    """

    return -t.mean(
        self.log_density(X,
                         inds=inds,
                         stabilize=stabilize,
                         missing_data_mask=missing_data_mask))

  def sample_autoregressive(self,
                            S: int,
                            batch_size: int = None,
                            inds: List[int] = None,
                            n_bisect_iter: int = 35,
                            upper_bound: float = 1e3,
                            lower_bound: float = -1e3,
                            eps: float = 1e-10) -> t.Tensor:
    """ Autoregressive sampling from density model.

    This is the standard sampling procedure that requires inverting a conditional CDF for each variable, and thus the
    runtime is linear in the number of dimensions.

    Args:
      S: Number of samples.
      batch_size: Batch size to use for sampling (if not provided, S is used)
      inds: List of indices between 0 and self.d for samples from the marginal distribution. Defaults to all variables.
      n_bisect_iter: Number of iterations of the bisection method when inverting univariate CDFs
      upper_bound: Upper bound on variable values, used for bisection.
      lower_bound: Lower bound on variable values, used for bisection.
      eps: Small numerical constant for stability.

    Returns:
      A [S x len(inds)] matrix of samples.
    """

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

  def sample(
      self,
      S: int,
      batch_size: int = None,
      inds: List[int] = ...,
      cond_inds: List[int] = None,
      cond_X: t.Tensor = None,
      n_bisect_iter: int = 35,
      upper_bound: float = 1e3,
      lower_bound: float = -1e3,
  ) -> t.Tensor:
    """ Sampling from MDMA by sampling a mixture component first and then sampling from a single CDF for each datapoint.

    Args:
      S: Number of samples.
      batch_size: Batch size to use for sampling (if not provided, S is used)
      inds: List of indices between 0 and self.d - 1 for which to compute the density.
      cond_inds: List of indices between 0 and self.d to condition on (disjoint from inds).
      cond_X: [n_conds x len(cond_inds)] matrix of values to condition on.
      n_bisect_iter: Number of iterations of the bisection method when inverting univariate CDFs
      upper_bound: Upper bound on variable values, used for bisection.
      lower_bound: Lower bound on variable values, used for bisection.

    Returns:
      If not conditioning or n_conds == 1, returns a [S x len(inds)] matrix of samples.
      Otherwise, returns a [n_conds x S x len(inds)] tensor of samples.
      All variables that are not in inds or cond_inds are marginalized out.
    """

    assert self.use_HT
    if batch_size is None:
      batch_size = S
    if cond_inds is None:
      n_conds = 1
    else:
      n_conds, n_cond_vars = cond_X.shape
      if inds is ...:
        inds = [ind for ind in range(self.d) if ind not in cond_inds]
    if inds is ...:
      n_vars = self.d
    else:
      n_vars = len(inds)
    n_samples = S * n_conds

    all_samples = []
    with t.no_grad():
      # Sample mixture components
      ks = t.zeros((1, n_samples)).long()

      if cond_inds is not None:
        # Compute features for conditional sampling
        ffs = self.conditional_features(cond_inds, cond_X)
      else:
        ffs = [None] * self.L_HT

      for a, ff in zip(reversed(self.a_HTs), reversed(ffs)):
        # If a.shape[0] is smaller than prev_a.shape[0], truncate ks
        ks = ks[:a.shape[0]]

        # Get coefficents of categorical distributions
        a = self.nonneg(a.permute(
            1, 0, 2))  # multinomial_coeffs[p,q,s] = a[p,q,ks[q,s]]
        multinomial_coeffs = a.gather(
            2,
            ks.unsqueeze(0).expand(a.shape[0], -1, -1))

        # If conditioning, compute conditional multinomial coefficients
        # at this point multinomial_coeffs is [m x width_l x S * n_conds]
        if cond_inds is not None:
          ff = ff.permute(2, 1, 0)  # ff is m x width_l x n_conds
          multinomial_coeffs = multinomial_coeffs * ff.repeat(1, 1, S)
          # We don't need to normalize by the sum

        # Reorder so that the rows of multinomial_coeffs are categorical dists
        multinomial_coeffs = multinomial_coeffs.reshape((self.m, -1)).t()

        # Sample from multinomial distribution and repeat the results
        new_ks = t.multinomial(multinomial_coeffs,
                               1).reshape_as(ks)  # dim_{l+1} x S
        ks = t.repeat_interleave(new_ks, repeats=self.HT_poolsize,
                                 dim=0)  # (self.HT_poolsize*dim_{l+1}) x S

      # Truncate if self.d is not a power of self.HT_poolsize, and restrict to the variables in inds
      ks = ks[:self.d]
      ks = ks[inds, ...]
      ks = ks.t()

      # Sample from each component
      U = t.rand(n_samples, n_vars)
      U_dataloader = t.utils.data.DataLoader(t.stack([U, ks]).transpose(0, 1),
                                             batch_size=batch_size,
                                             shuffle=False)
      for batch_idx, batch in enumerate(U_dataloader):
        batch_Us, batch_ks = batch[:, 0, :], batch[:, 1, :].long()

        # Define a CDF function R^(batch_size x self.d) -> [0,1]^(batch_size x self.d) and invert
        def CDF(s):
          phis = self.phis(s,
                           inds=inds,
                           single_phi_per_point=True,
                           ks=batch_ks)
          return phis.squeeze().t()

        samples = utils.invert(CDF,
                               batch_Us,
                               n_bisect_iter=n_bisect_iter,
                               ub=upper_bound,
                               lb=lower_bound)
        all_samples.append(samples.cpu().detach().numpy())
      return np.concatenate(all_samples).reshape(
          (n_conds, S, n_vars)).squeeze()

  def conditional_features(
      self,
      cond_inds: List[int],
      cond_X: t.Tensor,
  ):
    """

    Args:
      cond_inds:
      cond_X: n_conds x len(cond_inds)

    Returns:
      List of tensors of shape [n_conds x width_l x self.m]
    """
    n_conds = cond_X.shape[0]
    f = t.ones((n_conds, self.d, self.m))
    f[:, cond_inds, :] = self.phidots(cond_X, cond_inds)
    ffs = []

    for a_s, couplings in zip(self.a_HTs, self.all_couplings):
      f = [t.prod(f[:, coupling, :], dim=1) for coupling in couplings]
      # we only use products of fs, storing after computing products reduces the memory required
      ffs += [t.stack(f, dim=1)]

      # normalize sum of a_s across second dimension
      a_s = self.nonneg(a_s)
      a_s = a_s / t.sum(a_s, dim=1, keepdim=True)
      f = t.stack([t.matmul(phid, a) for phid, a in zip(f, a_s)], dim=1)
    return ffs

  def condCDF(self,
              k: int,
              prev_samples: t.Tensor,
              inds: List[int],
              eps=1e-10) -> Callable[[t.Tensor], t.Tensor]:
    """ Compute conditional CDF F(x_{inds[k]}|x_{inds[0]},...,x_{inds[k-1]}).

    Used for autoregressive sampling.

    Args:
      k: Index of variable for which to compute the CDF.
      prev_samples: [S x k - 1] matrix of values to condition on.
      inds: List of indices between 0 and self.d - 1.
      eps: Small numerical constant for stabilty.

    Returns:
      A function that maps a vector of S floats to a vector of S CDF values between 0 and 1.
    """

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

  def cond_density(self, X: t.Tensor, inds: List[int], cond_X: t.Tensor,
                   cond_inds: List[int]) -> t.Tensor:
    """ Compute marginal density at points X conditioned on a subset of variables.

    Args:
      X: [B x len(inds)] matrix of datapoints.
      inds: List of indices between 0 and self.d - 1 for which to compute the density.
      cond_X: [B x len(cond_inds)] matrix of values to condition on.
      cond_inds: List of indices between 0 and self.d to condition on.

    Returns:
      A size B tensor of conditional densities.
    """

    cond_phidots = self.phidots(cond_X, inds=cond_inds)
    denom = self.contract(cond_phidots, inds=cond_inds)
    joint_X = t.cat([X, cond_X], dim=1)
    joint_inds = inds + cond_inds
    phidots = self.phidots(joint_X, inds=joint_inds)
    numerator = self.contract(phidots, inds=joint_inds)
    return numerator / denom

  def create_default_couplings(self) -> List[List[List[int]]]:
    """ Create default couplings between variables for HT decomposition.

    Returns:
      List of lists of pairs of indices that indicate which subsets of variables to couple at every layer of the decomposition.
    """

    all_couplings = []
    dim_l = self.d
    for _ in range(self.L_HT):
      coupling_ranges = [(j, min(dim_l, j + self.HT_poolsize))
                         for j in range(0, dim_l, self.HT_poolsize)]
      couplings = [list(range(rng[0], rng[1])) for rng in coupling_ranges]
      dim_l = int(np.ceil(dim_l / self.HT_poolsize))
      all_couplings.append(couplings)
    return all_couplings

  def create_adaptive_couplings(self, batches: List[t.Tensor]):
    """ Create adaptive couplings between variables for HT decomposition.

    Variables are coupled together in a greedy fashion by pairing the most highly coupled variables that have not yet
    been coupled at every step. The process is repeated at every layer by averaging over the covariance matrix to obtain
    the covariance between subsets of variables. Assumes the data is standardized.

    Args:
      batches: A list of [B x self.d] tensors, each containing B datapoints.

    Returns:
      List of lists of pairs of indices that indicate which subsets of variables to couple at every layer of the
      decomposition.
    """

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
        # Add couples at layer l
        couplings = []
        coupled_inds = []
        n_couples = int(np.ceil(dim_l / self.HT_poolsize))

        # Sort correlation matrix if this isn't the last layer
        if n_couples > 1:
          sorted_flat_corrs = t.sort(Sigma.flatten(), descending=True)
          sorted_flat_inds = sorted_flat_corrs.indices[::
                                                       2]  # discard symmetric entries
          sorted_inds = [[ind // dim_l, ind % dim_l]
                         for ind in sorted_flat_inds]

        for _ in range(n_couples):
          if dim_l - len(coupled_inds) <= self.HT_poolsize:
            # We added all the couples, there are <= HT_poolsize variables left
            remainder = []
            for i in range(dim_l):
              if i not in coupled_inds:
                remainder.append(t.tensor(i))
                coupled_inds.append(i)
            couplings.append(remainder)
          else:
            # Find the next couple to add
            while sorted_inds[0][0] in coupled_inds \
                or sorted_inds[0][1] in coupled_inds \
                or sorted_inds[0][0] == sorted_inds[0][1]:
              # One of the variables is already coupled or this is diagonal entry, pop
              sorted_inds.pop(0)
            # A new couple to add has been found
            couplings.append(sorted_inds[0])
            coupled_inds.append(sorted_inds[0][0].item())
            coupled_inds.append(sorted_inds[0][1].item())

        # Coarse-grain Sigma and store
        perm_Sigma = Sigma[coupled_inds][:, coupled_inds]
        Sigma = pool_layer(perm_Sigma.unsqueeze(0)).squeeze()
        dim_l = n_couples
        all_couplings.append(couplings)

    self.all_couplings = all_couplings

  def mix_X(self, X: t.Tensor) -> t.Tensor:
    """ Transform a tensor using an invertible transformation

    Args:
      X: [B x dim(inds) x self.m x 1] tensor to transform.

    Returns:
      Tensor of transformed variables with the same shape as X
    """

    X = X + t.einsum('ijk,bkil->bjil', self.nonneg(self.mix_params),
                     t.sigmoid(X))

    return X
