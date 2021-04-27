class TenNet(nn.Module):
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
  ):
    super(TenNet, self).__init__()
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
    self.phidot_expr = [[None, None]] * self.L
    self.use_HT = use_HT

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
    return t.squeeze(phis, -1)

  def phidots(self, X, inds=...):
    # X is a M x dim(inds) tensor
    # returns M x n tensor

    phidots, fm = self.phidots_no_prod(X, inds=inds)
    phidots = t.prod(t.exp(fm).unsqueeze(1).unsqueeze(1) * phidots,
                     1)  # prod over d
    return t.squeeze(phidots), fm.squeeze()  # M x n

  def phidots_no_prod(self, X, inds=...):
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

    fm = -t.log(phidots + 1e-10).detach()
    fm = fm.mean(1, True)  # mean over d
    fm = fm.min(2, True)[0]  # min over n (fm is M dimensional)

    return t.squeeze(phidots, -1), fm.squeeze()

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

    if self.use_HT:
      phidots, fm = self.phidots_no_prod(X, inds)  # + 1e-10
      # phidots is M x d x n

      # regularize
      phidots = t.exp(fm).unsqueeze(1).unsqueeze(1) * phidots

      lk = self.HT_contraction(phidots, inds)
    else:
      phidots, fm = self.phidots(X, inds)
      import pdb
      pdb.set_trace()
      a = self.nonneg(self.a_s[-1])
      lk = einsum('mi,i->m', phidots, a / a.sum())

    return lk, fm

  def HT_contraction(self, T, inds=...):
    # T is a M x len(inds) x n tensor (some combination of phis and phidots) that is contracted with a
    # tensor of parameters to obtain a scalar
    # returns an M dimensional tensor

    # add ones for marginalized variables (can be avoided to speed up and save memory)
    if inds is not ...:
      T_full = t.ones((T.shape[0], self.d, self.n))
      T_full[:, inds, :] = T
      T = T_full
    # this can become unstable due to large values. we probably need to modify fm
    # (either make sure that it is defined correctly for the HT case or
    # add an equivalent term for large values rather than small
    dim_l = self.d
    for a_s in self.a_HTs:
      inds_to_prod = [(j, min(dim_l, j + self.HT_poolsize))
                      for j in range(0, dim_l, self.HT_poolsize)]

      T = [t.prod(T[:, inds[0]:inds[1], :], dim=1) for inds in inds_to_prod]
      # normalize sum of a_s across second dimension
      a_s = self.nonneg(a_s)
      a_s = a_s / t.sum(a_s, dim=1).unsqueeze(1)
      T = t.stack([t.matmul(phid, a) for phid, a in zip(T, a_s)], dim=1)
      dim_l = int(np.ceil(dim_l / self.HT_poolsize))

    return t.squeeze(T)

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
    if inds is ...:
      n_vars = self.d
    else:
      n_vars = len(inds)
    return t.log(lk + 1e-10) - fm * n_vars

  def nll(self, X, inds=...):
    # negative log likelihood
    # X : M x d tensor of sample points
    return -t.mean(self.log_density(X, inds=inds))