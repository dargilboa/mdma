import torch as t
import torch.nn as nn
from xitorch.interpolate import Interp1D
import utils

# %%
class CopNet(nn.Module):
  def __init__(self, n, d, z_update_samples_scale=3, z_update_samples=1000, W_std=0.1,
               b_bias=0, b_std=0.1, a_std=1.0):
    super(CopNet, self).__init__()
    self.n = n
    self.d = d
    self.z_update_samples_scale = z_update_samples_scale
    self.z_update_samples = z_update_samples
    self.phi = t.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))

    # initialize trianable parameters
    self.W = nn.Parameter(t.Tensor(W_std * t.randn(n, d)))
    self.b = nn.Parameter(t.Tensor(b_std * t.randn(n, d) + b_bias))
    self.a = nn.Parameter(t.Tensor(a_std * t.randn(n, )))

    # initialize z using normal samples
    self.update_zs()

  def update_zs(self, data=None, bp_through_z_update=False):
    if data is None:
      # generate data if not provided
      zdata = self.z_update_samples_scale * t.randn(self.z_update_samples, self.d)
    else:
      # assuming data is samples from the hypercube, transforming with z
      with t.no_grad():
        zdata = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, data.transpose(0, 1))]).transpose(0,1)

    if bp_through_z_update:
      g = self.g(zdata)
      self.z, self.zdot = self.z_zdot(zdata, g)
    else:
      with t.no_grad():
        g = self.g(zdata)
        self.z, self.zdot = self.z_zdot(zdata, g)

  def invert(self, f, r, n_bisect_iter=35):
    # return f^-1(r)
    return utils.bisect(f, r, 0, 1, n_iter=n_bisect_iter)

  def sample(self, M, n_bisect_iter=35):
    # return M x d tensor of samples from the copula using the inverse Rosenblatt formula
    R = t.rand(self.d, M)
    samples = t.zeros_like(R)
    samples[0] = R[0]
    for k in range(1, self.d):
      cC = self.condC(k, samples[:k])
      samples[k] = self.invert(cC, R[k], n_bisect_iter=n_bisect_iter)

    return samples.transpose(0,1)

  def condC(self, k, us):
    # returns a function C(u_k|u_1,...,u_{k-1}) for some 1 <= k < d
    # this function takes an M-dimensional vector as input
    # us: k-1 x M tensor of conditional values for u_1 ... u_{k-1} at M sampled points

    M = us.shape[1]
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z[:k], us)])
    zdu = t.stack([zdot_j(u_j)[0] for zdot_j, u_j in zip(self.zdot[:k], us)]) # check for d > 2
    A = t.einsum('ij,jm->ijm', t.exp(self.W[:,:k]), zu) + self.b[:,:k].unsqueeze(-1).expand(self.n, k, M) # n x k-1 x M
    AA = self.phidot(A) * t.exp(self.W[:,:k]).unsqueeze(-1).expand(self.n, k, M) * zdu.unsqueeze(0).expand(self.n, k, M)
    AAA = t.prod(AA,1) # n x M
    iZ = 1/(t.sum(t.exp(self.a)))
    cond_CDF = lambda u: iZ * t.einsum('i,im,im->m', t.exp(self.a), AAA,
                                    self.phi(t.exp(self.W[:,k]).unsqueeze(-1).expand(self.n, M) * self.z[k](u).unsqueeze(0).expand(self.n, M)
                                             + self.b[:,k].unsqueeze(-1).expand(self.n, M))) # M
    return lambda u: cond_CDF(u) / cond_CDF(t.ones(M))

  def NLL(self, u):
    # compute NLL (per datapoint)
    # u: M x d tensor
    M = u.shape[0]
    NLL = t.log(t.sum(t.exp(self.a)))
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, u.transpose(0,1))])
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, u.transpose(0, 1))]) # d x M
    A = t.einsum('ij,jm->ijm', t.exp(self.W), zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    AA = self.phidot(A) * t.exp(self.W).unsqueeze(-1).expand(self.n, self.d, M) * zdu
    AAA = t.prod(AA,1) # n x M
    NLL -= t.mean(t.log(t.einsum('i,im->m', t.exp(self.a), AAA)))
    return NLL

  def diag_H(self, us):
    # returns M x d tensor of diagonal Hessian elements
    # us: M x d tensor of conditional values for u_1 ... u_{k-1} at M sampled points
    us = us.transpose(0,1)
    M = us.shape[1]
    eW = t.exp(self.W)
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, us)]) # d x M
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, us)]) # d x M
    p = t.einsum('ij,jm->ijm', eW, zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M) # n x d x M
    AA = self.phidot(p) * eW.unsqueeze(-1).expand(self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d, M) # n x d x M
    T = utils.dddsigmoid(p) * t.pow(eW, 2).unsqueeze(-1).expand(self.n, self.d, M) * t.pow(zdu, 3).unsqueeze(0).expand(self.n, self.d, M) # n x d x M
    diag_H = t.zeros(M, self.d)
    for k in range(self.d):
      prod = t.prod(t.cat([AA[:, :k, :], AA[:, k+1:, :]], 1), 1) # n x M
      AAA = t.einsum('i,im,im->im', t.exp(self.a), prod, T[:,k,:])
      AAA = t.sum(AAA, 0) # M
      diag_H[:,k] = AAA
    return diag_H

  def H(self, us):
    # returns M x d x d tensor of Hessians at every datapoint
    # us: M x d tensor of conditional values for u_1 ... u_{k-1} at M sampled points
    us = us.transpose(0,1)
    M = us.shape[1]
    eW = t.exp(self.W)
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(self.z, us)]) # d x M
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, us)]) # d x M
    p = t.einsum('ij,jm->ijm', eW, zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M) # n x d x M
    AA = self.phidot(p) * eW.unsqueeze(-1).expand(self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d, M) # n x d x M
    T = utils.ddsigmoid(p) * t.pow(eW, 2).unsqueeze(-1).expand(self.n, self.d, M) * t.pow(zdu, 2).unsqueeze(0).expand(self.n, self.d, M) # n x d x M
    H = t.zeros(M, self.d, self.d)
    for k in range(self.d):
      for l in range(k+1):
        prod = t.prod(t.cat([AA[:, :l, :], AA[:, l+1:k, :], AA[:, k+1:, :]], 1), 1) # n x M
        AAA = t.einsum('i,im,im,im->im', t.exp(self.a), prod, T[:,k,:], T[:,l,:])
        AAA = t.sum(AAA, 0) # M
        H[:, k, l] = H[:, l, k] = AAA
    return H

  def log_density(self, u, bivariate=False, bv_i=0, bv_j=1):
    # compute log density
    # u: M x n_var tensor (n_var=d in the full case and 2 in the bivariate case)
    M = u.shape[0]
    if bivariate:
      n_vars = 2
      z = [self.z[bv_i], self.z[bv_j]]
      zdot = [self.zdot[bv_i], self.zdot[bv_j]]
      eW = t.exp(self.W[:,[bv_i, bv_j]])
      b = self.b[:,[bv_i, bv_j]]
    else:
      n_vars = self.d
      z = self.z
      zdot = self.zdot
      eW = t.exp(self.W)
      b = self.b
    assert u.shape[1] == n_vars

    zu = t.stack([z_j(u_j) for z_j, u_j in zip(z, u.transpose(0,1))])
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(zdot, u.transpose(0, 1))])
    A = t.einsum('ij,jm->ijm', eW, zu) + b.unsqueeze(-1).expand(self.n, n_vars, M)
    AA = self.phidot(A) * eW.unsqueeze(-1).expand(self.n, n_vars, M) * zdu.unsqueeze(0).expand(self.n, n_vars, M)
    AAA = t.prod(AA,1) # n x M
    log_density = t.log(t.einsum('i,im->m', t.exp(self.a), AAA)) - t.log(t.sum(t.exp(self.a))).unsqueeze(-1).expand(M)
    return log_density

  def log_biv_marginal_density(self, u, i, j):
    # compute log of bivariate marginal density of i and j variables for d > 2
    # u: M x 2 tensor
    M = u.shape[0]
    z_ij = [self.z[i], self.z[j]]
    zdot_ij = [self.zdot[i], self.zdot[j]]
    zu = t.stack([z_j(u_j) for z_j, u_j in zip(z_ij, u.transpose(0,1))])
    zdu = t.stack([zdot_j(u_j) for zdot_j, u_j in zip(zdot_ij, u.transpose(0, 1))])
    A = t.einsum('ij,jm->ijm', t.exp(self.W), zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    AA = self.phidot(A) * t.exp(self.W).unsqueeze(-1).expand(self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d, M)
    AAA = t.prod(AA,1) # n x M
    log_density = t.log(t.einsum('i,im->m', t.exp(self.a), AAA)) - t.log(t.sum(t.exp(self.a))).unsqueeze(-1).expand(M)
    return log_density

  def g(self, u):
    # u: M x d tensor of points
    M = u.size(0)
    A = t.einsum('ik,ka->ika', t.exp(self.W), u.transpose(0, 1)) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    return t.einsum('i,ika->ka', t.exp(self.a), self.phi(A)).transpose(0, 1) / t.sum(t.exp(self.a))

  def z_zdot(self, s, g):
    # returns a list of d functions z_j = g_j^{-1} by using spline interpolation
    # s: M x d tensor
    # g: M x d tensor such that s = z(g), where g : R -> [0,1]

    s = s.transpose(0,1)
    g = g.transpose(0,1)

    # reparameterize in terms of \tilde{z}_j: [0,1] -> [0,1], so that z = sigma^1 o \tilde{z}
    s = self.phi(s)
    # add endpoints
    endpoints = t.tensor([0,1]).unsqueeze(0).expand(self.d, 2)
    g = t.cat((g, endpoints), dim=1)
    s = t.cat((s, endpoints), dim=1)

    # interpolate \tilde{z}_j and \dot{\tilde{z}}_j, then use the results to construct z_j and \dot{z}_j
    g, _ = t.sort(g, dim=1)
    s, _ = t.sort(s, dim=1)
    tilde_splines = [Interp1D(g_k, s_k, method='linear', extrap='bound') for g_k, s_k in zip(g, s)]

    dsdg = t.div(s[:,1:] - s[:,:-1], g[:,1:] - g[:,:-1])
    mg = (g[:,:-1] + g[:,1:]) / 2
    tilde_dsplines = [Interp1D(mg_k, ds_k, method='linear', extrap='bound') for mg_k, ds_k in zip(mg, dsdg)]

    z = [lambda x: utils.invsigmoid(ztilde(x)) for ztilde in tilde_splines]
    zdot = [lambda x: utils.invsigmoiddot(ztilde(x)) * ztildedot(x) for ztilde, ztildedot in zip(tilde_splines,
                                                                                            tilde_dsplines)]
    return z, zdot

  def stabilize(self, u, stab_const=1e-5):
    return t.clamp(u, stab_const, 1 - stab_const)