import torch as t
import torch.nn as nn
import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import interp1d
import utils

# %%
class CopNet(nn.Module):
  def __init__(self, n, d, device='cpu', sample_scale=3, n_samples=1000, W_std=0.1, b_bias=0, b_std=1.0):
    super(CopNet, self).__init__()
    self.device = device
    self.n = n
    self.d = d
    self.sample_scale = sample_scale
    self.n_samples = n_samples
    self.phi = t.sigmoid
    self.phi_np = utils.sigmoid
    self.phidot = lambda x: t.sigmoid(x) * (1 - t.sigmoid(x))

    # initialize trianable parameters
    self.W = nn.Parameter(t.Tensor(W_std * np.random.randn(n, d)).double())
    self.b = nn.Parameter(t.Tensor(b_std * np.random.randn(n, d) + b_bias).double())
    self.a = nn.Parameter(t.Tensor(np.random.randn(n, )).double())

    # initialize z using normal samples
    self.update_zs()

  def update_zs(self, data=None):
    if data is None:
      # generate data if not provided
      data = self.sample_scale * t.randn(self.n_samples, self.d, dtype=t.double)
    g = self.g(data)
    self.z, self.zdot = self.z_zdot(data, g)

  def invert(self, f, r, xtol=1e-8):
    # return f^-1(r)
    return bisect(lambda x: f(x) - r, 0, 1, xtol=xtol)

  def sample(self, M):
    # return M samples from the copula using the inverse Rosenblatt formula
    samples = []
    for i in range(M):
      r = np.random.rand(self.d)
      us = [r[0]]
      for k in range(1, self.d):
        cC = self.condC(k, us)
        u_k = self.invert(cC, r[k])
        us += [u_k]
      samples += [us]
    return np.asarray(samples)

  def condC(self, k, us):
    # returns C(u_k|u_1,...,u_{k-1}) for some 1 <= i <= d
    # us: length k-1 vector of conditional values for u_1 ... u_{k-1}
    # returns a numpy function to be fed into bisect
    zu = t.Tensor([z_j(u_j) for z_j, u_j in zip(self.z[:k], us)])
    zdu = t.Tensor([zdot_j([u_j])[0] for zdot_j, u_j in zip(self.zdot[:k], us)])
    A = t.einsum('ij,j->ij', t.exp(self.W[:,:k]), zu) + self.b[:,:k] # n x k-1
    AA = self.phidot(A) * t.exp(self.W[:,:k]) * zdu.unsqueeze(0).expand(self.n, k)
    AAA = t.prod(AA,1) # n
    iZ = 1/(t.sum(t.exp(self.a))).detach().numpy()
    cond_CDF = lambda u: iZ * np.sum(t.exp(self.a).detach().numpy() * AAA.detach().numpy()
                                    * self.phi_np(t.exp(self.W[:,k]).detach().numpy() * self.z[k](u)
                                               + self.b[:,k].detach().numpy()))
    return lambda u: cond_CDF(u) / cond_CDF(1)

  def NLL(self, u):
    # compute NLL (per datapoint)
    # u: M x d array
    M = u.shape[0]
    NLL = t.log(t.sum(t.exp(self.a)))
    zu = t.Tensor([z_j(u_j) for z_j, u_j in zip(self.z, u.transpose(0,1))])
    zdu = t.Tensor([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, u.transpose(0, 1))])
    A = t.einsum('ij,jm->ijm', t.exp(self.W), zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    AA = self.phidot(A) * t.exp(self.W).unsqueeze(-1).expand(self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d, M)
    AAA = t.prod(AA,1) # n x M
    NLL -= t.mean(t.log(t.einsum('i,im->m', t.exp(self.a), AAA)))
    return NLL

  def log_density(self, u):
    # compute log density
    # u: M x d array
    M = u.shape[0]
    zu = t.Tensor([z_j(u_j) for z_j, u_j in zip(self.z, u.transpose(0,1))])
    zdu = t.Tensor([zdot_j(u_j) for zdot_j, u_j in zip(self.zdot, u.transpose(0, 1))])
    A = t.einsum('ij,jm->ijm', t.exp(self.W), zu) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    AA = self.phidot(A) * t.exp(self.W).unsqueeze(-1).expand(self.n, self.d, M) * zdu.unsqueeze(0).expand(self.n, self.d, M)
    AAA = t.prod(AA,1) # n x M
    log_density = t.log(t.einsum('i,im->m', t.exp(self.a), AAA)) - t.log(t.sum(t.exp(self.a))).unsqueeze(-1).expand(M)
    return log_density

  def g(self, u):
    # u: M x d array of points
    M = u.size(0)
    A = t.einsum('ik,ka->ika', t.exp(self.W), u.transpose(0, 1)) + self.b.unsqueeze(-1).expand(self.n, self.d, M)
    return t.einsum('i,ika->ka', t.exp(self.a), self.phi(A)).transpose(0, 1) / t.sum(t.exp(self.a))

  def z_zdot(self, s, g):
    # returns a list of d functions z_j = g_j^{-1} by using spline interpolation
    # s: M x d array
    # g: M x d array such that s = z(g), where g : R -> [0,1]

    s = s.detach().numpy().transpose().astype(np.float)
    g = g.detach().numpy().transpose().astype(np.float)

    #splines = [UnivariateSpline(sorted(g_k), sorted(s_k), k=5, s=1) for g_k, s_k in zip(g, s)] #this assumes g is monotonic, otherwise we should sort (g,u) pairs in order of increasing g
    # splines = [PchipInterpolator(np.asarray(sorted(g_k)), np.asarray(sorted(s_k))) for g_k, s_k in zip(g, s)] #this assumes g is monotonic, otherwise we should sort (g,u) pairs in order of increasing g
    #dsplines = [sp.derivative() for sp in splines]
    # # linear interpolation
    # splines = [interp1d(g_k, s_k, kind='linear', fill_value='extrapolate') for g_k, s_k in zip(g, s)]
    # dsplines = [self.d_interp(g_k,s_k) for g_k, s_k in zip(g, s)]

    # reparameterize in terms of \tilde{z}_j: [0,1] -> [0,1], so that z = sigma^1 o \tilde{z}
    s = utils.sigmoid(s)
    # add endpoints
    g = [np.concatenate((g_k, np.array([0,1]))) for g_k in g]
    s = [np.concatenate((s_k, np.array([0,1]))) for s_k in s]

    # interpolate \tilde{z}_j and \dot{\tilde{z}}_j, then use the results to construct z_j and \dot{z}_j
    tilde_splines = [interp1d(g_k, s_k, kind='linear', fill_value='extrapolate') for g_k, s_k in zip(g, s)]
    tilde_dsplines = [self.d_interp(g_k, s_k) for g_k, s_k in zip(g, s)]
    z = [lambda x: utils.invsigmoid(ztilde(x)) for ztilde in tilde_splines]
    zdot = [lambda x: utils.invsigmoiddot(ztilde(x)) * ztildedot(x) for ztilde, ztildedot in zip(tilde_splines,
                                                                                            tilde_dsplines)]
    return z, zdot#splines, dsplines

  def d_interp(self, x, y):
    interp = utils.d_interpolator(x, y)
    return interp.interpolate_derivative

  def stabilize(self, u, stab_const=1e-5):
    return t.clamp(u, stab_const, 1 - stab_const)