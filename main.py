# %%
import torch as t
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import utils
import models

#%% fitting the neural copula to gaussian data
#rhos = [-.95, -.75, 0, .75, .95]
rhos = [.8]
M = 500
d = 2
n = 500
n_iters = 500
n_samples_for_figs = 200
print_every = 1
update_z_every = 1
plot_NLL = True
np.random.seed(0)
t.manual_seed(0)

all_outs = []
for nr, rho in enumerate(rhos):
  NLLs = []
  outs = {}
  data = utils.generate_data(d, M, rho=rho)
  C = models.CopNet(n, d, b_bias=0, b_std=3,)
  verbose = True

  optimizer = optim.Adam(C.parameters(), lr=5e-3)
  scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=int(3*n_iters/4), gamma=0.1)

  t.autograd.set_detect_anomaly(True)

  # fit neural copula to data
  for i in range(n_iters):
    # update parameters
    optimizer.zero_grad()
    NLL = C.NLL(data)
    NLL.backward()
    optimizer.step()
    scheduler.step()

    # update z approximation, can also take the data as input
    if i % update_z_every == 0:
      C.update_zs()

    NLLs.append(NLL.detach().numpy())
    if verbose and i % print_every == 0:
      print('iteration {}, NLL: {:.4f}'.format(i,NLL.detach().numpy()))

  if plot_NLL:
    plt.figure()
    plt.plot(NLLs)
    plt.xlabel('iter')
    plt.ylabel('NLL per datapoint')
    plt.show()

  outs['NLLs'] = NLLs
  outs['model'] = C
  outs['data'] = data
  outs['rho'] = rho
  all_outs += [outs]

# sampling
for outs in all_outs:
  samples = outs['model'].sample(n_samples_for_figs)
  outs['samples'] = samples

# plotting samples
fig, axs = plt.subplots(len(rhos), 2, figsize=(6, 3*len(rhos)))
if len(rhos) == 1:
  axs = np.expand_dims(axs, 0)

for n_outs, outs in enumerate(all_outs):
  axs[n_outs, 0].scatter(outs['data'][:n_samples_for_figs,0],
                         outs['data'][:n_samples_for_figs,1],)
  axs[n_outs, 0].set_ylabel(str(outs['rho']))
  axs[n_outs, 1].scatter(outs['samples'][:,0], outs['samples'][:,1])
axs[0,0].set_title('training data')
axs[0,1].set_title('neural copula')
fig.show()


#%% plot log density
x = np.linspace(0.01, .99, 30)
y = np.linspace(0.01, .99, 30)
grid = np.meshgrid(x, y)
log_densities = C.log_density(t.tensor([g.flatten() for g in grid]).transpose(0,1)).detach().numpy().reshape((30,30))
gauss_log_densities = utils.gaussian_copula_log_density(t.tensor([g.flatten() for g in grid]).transpose(0,1), rho=0.8)
gauss_log_densities = np.array([g.detach().numpy() for g in gauss_log_densities]).reshape((30,30))
log_densities_flipped = log_densities[::-1]
gauss_log_densities_flipped = gauss_log_densities[::-1]
vmax = np.max([gauss_log_densities_flipped, log_densities_flipped])
vmin = np.min([gauss_log_densities_flipped, log_densities_flipped])

fig, axs = plt.subplots(1,2, figsize=(7, 3))
plt.setp(axs, xticks=[0,29], xticklabels=['0', '1'],
              yticks=[0,29], yticklabels=['1', '0'])
axs[0].imshow(log_densities_flipped, vmax=vmax, vmin=vmin, )
axs[0].title.set_text('neural copula log density')
im = axs[1].imshow(gauss_log_densities_flipped, vmax=vmax, vmin=vmin)
axs[1].title.set_text('gaussian log density')
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
fig.show()

#%% plot slices of density
fig, axs = plt.subplots(1,5, figsize=(15, 3))
for ni, ind in enumerate([0, 7, 14, 21, -1]):
  axs[ni].plot(np.exp(log_densities_flipped[ind]))
  axs[ni].plot(np.exp(gauss_log_densities_flipped[ind]))
  rx = x[::-1]
  axs[ni].title.set_text(str(rx[ind]))
plt.show()

#%% compare g to inverse (if the scatter plot and curve match, z is a good approx for g^-1
self = C
data = self.sample_scale * t.randn(1000, self.d, dtype=t.double)
g = self.g(data)
plt.plot(sorted(g[:,0].detach().numpy()), sorted(data[:,0].detach().numpy()), lw=4)
plt.plot(np.linspace(0.0001,.9999,1000), C.z[0](np.linspace(0.0001,.9999,1000)),'r', lw=4)
plt.legend(['g^{-1}',' z'])
plt.show()