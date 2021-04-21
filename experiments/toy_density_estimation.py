import torch as t
import numpy as np
import matplotlib.pyplot as plt
import time
import plots
import fit
import utils
from experiments.BNAF.data.generate2d import sample2d

#%% fit density
d = 2
dataset = '2spirals'
batch_size = 2000
n_iters = 200
M = n_iters * batch_size

h = fit.get_default_h()
h.dataset = dataset
h.M = M
h.M_val = 500
h.d = d
h.epochs = 1
h.batch_size = batch_size
h.n = 500
h.l2 = 1e-5
h.lr = 1e-2

# init params
h.m = 5
h.l = 4
h.w_std = .01
h.a_std = .01
h.b_std = 0

np.random.seed(0)
t.manual_seed(0)

data = [
    sample2d(h.dataset, h.M),
    sample2d(h.dataset, h.M_val),
    sample2d(h.dataset, h.M_val)
]
loaders = utils.create_loaders(data, batch_size)
start_time = time.time()
outs = fit.fit_neural_copula(h, loaders, checkpoint_every=500)
run_time = (time.time() - start_time) / 60
print(f'Runtime: {run_time:.3g} mins')

plots.plot_heatmap(outs['model'], outs, xlim=[-4, 4], ylim=[-4, 4])

samples = outs['model'].sample(500)
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
