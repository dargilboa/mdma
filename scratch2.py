import cdfnet.fit as fit
import UCI_density_estimation
h = fit.get_default_h()
h.dataset = 'power'
h.data_dir = '/data/data'
#h.data_dir = '/Users/dargilboa/Downloads/data'
h.tb_dir = '/data/tb'
h.batch_size = 1500
data = UCI_density_estimation.load_dataset(h)

import numpy as np
import cdfnet.utils as utils

#h = fit.get_default_h()

h.n = 500
h.m = 10
h.L = 6
h.lr = .05
#h.d = 6
#h.M = 1000
#dataset = np.random.randn(h.M, h.d)
#loaders = utils.create_loaders([dataset, 0, 0], h.batch_size)
h.eval_validation = False
h.eval_test = False
h.save_checkpoints = False
h.checkpoint_every = 1000
h.model_to_load = ''
h.use_HT = True
h.use_tb = False
h.exp_name = 'HT'
h.patience = 3000
h.set_detect_anomaly = True
outs = fit.fit_neural_copula(h, data)