import torch as t
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()
r = robjects.r

r['source']('experiments/causal_discovery/ci_test/parCopCITest.R')

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
  device = "cuda"
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')
  device = "cpu"


def get_condCDFs(model, x, i, j, cond_inds):

  if len(cond_inds) == 0:
    ui = model.CDF(x[:, i].unsqueeze(1), inds=[i])
    uj = model.CDF(x[:, j].unsqueeze(1), inds=[j])
  else:
    inds_i = cond_inds.copy()
    inds_j = cond_inds.copy()
    inds_i.append(i)
    inds_j.append(j)
    ui = model.condCDF(len(cond_inds), x[:, cond_inds], inds_i)(x[:, i])
    uj = model.condCDF(len(cond_inds), x[:, cond_inds], inds_j)(x[:, j])

  return ui, uj


def ci_test(model, data, i, j, cond_inds):

  # Because pcalg.py uses sets
  cond_inds = list(cond_inds)

  # Compute ui/uj
  with t.no_grad():
    ui = []
    uj = []
    for batch_idx, batch in enumerate(data[0]):
      batch_data = batch[0].to(device).squeeze()
      ui_batch, uj_batch = get_condCDFs(model, batch_data, i, j, cond_inds)
      ui.append(ui_batch)
      uj.append(uj_batch)

  # Convert to numpy
  ui = t.cat(ui).cpu().numpy()
  uj = t.cat(uj).cpu().numpy()

  # Run the test
  test_ci_r = r['test_CI'](ui, uj)
  return [xx for xx in test_ci_r]