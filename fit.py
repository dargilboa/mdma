import torch as t
import torch.optim as optim
import models
import copy
from torch.utils.data import TensorDataset, DataLoader

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.DoubleTensor')


def fit_neural_copula(data,
                      h,
                      verbose=True,
                      print_every=20,
                      checkpoint_every=100,
                      val_every=20,
                      max_iters=float("inf")):
  # h: dictionary of hyperparameters
  # default hyperparameters
  default_h = {
      # architecture
      'n': 200,
      'n_m': 5,
      'L_m': 4,
      'fit_marginals': False,
      # initialization
      'b_std': 0.01,
      'w_std': 0.01,
      'a_std': 0.01,
      # fitting
      'n_epochs': 10,
      'batch_size': 100,
      'lambda_l2': 1e-4,
      'lambda_l2_m': 0,
      # 'lambda_hess_full': 0,
      # 'lambda_hess_diag': 0,
      # 'lambda_ent': 0,
      #'clip_max_norm': 0,
      'decrease_lr_time': 1,
      'decrease_lr_factor': 0.1,
      #'update_z_every': 1,
      #'bp_through_z_update': False,
      'opt': 'adam',
      'lr': 5e-3,
  }

  # merge h and default_h, overriding values in default_h with those in h
  h = {**default_h, **h}

  if h['fit_marginals']:
    model = models.SklarNet(h['d'],
                            n=h['n'],
                            n_m=h['n_m'],
                            L_m=h['L_m'],
                            b_bias=0,
                            b_std=h['b_std'],
                            w_std=h['w_std'],
                            a_std=h['a_std'],
                            z_update_samples=h['batch_size'])
  else:
    model = models.CopNet(h['d'],
                          n=h['n'],
                          b_bias=0,
                          b_std=h['b_std'],
                          w_std=h['w_std'],
                          a_std=h['a_std'],
                          z_update_samples=h['batch_size'])

  train_data, val_data = data
  tr_hess_sq_0, hess_norm_sq_0 = init_regularizers(model, h, train_data)

  # optimizer
  if h['opt'] == 'adam':
    opt_type = optim.Adam
  elif h['opt'] == 'sgd':
    opt_type = optim.SGD
  else:
    raise NameError
  optimizer = opt_type(model.parameters(), lr=h['lr'])
  n_iters = h['n_epochs'] * h['M'] // h['batch_size']
  scheduler = t.optim.lr_scheduler.StepLR(optimizer,
                                          step_size=int(h['decrease_lr_time'] *
                                                        n_iters),
                                          gamma=h['decrease_lr_factor'])

  # set up data loader
  if type(train_data) is not t.Tensor:
    train_data = t.Tensor(train_data)
    val_data = t.Tensor(val_data)
  train_dataset = TensorDataset(train_data)
  train_loader = DataLoader(train_dataset, batch_size=h['batch_size'])
  val_dataset = TensorDataset(val_data)
  val_loader = DataLoader(val_dataset, batch_size=h['batch_size'])

  t.autograd.set_detect_anomaly(True)

  # fit neural copula to data
  iter = 0
  nlls = []
  val_nlls = []
  val_nll_iters = []
  checkpoints = []
  checkpoint_iters = []
  val_nll = eval_val_nll(model, val_loader)
  best_val_nll = val_nll
  best_val_nll_model = copy.deepcopy(model)
  clip_max_norm = 0
  update_z_every = 1
  bp_through_z_update = False
  print(h)
  print(f'Running {n_iters} iterations.')
  for epoch in range(h['n_epochs']):
    for batch_idx, batch in enumerate(train_loader):
      # update parameters
      batch_data = batch[0]
      optimizer.zero_grad()
      nll = model.nll(batch_data)
      obj = nll

      obj = add_regularization(model, obj, h, batch_data, tr_hess_sq_0,
                               hess_norm_sq_0)

      obj.backward()

      if clip_max_norm > 0:
        t.nn.utils.clip_grad_value_(model.parameters(), clip_max_norm)

      optimizer.step()
      scheduler.step()

      # update z approximation, can also take the data as input
      if iter % update_z_every == 0:
        model.update_zs(data=batch_data,
                        bp_through_z_update=bp_through_z_update,
                        fit_marginals=h['fit_marginals'])

      nlls.append(nll.cpu().detach().numpy())

      if iter % val_every == 0:
        val_nll = eval_val_nll(model, val_loader)
        val_nlls.append(val_nll.cpu().detach().numpy())
        val_nll_iters.append(iter)
        if val_nll < best_val_nll:
          best_val_nll = val_nll.detach().clone()
          best_val_nll_model = copy.deepcopy(model)

      if verbose and iter % print_every == 0:
        print('iteration {}, train nll: {:.4f}, val nll: {:.4f}'.format(
            iter,
            nll.cpu().detach().numpy(),
            val_nll.cpu().detach().numpy(),
        ))

      if (iter + 1) % checkpoint_every == 0:
        checkpoints.append(copy.deepcopy(model))
        checkpoint_iters.append(iter)

      iter += 1
      if iter > max_iters:
        break

  # collect outputs
  outs = {
      'nlls': nlls,
      'val_nlls': val_nlls,
      'val_nll_iters': val_nll_iters,
      'model': model,
      'h': h,
      'data': data,
      'checkpoints': checkpoints,
      'checkpoint_iters': checkpoint_iters,
      'best_val_nll_model': best_val_nll_model
  }

  return outs


def eval_val_nll(model, val_loader):
  val_nll = 0
  for batch_idx, batch in enumerate(val_loader):
    batch_data = batch[0]
    val_nll += model.nll(batch_data)
  return val_nll / (batch_idx + 1)


def add_regularization(model, obj, h, batch_data, tr_hess_sq_0,
                       hess_norm_sq_0):
  # add regularization
  L2 = (t.norm(model.w)**2 + t.norm(model.a)**2 + t.norm(model.b)**2)
  obj += h['lambda_l2'] * L2

  if h['fit_marginals']:
    L2_m = sum([t.norm(w) ** 2 for w in model.w_ms]) + \
           sum([t.norm(a) ** 2 for a in model.a_ms]) + \
           sum([t.norm(b) ** 2 for b in model.b_ms])
    obj += h['lambda_l2_m'] * L2_m

  # if h['lambda_ent'] > 0:
  #   samples = model.c_sample(h['M'], n_bisect_iter=25)
  #   ent = model.nll(samples)
  #   obj = obj - h['lambda_ent'] * ent
  #
  # if h['lambda_hess_diag'] > 0:
  #   tr_hess_sq = t.mean(model.diag_hess(batch_data)**2) / tr_hess_sq_0
  #   obj += h['lambda_hess_diag'] * tr_hess_sq
  #
  # if h['lambda_hess_full'] > 0:
  #   hess_norm_sq = model.hess(batch_data).norm()**2 / hess_norm_sq_0
  #   obj += h['lambda_hess_full'] * hess_norm_sq

  return obj


def init_regularizers(model, h, train_data):
  tr_hess_sq_0 = None
  hess_norm_sq_0 = None
  # with t.no_grad():
  #   if h['lambda_hess_diag'] > 0:
  #     tr_hess_sq_0 = t.mean(model.diag_hess(train_data)**2)
  #   if h['lambda_hess_full'] > 0:
  #     hess_norm_sq_0 = model.hess(train_data).norm()**2
  return tr_hess_sq_0, hess_norm_sq_0
