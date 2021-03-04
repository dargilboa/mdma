import torch as t
import torch.optim as optim
import models
import copy

def fit_neural_copula(data, h, verbose=True, print_every=20, checkpoint_every=100):
  # h: dictionary of hyperparameters
  # default hyperparameters
  default_h = {
              'n': 200,
              'M_val': 500,
              'n_iters': 600,
              'update_z_every': 1,
              'b_std': 0.01,
              'W_std': 0.01,
              'a_std': 0.01,
              'lambda_l2': 1e-4,
              'lambda_H_full': 0,
              'lambda_H_diag': 0,
              'lambda_ent': 0,
              'clip_max_norm': 0,
              'decrease_lr_time': 1,
              'decrease_lr_factor': 0.1,
              'bp_through_z_update': False,
              'opt': 'adam',
              'lr': 5e-3,
              }

  # merge h and default_h, overriding values in default_h with those in h
  h = {**default_h, **h}

  C = models.CopNet(h['n'], h['d'], b_bias=0, b_std=h['b_std'], W_std=h['W_std'], a_std=h['a_std'],
                    z_update_samples=h['M'])
  train_data, val_data = data

  with t.no_grad():
    if h['lambda_H_diag'] > 0:
      tr_H_sq_0 = t.mean(C.diag_H(train_data) ** 2)
    if h['lambda_H_full'] > 0:
      H_norm_sq_0 = C.H(train_data).norm() ** 2

  if h['opt'] == 'adam':
    optimizer = optim.Adam(C.parameters(), lr=h['lr'])
  elif h['opt'] == 'sgd':
    optimizer = optim.SGD(C.parameters(), lr=h['lr'])
  else:
    raise NameError

  scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=int(h['decrease_lr_time']*h['n_iters']),
                                          gamma=h['decrease_lr_factor'])

  t.autograd.set_detect_anomaly(True)

  # fit neural copula to data
  NLLs = []
  val_NLLs = []
  checkpoints = []
  checkpoint_iters = []
  best_val_NLL = t.tensor(float("Inf"))
  print(h)
  for i in range(h['n_iters']):
    # update parameters
    optimizer.zero_grad()
    NLL = C.NLL(train_data)
    val_NLL = C.NLL(val_data)
    obj = NLL

    # regularization
    L2 = (t.norm(C.W) ** 2 + t.norm(C.a) ** 2 + t.norm(C.b) ** 2)
    obj += h['lambda_l2'] * L2

    if h['lambda_ent'] > 0:
      samples = C.sample(h['M'], n_bisect_iter=25)
      ent = C.NLL(samples)
      obj = obj - h['lambda_ent'] * ent

    if h['lambda_H_diag'] > 0:
      tr_H_sq = t.mean(C.diag_H(train_data) ** 2) / tr_H_sq_0
      obj += h['lambda_H_diag'] * tr_H_sq

    if h['lambda_H_full'] > 0:
      H_norm_sq = C.H(train_data).norm() ** 2 / H_norm_sq_0
      obj += h['lambda_H_full'] * H_norm_sq

    obj.backward()

    if h['clip_max_norm'] > 0:
      t.nn.utils.clip_grad_value_(C.parameters(), h['clip_max_norm'])

    optimizer.step()
    scheduler.step()

    # update z approximation, can also take the data as input
    if i % h['update_z_every'] == 0:
      C.update_zs(bp_through_z_update=h['bp_through_z_update'])

    NLLs.append(NLL.cpu().detach().numpy())
    val_NLLs.append(val_NLL.cpu().detach().numpy())
    if val_NLL < best_val_NLL:
      best_val_NLL = val_NLL.detach().clone()
      best_val_NLL_model = copy.deepcopy(C)

    if verbose and i % print_every == 0:
      print('iteration {}, train NLL: {:.4f}, val NLL: {:.4f}'
            .format(i,
                    NLL.cpu().detach().numpy(),
                    val_NLL.cpu().detach().numpy(),
                    ))

    if (i + 1) % checkpoint_every == 0:
      checkpoints.append(copy.deepcopy(C))
      checkpoint_iters.append(i)

    if i > 300 and NLL > 0:
      print('fitting unstable, terminating')
      break

  outs = {'NLLs': NLLs, 'val_NLLs': val_NLLs, 'model': C, 'h': h, 'data': data, 'checkpoints': checkpoints,
          'checkpoint_iters': checkpoint_iters, 'best_val_NLL_model': best_val_NLL_model}

  return outs
