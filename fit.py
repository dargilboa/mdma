import torch as t
import torch.optim as optim
import models
import copy
import argparse
import datetime
import utils
import json

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')


def get_default_h():
  h_parser = argparse.ArgumentParser()
  # data
  h_parser.add_argument('--d', type=int, default=2)
  h_parser.add_argument('--dataset', type=str, default='')
  # architecture
  h_parser.add_argument('--n', type=int, default=100)
  h_parser.add_argument('--m', type=int, default=5)
  h_parser.add_argument('--L', type=int, default=4)
  # initialization
  h_parser.add_argument('--w_std', type=float, default=.1)
  h_parser.add_argument('--b_std', type=float, default=0)
  h_parser.add_argument('--b_bias', type=float, default=0)
  h_parser.add_argument('--a_std', type=float, default=.1)
  # fitting
  h_parser.add_argument('--n_epochs', type=int, default=10)
  h_parser.add_argument('--batch_size', type=int, default=100)
  h_parser.add_argument('--lambda_l2', type=float, default=1e-4)
  h_parser.add_argument('--opt',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'])
  h_parser.add_argument('--lr', type=int, default=5e-3)
  h_parser.add_argument('--patience', type=int, default=50)

  h = h_parser.parse_known_args()[0]
  return h


def fit_neural_copula(h,
                      data,
                      verbose=True,
                      print_every=20,
                      checkpoint_every=100,
                      val_every=20,
                      max_iters=float("inf"),
                      model_to_load=None,
                      use_tb=False,
                      tb_log_dir=None,
                      save_checkpoints=False,
                      eval_test=False,
                      eval_validation=True,
                      exp_name=''):
  """
  :param h: hyperparameters in the form of an argparser
  :param data: A list of train, val and test dataloaders
  :param verbose:
  :param print_every:
  :param checkpoint_every:
  :param val_every:
  :param max_iters:
  :param model_to_load:
  :return:
  """
  model = init_model(h, model_to_load)
  n_iters = h.n_epochs * h.M // h.batch_size
  print(h)
  print(f'Running {n_iters} iterations.')

  if use_tb:
    tb_path = get_tb_path(tb_log_dir, h, exp_name)
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(tb_path)
    with open(tb_path + '/h.json', 'w') as f:
      json.dump(h.__dict__, f, indent=4, sort_keys=True)
  if h.opt == 'adam':
    opt_type = optim.Adam
  elif h.opt == 'sgd':
    opt_type = optim.SGD
  else:
    raise NameError
  optimizer = opt_type(model.parameters(), lr=h.lr, weight_decay=h.lambda_l2)
  scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     verbose=True,
                                                     patience=h.patience,
                                                     eps=1e-5)

  # set up data loaders
  train_loader, val_loader, test_loader = data

  t.autograd.set_detect_anomaly(True)

  # fit neural copula to data
  iter = 0
  nlls = []
  val_nlls = []
  val_nll_iters = []
  checkpoints = []
  checkpoint_iters = []
  if eval_validation:
    val_nll = eval_nll(model, val_loader)
    best_val_nll = val_nll
    best_val_nll_model = copy.deepcopy(model)
  clip_max_norm = 0
  for epoch in range(h.n_epochs):
    for batch_idx, batch in enumerate(train_loader):
      batch_data = batch[0]

      # update parameters
      optimizer.zero_grad()

      nll = model.nll(batch_data)
      nll.backward()

      if clip_max_norm > 0:
        t.nn.utils.clip_grad_value_(model.parameters(), clip_max_norm)

      optimizer.step()
      scheduler.step(nll)

      nlls.append(nll.cpu().detach().numpy())

      if eval_validation and iter % val_every == 0:
        val_nll = eval_nll(model, val_loader)
        val_nlls.append(val_nll)
        val_nll_iters.append(iter)
        if val_nll < best_val_nll:
          best_val_nll = val_nll
          best_val_nll_model = copy.deepcopy(model)

      if verbose and iter % print_every == 0:
        print_str = f'iteration {iter}, train nll: {nll.cpu().detach().numpy():.4f}'
        if eval_validation:
          print_str += f', val nll: {val_nll:.4f}'
        print(print_str)

      if (iter + 1) % checkpoint_every == 0:
        checkpoints.append(copy.deepcopy(model))
        checkpoint_iters.append(iter)
        if save_checkpoints:
          t.save(
              {
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'iter': iter
              }, tb_path + '/checkpoint.pt')

      if use_tb:
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter)
        writer.add_scalar('loss/train', nll.item(), iter)
        if eval_validation:
          writer.add_scalar('loss/validation', val_nll, iter)

      iter += 1
      if iter > max_iters:
        break

  # collect outputs
  outs = {
      'nlls': nlls,
      'model': model,
      'h': h,
      'data': data,
      'checkpoints': checkpoints,
      'checkpoint_iters': checkpoint_iters,
  }

  if eval_test:
    test_nll = eval_nll(model, test_loader)
    outs['test_nll'] = test_nll
    if use_tb:
      writer.add_scalar('loss/test', test_nll, iter)

  if eval_validation:
    outs['val_nlls'] = val_nlls
    outs['val_nll_iters'] = val_nll_iters
    outs['best_val_nll_model'] = best_val_nll_model

  return outs


def eval_nll(model, loader):
  val_nll = 0
  for batch_idx, batch in enumerate(loader):
    batch_data = batch[0]
    val_nll += model.nll(batch_data, create_graph=False).cpu().detach().numpy()
  return val_nll / (batch_idx + 1)


def init_model(h, model_to_load):
  if model_to_load is not None:
    print('Loaded model')
    print(model_to_load)
    return copy.deepcopy(model_to_load)

  model = models.CDFNet(
      h.d,
      n=h.n,
      L=h.L,
      m=h.m,
      w_std=h.w_std,
      b_bias=h.b_bias,
      b_std=h.b_std,
      a_std=h.a_std,
  )
  return model


def get_tb_path(tb_log_dir, h, exp_name):
  fields = ['dataset', 'n', 'm', 'L', 'batch_size', 'lr', 'patience']
  dt_str = str(datetime.datetime.now())[:-7].replace(' ',
                                                     '-').replace(':', '-')
  folder_name = [f'{utils.shorten(f)}:{h.__dict__[f]}'
                 for f in fields] + [dt_str]
  if exp_name != '':
    folder_name = [exp_name] + folder_name
  folder_name = '_'.join(folder_name)
  folder_name.replace('.', 'p')
  tb_path = tb_log_dir + '/' + folder_name
  return tb_path
