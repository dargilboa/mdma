import torch as t
import torch.optim as optim
from MDMA import models
from MDMA import utils
import argparse
import datetime
import json
import os
import time

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
  device = "cuda"
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')
  device = "cpu"


def get_default_h(parent=None):
  if parent is not None:
    h_parser = argparse.ArgumentParser(parents=[parent])
  else:
    h_parser = argparse.ArgumentParser()
  # data
  h_parser.add_argument('--d', type=int, default=2)
  h_parser.add_argument('--M', type=int, default=1000)
  h_parser.add_argument('--causal_mechanism', type=str, default='')
  h_parser.add_argument('--dataset', type=str, default='')
  h_parser.add_argument('--missing_data_pct', type=float, default=0.0)

  # architecture
  h_parser.add_argument('--m', type=int, default=1000)
  h_parser.add_argument('--r', type=int, default=3)
  h_parser.add_argument('--l', type=int, default=2)
  h_parser.add_argument('--use_HT', type=utils.str2bool, default=True)
  h_parser.add_argument('--use_MERA', type=utils.str2bool, default=False)
  h_parser.add_argument('--HT_poolsize', type=int, default=2)
  h_parser.add_argument('--adaptive_coupling',
                        type=utils.str2bool,
                        default=False)
  h_parser.add_argument('--mix_vars', type=utils.str2bool, default=False)
  h_parser.add_argument('--n_mix_terms', type=int, default=1)

  # initialization
  h_parser.add_argument('--w_std', type=float, default=1.0)
  h_parser.add_argument('--b_std', type=float, default=0)
  h_parser.add_argument('--b_bias', type=float, default=0)
  h_parser.add_argument('--a_std', type=float, default=.1)

  # fitting
  h_parser.add_argument('--n_epochs', '-ne', type=int, default=1000)
  h_parser.add_argument('--batch_size', '-b', type=int, default=500)
  h_parser.add_argument('--lambda_l2', type=float, default=0)
  h_parser.add_argument('--opt',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'])
  h_parser.add_argument('--lr', type=float, default=.01)
  h_parser.add_argument('--patience', '-p', type=int, default=30)
  h_parser.add_argument('--es_patience', '-esp', type=int, default=50)
  h_parser.add_argument('--stable_nll_iters', type=int, default=5)
  h_parser.add_argument('--gaussian_noise', type=float, default=0)
  h_parser.add_argument('--subsample_inds', type=utils.str2bool, default=False)
  h_parser.add_argument('--n_inds_to_subsample', type=int, default=20)
  h_parser.add_argument('--cdf_regularization',
                        type=utils.str2bool,
                        default=False)
  h_parser.add_argument('--hessian_regularization',
                        type=utils.str2bool,
                        default=False)

  # logging
  h_parser.add_argument('--data_dir', type=str, default='data')
  h_parser.add_argument('--use_tb', type=utils.str2bool, default=False)
  h_parser.add_argument('--tb_dir', type=str, default='data/tb')
  h_parser.add_argument('--exp_name', type=str, default='')
  h_parser.add_argument('--add_dt_str', type=utils.str2bool, default=True)
  h_parser.add_argument('--model_to_load', '-mtl', type=str, default='')
  h_parser.add_argument('--set_detect_anomaly',
                        '-sde',
                        type=utils.str2bool,
                        default=False)
  h_parser.add_argument('--save_checkpoints',
                        '-sc',
                        type=utils.str2bool,
                        default=False)
  h_parser.add_argument('--save_path', type=str, default='data/checkpoint')
  h_parser.add_argument('--eval_validation', type=utils.str2bool, default=True)
  h_parser.add_argument('--eval_test',
                        '-et',
                        type=utils.str2bool,
                        default=True)
  h_parser.add_argument('--verbose', '-v', type=utils.str2bool, default=True)
  h_parser.add_argument('--print_every', '-pe', type=int, default=20)
  h_parser.add_argument('--max_iters', type=int, default=None)
  h_parser.add_argument('--spot_instance', type=utils.str2bool, default=False)

  h = h_parser.parse_known_args()[0]
  return h


def fit_MDMA(
    h,
    data,
):
  """
  :param h: hyperparameters in the form of an argparser
  :param data: A list of train, val and test dataloaders
  :return: model: The fitted model
  """
  n_iters = h.n_epochs * h.M // h.batch_size

  save_path = h.save_path
  if h.use_tb:
    tb_path = get_tb_path(h)
    if not os.path.isdir(tb_path):
      os.mkdir(tb_path)
      with open(tb_path + '/h.json', 'w') as f:
        json.dump(h.__dict__, f, indent=4, sort_keys=True)
    save_path = tb_path
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(tb_path)
    print('Saving tensorboard logs to ' + tb_path)

  model, optimizer, scheduler, start_epoch, use_stable_nll = initialize(h)
  model, optimizer, scheduler, start_epoch, use_stable_nll = load_checkpoint(
      model, optimizer, scheduler, start_epoch, use_stable_nll, save_path)

  total_params = sum(p.numel() for p in model.parameters())
  print(
      f"Running {n_iters} iterations. Model contains {total_params} parameters."
  )
  h.total_params = total_params
  print(h.__dict__)

  # set up data loaders
  train_loader, val_loader, test_loader = data

  t.autograd.set_detect_anomaly(h.set_detect_anomaly)

  # fit MDMA
  if h.use_HT and h.adaptive_coupling:
    set_adaptive_coupling(h, model, train_loader)
  clip_max_norm = 0
  step = start_epoch * len(train_loader)
  inds = ...
  missing_data_mask = None
  use_stable_nll = True
  tic = time.time()
  es = utils.EarlyStopping(patience=h.es_patience)
  for epoch in range(start_epoch, h.n_epochs):
    for batch_idx, batch in enumerate(train_loader):
      batch_data = batch[0][:, 0, :].to(device)
      if h.missing_data_pct > 0:
        missing_data_mask = batch[0][:, 1, :].to(device)

      if h.subsample_inds:
        inds = t.randperm(h.d)[:h.n_inds_to_subsample]

      if step == h.stable_nll_iters:
        use_stable_nll = False

      for param in model.parameters():
        param.grad = None

      obj = model.nll(batch_data[:, inds],
                      inds=inds,
                      stabilize=use_stable_nll,
                      missing_data_mask=missing_data_mask)
      nll_value = obj.item()

      obj_value = obj.item()
      obj.backward()

      if clip_max_norm > 0:
        t.nn.utils.clip_grad_value_(model.parameters(), clip_max_norm)

      optimizer.step()
      if not h.eval_validation:
        scheduler.step(obj)

      if h.verbose and step % h.print_every == 0:

        print_str = f'Iteration {step}, train nll: {nll_value:.4f}'
        print_str += f', obj value: {obj_value:.4f}'

        # if h.eval_validation:
        #   print_str += f', val nll: {val_nll:.4f}'

        toc = time.time()
        print_str += f', elapsed: {toc - tic:.4f}, {h.print_every / (toc - tic):.4f} iterations per sec.'
        tic = time.time()
        print(print_str)

        if h.use_tb:
          writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
          writer.add_scalar('loss/train', nll_value, step)

      step += 1
      if step == h.max_iters:
        print(f'Terminating after {h.max_iters} iterations.')
        return model

    if h.save_checkpoints:
      cp_file = save_path + '/checkpoint.pt'
      print('Saving model to ' + cp_file)
      t.save(
          {
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'epoch': epoch + 1,
              'use_stable_nll': use_stable_nll
          }, cp_file)

    if h.eval_test:
      test_nll = eval_test(model, test_loader)
      print(f'Epoch {epoch}, test nll: {test_nll:.4f}')
      if h.use_tb:
        writer.add_scalar('loss/test', test_nll, epoch)

    if h.eval_validation:
      val_nll = eval_validation(model, val_loader, h)
      scheduler.step(val_nll)
      print(f'Epoch {epoch}, validation nll: {val_nll:.4f}')
      if h.use_tb:
        writer.add_scalar('loss/validation', val_nll, epoch)

      if es.step(val_nll):
        print('Early stopping criterion met, terminating.')
        return model

  return model


def eval_validation(model, loader, h):
  with t.no_grad():
    val_nll = 0
    nll = model.nll
    for batch_idx, batch in enumerate(loader):
      batch_data = batch[0][:, 0, :].to(device)
      if h.missing_data_pct > 0:
        missing_data_mask = batch[0][:, 1, :].to(device)
        val_nll += nll(batch_data, missing_data_mask=missing_data_mask).item()
      else:
        val_nll += nll(batch_data).item()
  return val_nll / (batch_idx + 1)


def eval_test(model, loader):
  with t.no_grad():
    test_nll = 0
    nll = model.nll
    for batch_idx, batch in enumerate(loader):
      batch_data = batch[0].to(device)
      test_nll += nll(batch_data).item()
  return test_nll / (batch_idx + 1)


def initialize(h):
  model = models.MDMA(
      h.d,
      m=h.m,
      l=h.l,
      r=h.r,
      w_std=h.w_std,
      b_bias=h.b_bias,
      b_std=h.b_std,
      a_std=h.a_std,
      use_HT=h.use_HT,
      use_MERA=h.use_MERA,
      adaptive_coupling=h.adaptive_coupling,
      HT_poolsize=h.HT_poolsize,
      mix_vars=h.mix_vars,
      n_mix_terms=h.n_mix_terms,
  )
  if h.opt == 'adam':
    opt_type = optim.Adam
  elif h.opt == 'sgd':
    opt_type = optim.SGD
  else:
    raise NameError
  optimizer = opt_type(model.parameters(),
                       lr=h.lr,
                       weight_decay=h.lambda_l2,
                       amsgrad=True)
  scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     verbose=True,
                                                     patience=h.patience,
                                                     min_lr=1e-4,
                                                     factor=0.5)
  start_epoch = 0
  use_stable_nll = True
  return model, optimizer, scheduler, start_epoch, use_stable_nll


def load_checkpoint(model, optimizer, scheduler, epoch, use_stable_nll,
                    save_path):
  cp_file = save_path + '/checkpoint.pt'
  if os.path.isfile(cp_file):
    print('Loading model from ' + cp_file)
    checkpoint = t.load(cp_file)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    use_stable_nll = checkpoint[
        'use_stable_nll']  # assumes # stable iters < # iters in 1 epoch
  else:
    print('No model to load.')
  return model, optimizer, scheduler, epoch, use_stable_nll


def get_tb_path(h):
  fields = [
      'dataset', 'm', 'r', 'l', 'mix_vars', 'batch_size', 'n_epochs', 'lr',
      'patience', 'missing_data_pct'
  ]
  folder_name = [f'{utils.shorten(f)}:{h.__dict__[f]}' for f in fields]
  if h.add_dt_str:
    dt_str = str(datetime.datetime.now())[:-7].replace(' ',
                                                       '-').replace(':', '-')
    folder_name += [dt_str]
  if h.exp_name != '':
    folder_name = [h.exp_name] + folder_name
  folder_name = '_'.join(folder_name)
  folder_name.replace('.', 'p')
  tb_path = h.tb_dir + '/' + folder_name
  return tb_path


def set_adaptive_coupling(h, model, train_loader):
  n_batches = 10 * h.d**2 // h.batch_size + 1
  train_iter = iter(train_loader)
  if h.missing_data_pct > 0:
    # multiply by mask
    batches = [t.prod(next(train_iter)[0], dim=1) for _ in range(n_batches)]
  else:
    batches = [next(train_iter)[0][:, 0, :] for _ in range(n_batches)]

  model.create_adaptive_couplings(batches)
  print('Using adaptive variable coupling')