import torch as t
import torch.optim as optim
from cdfnet import models
from cdfnet import utils
import copy
import argparse
import datetime
import json
import os
import time

if t.cuda.is_available():
  t.set_default_tensor_type('torch.cuda.FloatTensor')
else:
  print('No GPU found')
  t.set_default_tensor_type('torch.FloatTensor')


def get_default_h(parent=None):
  if parent is not None:
    h_parser = argparse.ArgumentParser(parents=[parent])
  else:
    h_parser = argparse.ArgumentParser()
  # data
  h_parser.add_argument('--d', type=int, default=2)
  h_parser.add_argument(
      '--dataset',
      type=str,
      default='',
      choices=['', 'gas', 'bsds300', 'hepmass', 'miniboone', 'power'])
  # architecture
  h_parser.add_argument('--type',
                        type=str,
                        default='CP1Net',
                        choices=['CP1Net', 'CDFNet'])
  h_parser.add_argument('--n', type=int, default=100)
  h_parser.add_argument('--r', type=int, default=100)
  h_parser.add_argument('--m', type=int, default=5)
  h_parser.add_argument('--l', type=int, default=4)
  # initialization
  h_parser.add_argument('--w_std', type=float, default=.1)
  h_parser.add_argument('--b_std', type=float, default=0)
  h_parser.add_argument('--b_bias', type=float, default=0)
  h_parser.add_argument('--a_std', type=float, default=.1)
  # fitting
  h_parser.add_argument('--epochs', '-ne', type=int, default=10)
  h_parser.add_argument('--batch_size', '-b', type=int, default=100)
  h_parser.add_argument('--l2', type=float, default=0)
  h_parser.add_argument('--opt',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'])
  h_parser.add_argument('--lr', type=float, default=5e-3)
  h_parser.add_argument('--patience', '-p', type=int, default=50)
  # logging
  h_parser.add_argument('--data_dir', type=str, default='data/data')
  h_parser.add_argument('--use_tb', type=utils.str2bool, default=False)
  h_parser.add_argument('--tb_dir', type=str, default='data/tb')
  h_parser.add_argument('--exp_name', type=str, default='')
  h_parser.add_argument('--model_to_load', '-mtl', type=str, default='')
  h_parser.add_argument('--set_detect_anomaly',
                        '-sde',
                        type=utils.str2bool,
                        default=False)
  h_parser.add_argument('--save_checkpoints',
                        '-sc',
                        type=utils.str2bool,
                        default=True)
  h_parser.add_argument('--save_path', type=str, default='data/checkpoint')
  h_parser.add_argument('--checkpoint_every', '-ce', type=int, default=200)
  h_parser.add_argument('--eval_validation', type=utils.str2bool, default=True)
  h_parser.add_argument('--val_every', '-ve', type=int, default=200)
  h_parser.add_argument('--eval_test',
                        '-et',
                        type=utils.str2bool,
                        default=True)
  h_parser.add_argument('--verbose', '-v', type=utils.str2bool, default=True)
  h_parser.add_argument('--print_every', '-pe', type=int, default=20)
  h_parser.add_argument('--max_iters', type=int, default=None)

  h = h_parser.parse_known_args()[0]
  return h


def fit_neural_copula(
    h,
    data,
):
  """
  :param h: hyperparameters in the form of an argparser
  :param data: A list of train, val and test dataloaders
  :return:
  """
  n_iters = h.epochs * h.M // h.batch_size
  print(h)
  print(f"Running {n_iters} iterations.")

  save_path = h.save_path
  if h.use_tb:
    if h.model_to_load != '':
      tb_path = h.model_to_load
    else:
      tb_path = get_tb_path(h)
      os.mkdir(tb_path)
      with open(tb_path + '/h.json', 'w') as f:
        json.dump(h.__dict__, f, indent=4, sort_keys=True)
    save_path = tb_path
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(tb_path)

  model, optimizer, scheduler, iter = initialize(h)
  model, optimizer, scheduler, iter = load_checkpoint(model, optimizer,
                                                      scheduler, iter,
                                                      h.model_to_load)
  # set up data loaders
  train_loader, val_loader, test_loader = data

  t.autograd.set_detect_anomaly(h.set_detect_anomaly)

  # fit neural copula to data
  if h.eval_validation:
    optimizer.zero_grad()
    val_nll = eval_nll(model, val_loader)
  clip_max_norm = 0
  tic = time.time()
  for epoch in range(h.epochs):
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

      # # add noise to small weights
      # with t.no_grad():
      #   small_a_inds = t.where(model.a_s[-1] < -1)[0]
      #   for param in model.parameters():
      #     if len(param.shape) > 1:
      #       # param.add_((t.abs(param) < 0.5 * t.mean(t.abs(param))) *
      #       #            t.std(param) * t.randn(param.size()) * 0.5)
      #       param[:, small_a_inds, ...].add_(
      #           t.std(param) * t.randn(param[:, small_a_inds, ...].size()) *
      #           0.1)
      #     else:
      #       # these are last layer weights
      #       param[small_a_inds] = 1

      if h.eval_validation and iter % h.val_every == 0:
        val_nll = eval_nll(model, val_loader)

      if h.verbose and iter % h.print_every == 0:
        print_str = f'iteration {iter}, train nll: {nll.cpu().detach().numpy():.4f}'
        if h.eval_validation:
          print_str += f', val nll: {val_nll:.4f}'

        toc = time.time()
        print_str += f', elapsed: {toc - tic:.4f}'
        tic = time.time()
        print(print_str)

      if h.save_checkpoints and (iter + 1) % h.checkpoint_every == 0:
        t.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iter': iter
            }, save_path + '/checkpoint.pt')

      if h.use_tb:
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter)
        writer.add_scalar('loss/train', nll.item(), iter)
        if h.eval_validation:
          writer.add_scalar('loss/validation', val_nll, iter)

      iter += 1
      if h.max_iters is not None and iter == h.max_iters:
        print(f'Terminating after {h.max_iters} iterations.')
        break

  if h.eval_test:
    test_nll = eval_nll(model, test_loader)
    if h.use_tb:
      writer.add_scalar('loss/test', test_nll, iter)


def eval_nll(model, loader):
  val_nll = 0
  for batch_idx, batch in enumerate(loader):
    batch_data = batch[0]
    val_nll += model.nll(batch_data).cpu().detach().numpy()
  return val_nll / (batch_idx + 1)


def initialize(h):
  if h.type == 'CDFNet':
    model = models.CDFNet(
        h.d,
        n=h.n,
        l=h.l,
        m=h.m,
        w_std=h.w_std,
        b_bias=h.b_bias,
        b_std=h.b_std,
        a_std=h.a_std,
    )
  elif h.type == 'CP1Net':
    model = models.CP1Net(
        h.d,
        n=h.n,
        r=h.r,
        l=h.l,
        m=h.m,
        w_std=h.w_std,
        b_bias=h.b_bias,
        b_std=h.b_std,
        a_std=h.a_std,
    )
  else:
    raise NameError

  if h.opt == 'adam':
    opt_type = optim.Adam
  elif h.opt == 'sgd':
    opt_type = optim.SGD
  else:
    raise NameError
  optimizer = opt_type(model.parameters(), lr=h.lr, weight_decay=h.l2)
  scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     verbose=True,
                                                     patience=h.patience,
                                                     min_lr=1e-5,
                                                     factor=0.5)
  # scheduler = t.optim.lr_scheduler.CyclicLR(optimizer,
  #                                           base_lr=1e-5,
  #                                           max_lr=0.1,
  #                                           cycle_momentum=False)
  iter = 0
  return model, optimizer, scheduler, iter


def load_checkpoint(model, optimizer, scheduler, iter, checkpoint_to_load):
  if checkpoint_to_load != '':
    print('Loading model..')
    checkpoint = t.load(checkpoint_to_load + '/checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    iter = checkpoint['iter']
  return model, optimizer, scheduler, iter


def get_tb_path(h):
  fields = [
      'dataset', 'type', 'n', 'r', 'm', 'l', 'batch_size', 'epochs', 'lr',
      'patience', 'l2'
  ]
  dt_str = str(datetime.datetime.now())[:-7].replace(' ',
                                                     '-').replace(':', '-')
  folder_name = [f'{utils.shorten(f)}:{h.__dict__[f]}'
                 for f in fields] + [dt_str]
  if h.exp_name != '':
    folder_name = [h.exp_name] + folder_name
  folder_name = '_'.join(folder_name)
  folder_name.replace('.', 'p')
  tb_path = h.tb_dir + '/' + folder_name
  return tb_path
