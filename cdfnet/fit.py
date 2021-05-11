import torch as t
import torch.optim as optim
from cdfnet import models
from cdfnet import utils
from cdfnet import hessian_penalty_pytorch
import copy
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
  h_parser.add_argument(
      '--dataset',
      type=str,
      default='',
      choices=['', 'gas', 'bsds300', 'hepmass', 'miniboone', 'power'])
  # architecture
  h_parser.add_argument('--n', type=int, default=100)
  h_parser.add_argument('--m', type=int, default=5)
  h_parser.add_argument('--L', type=int, default=4)
  h_parser.add_argument('--use_HT', type=utils.str2bool, default=True)
  h_parser.add_argument('--HT_poolsize', type=int, default=2)
  h_parser.add_argument('--adaptive_coupling',
                        type=utils.str2bool,
                        default=True)
  # initialization
  h_parser.add_argument('--w_std', type=float, default=1.0)
  h_parser.add_argument('--b_std', type=float, default=0)
  h_parser.add_argument('--b_bias', type=float, default=0)
  h_parser.add_argument('--a_std', type=float, default=.1)
  # fitting
  h_parser.add_argument('--n_epochs', '-ne', type=int, default=10)
  h_parser.add_argument('--batch_size', '-b', type=int, default=100)
  h_parser.add_argument('--lambda_l2', type=float, default=0)
  h_parser.add_argument('--opt',
                        type=str,
                        default='adam',
                        choices=['adam', 'sgd'])
  h_parser.add_argument('--lr', type=float, default=5e-3)
  h_parser.add_argument('--patience', '-p', type=int, default=1000)
  h_parser.add_argument('--stable_nll_iters', type=int, default=5)
  h_parser.add_argument('--gaussian_noise', type=float, default=0)
  h_parser.add_argument('--subsample_inds', type=utils.str2bool, default=False)
  h_parser.add_argument('--n_inds_to_subsample', type=int, default=20)
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
  h_parser.add_argument('--val_every', '-ve', type=int, default=1000)
  h_parser.add_argument('--eval_test',
                        '-et',
                        type=utils.str2bool,
                        default=True)
  h_parser.add_argument('--test_every', '-te', type=int, default=1000)
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
  :return: The fitted model
  """
  n_iters = h.n_epochs * h.M // h.batch_size

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
    print('Saving tensorboard logs to ' + tb_path)

  model, optimizer, scheduler, step = initialize(h)
  model, optimizer, scheduler, step = load_checkpoint(model, optimizer,
                                                      scheduler, step,
                                                      h.model_to_load)

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
  if h.eval_validation:
    val_nll = eval_nll(model, val_loader)
  if h.use_HT and h.adaptive_coupling:
    set_adaptive_coupling(h, model, train_loader)
  clip_max_norm = 0
  inds = ...
  tic = time.time()
  for epoch in range(h.n_epochs):
    for batch_idx, batch in enumerate(train_loader):
      batch_data = batch[0].to(device)

      if h.subsample_inds:
        inds = t.randperm(h.d)[:h.n_inds_to_subsample]

      # update parameters
      for param in model.parameters():
        param.grad = None

      # use stabilized nll if needed
      nll_value = model.nll(batch_data).item()
      obj_type = 'nll'
      # obj_type = 'cdf_regression'
      if obj_type == 'cdf_regression':
        obj = model.cdf_regression_loss(batch_data[:, inds], inds=inds)
        obj_value = obj.item()
      else:
        if step < h.stable_nll_iters:
          #nll_value = model.nll(batch_data).item()
          obj = model.nll(batch_data[:, inds], inds=inds, stabilize=True)
        else:
          obj = model.nll(batch_data[:, inds], inds=inds)
          #nll_value = obj.item()

      obj.backward()

      if clip_max_norm > 0:
        t.nn.utils.clip_grad_value_(model.parameters(), clip_max_norm)

      optimizer.step()
      scheduler.step(obj)

      if h.eval_validation and step % h.val_every == 0:
        val_nll = eval_nll(model, val_loader)
        print(f'iteration {step}, validation nll: {val_nll:.4f}')
        if h.use_tb:
          writer.add_scalar('loss/validation', val_nll, step)

      if h.eval_test and step % h.test_every == 0:
        test_nll = eval_nll(model, test_loader)
        print(f'iteration {step}, test nll: {test_nll:.4f}')
        if h.use_tb:
          writer.add_scalar('loss/test', test_nll, step)

      if h.verbose and step % h.print_every == 0:

        print_str = f'iteration {step}, train nll: {nll_value:.4f}'

        # median NLL
        median_nll = model.median_nll(batch_data[:, inds], inds=inds).item()
        print_str += f', median nll: {median_nll:.4f}'
        if h.use_tb:
          writer.add_scalar('loss/median_nll', median_nll, step)

        print_str += f', obj value: {obj_value:.4f}'

        if h.eval_validation:
          print_str += f', val nll: {val_nll:.4f}'

        toc = time.time()
        print_str += f', elapsed: {toc - tic:.4f}, {h.print_every / (toc - tic):.4f} iterations per sec.'
        tic = time.time()
        print(print_str)

      if h.save_checkpoints and (step + 1) % h.checkpoint_every == 0:
        t.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': step
            }, save_path + '/checkpoint.pt')

      if h.use_tb:
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        writer.add_scalar('loss/train', nll_value, step)

      step += 1
      #if h.max_iters is not None and step == h.max_iters:
      if step == h.max_iters:
        print(f'Terminating after {h.max_iters} iterations.')
        return model

  if h.eval_test:
    test_nll = eval_nll(model, test_loader)
    if h.use_tb:
      writer.add_scalar('loss/test', test_nll, step)
  return model


def eval_nll(model, loader):
  with t.no_grad():
    val_nll = 0
    nll = model.nll
    for batch_idx, batch in enumerate(loader):
      batch_data = batch[0].to(device)
      val_nll += nll(batch_data).item()
  return val_nll / (batch_idx + 1)


def initialize(h):
  model = models.CDFNet(
      h.d,
      n=h.n,
      L=h.L,
      m=h.m,
      w_std=h.w_std,
      b_bias=h.b_bias,
      b_std=h.b_std,
      a_std=h.a_std,
      use_HT=h.use_HT,
      adaptive_coupling=h.adaptive_coupling,
      HT_poolsize=h.HT_poolsize,
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
  step = 0
  return model, optimizer, scheduler, step


def load_checkpoint(model, optimizer, scheduler, step, checkpoint_to_load):
  if checkpoint_to_load != '':
    print('Loading model..')
    checkpoint = t.load(checkpoint_to_load + '/checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    step = checkpoint['step']
  return model, optimizer, scheduler, step


def get_tb_path(h):
  fields = [
      'dataset', 'n', 'm', 'L', 'batch_size', 'n_epochs', 'lr', 'patience'
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


def set_adaptive_coupling(h, model, train_loader):
  n_batches = 10 * h.d**2 // h.batch_size + 1
  train_iter = iter(train_loader)
  batches = [next(train_iter)[0] for _ in range(n_batches)]
  model.create_adaptive_couplings(batches)
  print('Using adaptive variable coupling')
