import os
import json
import argparse
import pprint
import numpy as np
import datetime
import torch
from torch.utils import data
from bnaf import *
from tqdm import tqdm
from optim.adam import Adam
from optim.lr_scheduler import ReduceLROnPlateau

from data.gas import GAS
#from data.bsds300 import BSDS300
from data.hepmass import HEPMASS
from data.miniboone import MINIBOONE
from data.power import POWER
from sklearn.impute import KNNImputer
import pandas as pd

NAF_PARAMS = {
    'power': (414213, 828258),
    'gas': (401741, 803226),
    'hepmass': (9272743, 18544268),
    'miniboone': (7487321, 14970256),
    'bsds300': (36759591, 73510236)
}


def load_dataset(args):
  if args.dataset == 'gas':
    dataset = GAS('/data/data/gas/ethylene_CO.pickle')
  elif args.dataset == 'bsds300':
    dataset = BSDS300('/data/data/BSDS300/BSDS300.hdf5')
  elif args.dataset == 'hepmass':
    dataset = HEPMASS('/data/data/hepmass')
  elif args.dataset == 'miniboone':
    dataset = MINIBOONE('/data/data/miniboone/data.npy')
  elif args.dataset == 'power':
    dataset = POWER('/data/data/power/data.npy')
  else:
    raise RuntimeError()

  if args.missing_data_pct > 0:
    # create missing data mask
    mask = np.random.rand(*dataset.trn.x.shape) > args.missing_data_pct

    if args.missing_data_strategy == 'drop':
      data = dataset.trn.x[np.where(np.product(mask, axis=1) == 1)[0], :]
      dataset_train = torch.utils.data.TensorDataset(
          torch.tensor(np.expand_dims(data, 1)).float().to(args.device))
    elif args.missing_data_strategy == 'mice':
      import miceforest as mf
      #
      # mice_data = np.load(
      #     f'/data/data/mice/gas_mice_{args.missing_data_pct}.npy')
      traindata = dataset.trn.x
      df = pd.DataFrame(data=traindata)
      data_amp = mf.ampute_data(df, perc=args.missing_data_pct)
      kdf = mf.KernelDataSet(data_amp, save_all_iterations=True)
      kdf.mice(3)
      completed_data = kdf.complete_data()
      mice_data = completed_data.to_numpy()
      print(
          f'Created dataset using MICE, missing proportion {args.missing_data_pct}'
      )

      # create tensordataset from tensor
      dataset_train = torch.utils.data.TensorDataset(
          torch.tensor(np.expand_dims(mice_data, 1)).float().to(args.device))
    elif args.missing_data_strategy == 'knn':
      traindata = dataset.trn.x
      mask = np.random.rand(*traindata.shape) < args.missing_data_pct
      # missing_traindata = traindata + mask * np.nan
      # imputer = KNNImputer(n_neighbors=2)
      # knn_data = imputer.fit_transform(missing_traindata)
      missing_traindata = traindata
      missing_traindata[np.where(mask)] = np.nan
      imputer = KNNImputer(n_neighbors=3)
      imputed = []
      for block in np.array_split(missing_traindata, 100):
        knn_data = imputer.fit_transform(block)
        imputed += [knn_data]
      all_imputed = np.concatenate(imputed)
      all_imputed = np.squeeze(all_imputed)

      print(
          f'Created dataset using KNN, missing proportion {args.missing_data_pct}'
      )
      dataset_train = torch.utils.data.TensorDataset(
          torch.tensor(np.expand_dims(all_imputed, 1)).float().to(args.device))
    else:
      # mean imputation
      data_and_mask = np.array([dataset.trn.x, mask]).swapaxes(0, 1)
      dataset_train = torch.utils.data.TensorDataset(
          torch.tensor(data_and_mask).float().to(args.device))

  else:
    dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(np.expand_dims(dataset.trn.x, 1)).float().to(args.device))

  # dataset_train = torch.utils.data.TensorDataset(
  #     torch.from_numpy(dataset.trn.x).float().to(args.device))
  data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                  batch_size=args.batch_dim,
                                                  shuffle=True)

  if args.missing_data_pct > 0:
    if args.missing_data_strategy == 'mice':
      valdata = dataset.val.x
      df = pd.DataFrame(data=valdata)
      data_amp = mf.ampute_data(df, perc=args.missing_data_pct)
      kdf = mf.KernelDataSet(data_amp, save_all_iterations=True)
      kdf.mice(3)
      completed_data = kdf.complete_data()
      mice_data = completed_data.to_numpy()
      print(
          f'Created dataset using MICE, missing proportion {args.missing_data_pct}'
      )
      dataset_valid = torch.utils.data.TensorDataset(
          torch.tensor(mice_data).float().to(args.device))
    elif args.missing_data_strategy == 'knn':
      valdata = dataset.val.x
      mask = np.random.rand(*valdata.shape) < args.missing_data_pct
      #missing_valdata = valdata + mask * np.nan
      missing_valdata = valdata
      missing_valdata[np.where(mask)] = np.nan
      imputer = KNNImputer(n_neighbors=3)
      imputed = []
      for block in np.array_split(missing_valdata, 100):
        knn_data = imputer.fit_transform(block)
        imputed += [knn_data]
        print('.')
      all_imputed = np.concatenate(imputed)
      all_imputed = np.squeeze(all_imputed)
      #knn_data = imputer.fit_transform(missing_valdata)
      print(
          f'Created dataset using KNN, missing proportion {args.missing_data_pct}'
      )
      dataset_valid = torch.utils.data.TensorDataset(
          torch.tensor(all_imputed).float().to(args.device))
  else:
    dataset_valid = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.val.x).float().to(args.device))
  data_loader_valid = torch.utils.data.DataLoader(dataset_valid,
                                                  batch_size=args.batch_dim,
                                                  shuffle=False)

  dataset_test = torch.utils.data.TensorDataset(
      torch.from_numpy(dataset.tst.x).float().to(args.device))
  data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size=args.batch_dim,
                                                 shuffle=False)

  args.n_dims = dataset.n_dims

  return data_loader_train, data_loader_valid, data_loader_test


def create_model(args, verbose=False):

  flows = []
  for f in range(args.flows):
    layers = []
    for _ in range(args.layers - 1):
      layers.append(
          MaskedWeight(args.n_dims * args.hidden_dim,
                       args.n_dims * args.hidden_dim,
                       dim=args.n_dims))
      layers.append(Tanh())

    flows.append(
        BNAF(*([MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims), Tanh()] + \
               layers + \
               [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims)]),\
             res=args.residual if f < args.flows - 1 else None
        )
    )

    if f < args.flows - 1:
      flows.append(Permutation(args.n_dims, 'flip'))

  model = Sequential(*flows).to(args.device)
  params = sum(
      (p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item()
      for p in model.parameters()).item()

  if verbose:
    print('{}'.format(model))
    print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(
        params, NAF_PARAMS[args.dataset][0] / params,
        NAF_PARAMS[args.dataset][1] / params, args.n_dims))

  if args.save and not args.load:
    with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
      print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(
          params, NAF_PARAMS[args.dataset][0] / params,
          NAF_PARAMS[args.dataset][1] / params, args.n_dims),
            file=f)

  return model


def save_model(model, optimizer, epoch, args):
  def f():
    return 0
    # if args.save:
    #   print('Saving model..')
    #   torch.save(
    #       {
    #           'model': model.state_dict(),
    #           'optimizer': optimizer.state_dict(),
    #           'epoch': epoch
    #       }, os.path.join(args.load or args.path, 'checkpoint.pt'))

  return f


def load_model(model, optimizer, args, load_start_epoch=False):
  def f():
    print('Loading model..')
    checkpoint = torch.load(
        os.path.join(args.load or args.path, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if load_start_epoch:
      args.start_epoch = checkpoint['epoch']

  return f


def compute_log_p_x(model, x_mb):
  y_mb, log_diag_j_mb = model(x_mb)
  log_p_y_mb = torch.distributions.Normal(
      torch.zeros_like(y_mb), torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
  return log_p_y_mb + log_diag_j_mb


def train(model, optimizer, scheduler, data_loader_train, data_loader_valid,
          data_loader_test, args):

  if args.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        os.path.join(args.tensorboard, args.load or args.path))

  epoch = args.start_epoch
  for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

    t = tqdm(data_loader_train, smoothing=0, ncols=80)
    train_loss = []

    for batch, in t:
      # if args.missing_data_pct > 0:
      #   x_mb = batch[:, 0, :]
      #   mask = batch[:, 1, :]
      #   if args.missing_data_strategy == 'drop':
      #     x_mb = x_mb[torch.where(torch.prod(mask, dim=1) == 1)[0], :]
      #   elif args.missing_data_strategy == 'mean_imputation':
      #     means = torch.mean(x_mb, dim=0)
      #     x_mb = x_mb * mask + means * (1 - mask)
      if args.missing_data_pct > 0 and args.missing_data_strategy == 'mean_imputation':
        x_mb = batch[:, 0, :]
        mask = batch[:, 1, :]
        means = torch.mean(x_mb, dim=0)
        x_mb = x_mb * mask + means * (1 - mask)
      else:
        x_mb = batch[:, 0, :]

      loss = -compute_log_p_x(model, x_mb).mean()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),
                                     max_norm=args.clip_norm)

      optimizer.step()
      optimizer.zero_grad()

      t.set_postfix(loss='{:.2f}'.format(loss.item()), refresh=False)
      train_loss.append(loss)

    train_loss = torch.stack(train_loss).mean()
    optimizer.swap()
    validation_loss = -torch.stack([
        compute_log_p_x(model, x_mb).mean().detach()
        for x_mb, in data_loader_valid
    ], -1).mean()
    test_loss = -torch.stack([
        compute_log_p_x(model, x_mb).mean().detach()
        for x_mb, in data_loader_test
    ], -1).mean()
    optimizer.swap()

    print(
        'Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f} -- test_loss: {:4.3f}'
        .format(epoch + 1, args.start_epoch + args.epochs, train_loss.item(),
                validation_loss.item(), test_loss.item()))

    with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
      print(
          'Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f} -- test_loss: {:4.3f}'
          .format(epoch + 1, args.start_epoch + args.epochs, train_loss.item(),
                  validation_loss.item(), test_loss.item()),
          file=f)

    stop = scheduler.step(validation_loss,
                          callback_best=save_model(model, optimizer, epoch + 1,
                                                   args),
                          callback_reduce=load_model(model, optimizer, args))

    if args.tensorboard:
      writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
      writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
      writer.add_scalar('loss/train', train_loss.item(), epoch + 1)
      writer.add_scalar('loss/test', test_loss.item(), epoch + 1)

    if stop:
      break

  #load_model(model, optimizer, args)()
  optimizer.swap()
  validation_loss = -torch.stack([
      compute_log_p_x(model, x_mb).mean().detach()
      for x_mb, in data_loader_valid
  ], -1).mean()
  test_loss = -torch.stack([
      compute_log_p_x(model, x_mb).mean().detach()
      for x_mb, in data_loader_test
  ], -1).mean()

  print('###### Stop training after {} epochs!'.format(epoch + 1))
  print('Validation loss: {:4.3f}'.format(validation_loss.item()))
  print('Test loss:       {:4.3f}'.format(test_loss.item()))

  #if args.save:
  with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    print('###### Stop training after {} epochs!'.format(epoch + 1), file=f)
    print('Validation loss: {:4.3f}'.format(validation_loss.item()), file=f)
    print('Test loss:       {:4.3f}'.format(test_loss.item()), file=f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument(
      '--dataset',
      type=str,
      default='power',
      choices=['gas', 'bsds300', 'hepmass', 'miniboone', 'power'])

  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument('--batch_dim', type=int, default=200)
  parser.add_argument('--clip_norm', type=float, default=0.1)
  parser.add_argument('--epochs', type=int, default=1000)

  parser.add_argument('--patience', type=int, default=20)
  parser.add_argument('--cooldown', type=int, default=10)
  parser.add_argument('--early_stopping', type=int, default=100)
  parser.add_argument('--decay', type=float, default=0.5)
  parser.add_argument('--min_lr', type=float, default=5e-4)
  parser.add_argument('--polyak', type=float, default=0.998)

  parser.add_argument('--flows', type=int, default=5)
  parser.add_argument('--layers', type=int, default=2)  # 2
  parser.add_argument('--hidden_dim', type=int, default=240)  # 240
  parser.add_argument('--residual',
                      type=str,
                      default='gated',
                      choices=[None, 'normal', 'gated'])

  parser.add_argument('--expname', type=str, default='')
  parser.add_argument('--load', type=str, default=None)
  parser.add_argument('--save', action='store_true', default=True)
  parser.add_argument('--tensorboard', type=str, default='tensorboard')
  parser.add_argument('--missing_data_pct', type=float, default=0.0)
  parser.add_argument('--missing_data_strategy', type=str, default='drop')

  args = parser.parse_args()

  print('Arguments:')
  pprint.pprint(args.__dict__)

  #args.path = './checkpoint'
  args.path = os.path.join(
      '/data/tb', '{}{}_layers{}_h{}_flows{}{}_mdp_{}_mds_{}_{}'.format(
          args.expname + ('_' if args.expname != '' else ''), args.dataset,
          args.layers, args.hidden_dim, args.flows,
          '_' + args.residual if args.residual else '', args.missing_data_pct,
          args.missing_data_strategy,
          str(datetime.datetime.now())[:-7].replace(' ',
                                                    '-').replace(':', '-')))

  print('Loading dataset..')
  data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

  if args.save and not args.load:
    print('Creating directory experiment..')
    os.mkdir(args.path)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
      json.dump(args.__dict__, f, indent=4, sort_keys=True)

  print('Creating BNAF model..')
  model = create_model(args, verbose=True)

  print('Creating optimizer..')
  optimizer = Adam(model.parameters(),
                   lr=args.learning_rate,
                   amsgrad=True,
                   polyak=args.polyak)

  print('Creating scheduler..')
  scheduler = ReduceLROnPlateau(optimizer,
                                factor=args.decay,
                                patience=args.patience,
                                cooldown=args.cooldown,
                                min_lr=args.min_lr,
                                verbose=True,
                                early_stopping=args.early_stopping,
                                threshold_mode='abs')

  args.start_epoch = 0
  if args.load:
    load_model(model, optimizer, args, load_start_epoch=True)()

  print('Training..')
  train(model, optimizer, scheduler, data_loader_train, data_loader_valid,
        data_loader_test, args)


if __name__ == '__main__':
  main()
