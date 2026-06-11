import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils_pack.utils import sum_pooling, print_model_parm_nums

def get_dataloader(args, datapath, dataset="TaxiBJ", batch_size=16, mode='train', task_id=0, scale_x=1, scale_y=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    sequence = ['P1', 'P2', 'P3', 'P4']

    ori_datapath = os.path.join(datapath, dataset)
    if mode == 'train':
        shuffle = True
        task = sequence[task_id - 1]
        print("# load {} datset {}".format(mode, task))
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)
    else:
        shuffle = False
        task = sequence[task_id - 1]
        if mode == 'test':
            print("# load {} datset {}".format(mode, task))
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)

    X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
    Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
    ext = np.load(os.path.join(datapath, 'ext.npy'))

    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

    if dataset in ['XiAn', 'Xi_an', 'CD', 'ChengDu']:
        if len(X.shape) == 3:
            X = Tensor(X).unsqueeze(1)
            Y = Tensor(Y).unsqueeze(1)
        else:
            X = Tensor(X)
            Y = Tensor(Y)
    else:
        X = Tensor(np.expand_dims(X, 1))
        Y = Tensor(np.expand_dims(Y, 1))

    ext = Tensor(ext)

    if scale_x != 1:
        X = sum_pooling(X, scale_x)

    if scale_y != 0:
        Y = sum_pooling(Y, scale_y)

    assert len(X) == len(Y)
    if mode != 'valid':
        print('# {} samples: {}'.format(mode, len(X)))

    data = TensorDataset(X, Y, ext)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_dataloader_joint(args, datapath, dataset="ChengDu", batch_size=16, mode='train', task_id=0):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = None
    Y = None
    ext = None

    sequence = ['P1']

    ori_datapath = os.path.join(datapath, dataset)
    if mode == 'train':
        shuffle = True
        for task in sequence[:task_id]:
            if task != sequence[task_id - 1]:
                for task_mode in ['train', 'valid', 'test']:
                    print("# load {} datset {}".format(task_mode, task))
                    datapath = os.path.join(ori_datapath, task)
                    datapath = os.path.join(datapath, task_mode)
                    if X is None:
                        X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
                        Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
                        ext = np.load(os.path.join(datapath, 'ext.npy'))

                    else:
                        X = np.concatenate([X, np.load(os.path.join(datapath, 'X.npy'))], axis=0) / args.scaler_X
                        Y = np.concatenate([Y, np.load(os.path.join(datapath, 'Y.npy'))], axis=0) / args.scaler_X
                        ext = np.concatenate([ext, np.load(os.path.join(datapath, 'ext.npy'))], axis=0)

            else:
                print("# load {} datset {}".format(mode, task))
                datapath = os.path.join(ori_datapath, task)
                datapath = os.path.join(datapath, mode)
                X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
                Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
                ext = np.load(os.path.join(datapath, 'ext.npy'))
    else:
        shuffle = False
        task = sequence[task_id - 1]
        datapath = os.path.join(ori_datapath, task)
        datapath = os.path.join(datapath, mode)
        X = np.load(os.path.join(datapath, 'X.npy')) / args.scaler_X
        Y = np.load(os.path.join(datapath, 'Y.npy')) / args.scaler_Y
        ext = np.load(os.path.join(datapath, 'ext.npy'))

    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")

    if dataset in ['XiAn', 'Xi_an', 'CD', 'ChengDu']:
        if len(X.shape) == 3:
            X = Tensor(X).unsqueeze(1)
            Y = Tensor(Y).unsqueeze(1)
        else:
            X = Tensor(X)
            Y = Tensor(Y)
    else:
        X = Tensor(np.expand_dims(X, 1))
        Y = Tensor(np.expand_dims(Y, 1))

    ext = Tensor(ext)

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))
    data = TensorDataset(X, Y, ext)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader
