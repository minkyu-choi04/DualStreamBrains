import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import numpy as np
import time
import torch.nn.functional as F
from torch.nn import init
import os
import argparse
import sys

import vgg_backbone as retina
import misc
import gaussian_mask as G
import neural_population as CW
import foveated_image as FI
import model_retina1d

parser = argparse.ArgumentParser(description='PyTorch Dual-Stream Model')
parser.add_argument('--density_ratio_V', default=15.0, type=float, help='Control parameter for transformation function. ')
parser.add_argument('--grid_size_V', default=64, type=int, help='sampling resolution, w == h')
parser.add_argument('--density_ratio_D', default=2.5, type=float, help='Control parameter for transformation function. ')
parser.add_argument('--grid_size_D', default=64, type=int, help='sampling resolution, w == h')
parser.add_argument('--actmap_s', default=16, type=int, help='actmap size')

parser.add_argument('--model_dir', default='./model_weights.pt', type=str, help='model weights')
parser.add_argument('--image_size_h', default=174, type=int, help='input resoultion height')
parser.add_argument('--image_size_w', default=336, type=int, help='input resoultion height')


def main():
    args = parser.parse_args()
    
    isFixedFixs = False 
    args.n_classes = 80
    pad_mod = 'reflect'
    args.useStocFixs = False
    args.useIOR = False
    args.useWindow = True
    train(args, model_dir, isFixedFixs, pad_mod)


def train(args, model_dir, isFixedFixs, pad_mod, isRandomFixs=False):
    ## Agent model
    model = model_retina1d.CRNN_Model(args, isFixedFixs, isRandomFixs)
    model.apply(misc.initialize_weight)

    model_weights = torch.load(model_dir)
    model.fc.load_state_dict(model_weights['fc'])
    model.gru.load_state_dict(model_weights['gru'])
    model.retina_net.load_state_dict(model_weights['retina_net'])
    model.init_hidden = torch.nn.Parameter(model_weights['init_hidden'])
    model.agent_net.load_state_dict(model_weights['agent_net'])


    ## Use GPU
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    import get_data as gd
    gd.get_data_HW(model, input_img_size=(args.image_size_h, args.image_size_w), pad_mod=pad_mod, useWindow=args.useWindow)
    return


if __name__ == "__main__":
    main()
