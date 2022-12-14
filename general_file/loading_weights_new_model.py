from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse, configparser
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from torch.optim import Adam as AdamW
from torch.optim.adamw import AdamW
from core.onecyclelr import OneCycleLR
from core import create_model
from core.extractor import BasicEncoder, BasicConvEncoder, Non_uniform_Encoder
from core.loss import compute_supervision_coarse, compute_coarse_loss, backwarp
from core.swin_transformer import POLAUpdate, MixAxialPOLAUpdate
import evaluate
import datasets.dataloader.datasets as datasets

from tensorboardX import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled=False):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremely large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def load_save_model(args):
    print(args.gpus)
    model = nn.DataParallel(create_model(args), device_ids=args.gpus)
    # print(model)
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)

    # feature_extractor_model = Non_uniform_Encoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
                            
    
    
    # model.module.fnet[0] = feature_extractor_model
    
    print(model)
    test_image = 255*torch.randn(1, 3, 64, 64)
    output = model(test_image,test_image.transpose(2,3))
    print("output:")
    # print(output.shape)
    #test simple backward pass
    
    #save state dict
    # torch.save(model.state_dict(), 'pretrained_models/new_model.pth')
    
    # model.cuda()
    # model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gmflownet', help="name of your experiment. The saved checkpoint will be named after this in `./checkpoints/.`")
    parser.add_argument('--model', default='gmflownet', help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt',default='pretrained_models/new_model.pth', help="restore checkpoint")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.set_num_threads(16)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('runs'):
        os.mkdir('runs')

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))
    args.gpus = [i for i in range(len(args.gpus))]
    print(args)
    # sys.exit(0)
    load_save_model(args)
