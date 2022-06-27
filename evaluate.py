import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import datasets.dataloader.datasets as datasets
from utils import flow_viz
from utils import frame_utils

# from raft import RAFT, RAFT_Transformer
from core import create_model
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    #print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, warm_start=False):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, is_validate=True)
        epe_list = []

        flow_prev, sequence_prev = None, None
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            sequence_prev = sequence

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        #print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    #print("completed dataloading")
    out_list, epe_list = [], []
    flow_estimation_time =[]
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        start = time.time()
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()
        flow_estimation_time.append(end - start)
        flow = padder.unpad(flow_pr[0]).cpu()
        flow2 = (flow.numpy().transpose(1,2,0) + 512)*64
        flow2_kitti_format = np.zeros([flow2.shape[0],flow2.shape[1],3])
        flow2_kitti_format[:,:,2] = flow2[:,:,0]
        flow2_kitti_format[:,:,1] = flow2[:,:,1]
        flow2_kitti_format[:,:,0] = np.ones([flow2.shape[0],flow2.shape[1]]).reshape(flow2_kitti_format[:,:,0].shape)
        flow2_kitti_format = flow2_kitti_format.astype(np.uint16)
        cv.imwrite('result/gmflownet/'+ str(val_id).zfill(6) + "_10.png",flow2_kitti_format)
        # #print(flow2_kitti_format.shape)
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    data = {"total_time" : flow_estimation_time}
    df = pd.DataFrame.from_dict(data)
    #print("time:\n", df.describe())
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    #print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_vkitti(model, iters=24,setup = ["fog"],output_dir = "result/vkitti/"):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.VirtualKITTI(split='training', seq= ["0001"], setup_type = setup, is_validate=True)
    #print("completed dataloading")
    out_list, epe_list = [], []
    flow_estimation_time =[]
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        start = time.time()
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()
        flow_estimation_time.append(end - start)
        flow = padder.unpad(flow_pr[0]).cpu()
        flow = flow.numpy().transpose(1,2,0)
        flow_kitti_format = np.zeros([flow.shape[0],flow.shape[1],3])
        flow_kitti_format[:,:,2] = (flow[:,:,0]/(flow.shape[1] - 1) + 1)*((2**16 - 1.0)/2)
        flow_kitti_format[:,:,1] = (flow[:,:,1]/(flow.shape[0] - 1) + 1)*((2**16 - 1.0)/2)
        flow_kitti_format[:,:,0] = np.ones([flow.shape[0],flow.shape[1]]).reshape(flow_kitti_format[:,:,0].shape)*(2**16 - 1.0)
        flow_kitti_format = flow_kitti_format.astype(np.uint16)
        cv.imwrite(output_dir + str(val_id).zfill(5) + ".png",flow_kitti_format)
        # #print(flow2_kitti_format.shape)
        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # mag = torch.sum(flow_gt**2, dim=0).sqrt()

        # epe = epe.view(-1)
        # mag = mag.view(-1)
        # val = valid_gt.view(-1) >= 0.5

        # out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        # epe_list.append(epe[val].mean().item())
        # out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)
    data = {"total_time" : flow_estimation_time}
    df = pd.DataFrame.from_dict(data)
    #print("time:\n", df.describe())
    # epe = np.mean(epe_list)
    # f1 = 100 * np.mean(out_list)

    # #print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_carla(model, iters=24,setup = ["fog"],output_dir = "result/vkitti/"):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.Carla_Dataset(split='training', seq= ["SoftRainNight"], setup_type = setup, is_validate=True)
    #print("completed dataloading")
    out_list, epe_list = [], []
    flow_estimation_time =[]
    #print(len(val_dataset))
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, extra_info = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        print(extra_info)
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        start = time.time()
        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        end = time.time()
        flow_estimation_time.append(end - start)
        flow = padder.unpad(flow_pr[0]).cpu()
        flow = flow.numpy().transpose(1,2,0)
        # print(extra_info)
        np.savez(output_dir + extra_info[0].split(".")[0] + "." + extra_info[0].split(".")[1] + ".npz",flow = flow)
        flow_kitti_format = np.zeros([flow.shape[0],flow.shape[1],3])
        flow_kitti_format[:,:,2] = (flow[:,:,0]/(flow.shape[1] - 1) + 1)*((2**16 - 1.0)/2)
        flow_kitti_format[:,:,1] = (flow[:,:,1]/(flow.shape[0] - 1) + 1)*((2**16 - 1.0)/2)
        flow_kitti_format[:,:,0] = np.ones([flow.shape[0],flow.shape[1]]).reshape(flow_kitti_format[:,:,0].shape)*(2**16 - 1.0)
        flow_kitti_format = flow_kitti_format.astype(np.uint16)
        cv.imwrite(output_dir + extra_info[0] ,flow_kitti_format)
        # #print(flow2_kitti_format.shape)
        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # mag = torch.sum(flow_gt**2, dim=0).sqrt()

        # epe = epe.view(-1)
        # mag = mag.view(-1)
        # val = valid_gt.view(-1) >= 0.5

        # out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        # epe_list.append(epe[val].mean().item())
        # out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)
    data = {"total_time" : flow_estimation_time}
    df = pd.DataFrame.from_dict(data)
    #print("time:\n", df.describe())
    # epe = np.mean(epe_list)
    # f1 = 100 * np.mean(out_list)

    # #print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti-epe': epe, 'kitti-f1': f1}




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gmflownet', help="mdoel class. `<args.model>`_model.py should be in ./core and `<args.model>Model` should be defined in this file")
    parser.add_argument('--ckpt', help="restored checkpoint")

    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--use_mix_attn', action='store_true', help='use mixture of POLA and axial attentions')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(create_model(args))
    model.load_state_dict(torch.load(args.ckpt), strict=True)

    model.cuda()
    model.eval()

    temp_seq = sorted(os.listdir("/home/sushlok/new_approach/datasets/carla/ClearNoon"))
    seq = []
    for i in temp_seq:
        if(i[:3] == "rgb"):
            seq.append(i[4:])
    #print(seq)
    for i in seq:
        #print(i)
        if not os.path.exists("result/SoftRainNight/flow_" + i + "/"):
            os.makedirs(("result/SoftRainNight/flow_" + i + "/"))
        list_seq = []
        list_seq.append(i)
        # create_sintel_submission(model.module, warm_start=True)
        # create_kitti_submission(model.module)

        with torch.no_grad():
            if args.dataset == 'chairs':
                validate_chairs(model.module)

            elif args.dataset == 'sintel':
                validate_sintel(model.module)

            elif args.dataset == 'sintel_test':
                create_sintel_submission(model.module)

            elif args.dataset == 'kitti':
                validate_kitti(model.module)
                
            elif args.dataset == 'vkitti':
                validate_vkitti(model.module,setup= list_seq, output_dir = "result/SoftRainNight/flow_" + i + "/")
            
            elif args.dataset == 'carla':
                validate_carla(model.module,setup= list_seq, output_dir = "result/SoftRainNight/flow_" + i + "/")

            elif args.dataset == 'kitti_test':
                create_kitti_submission(model.module)
