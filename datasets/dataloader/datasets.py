# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2 as cv
import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import albumentations as A
import sys


class FlowDataset(data.Dataset):
    def __init__(self, aug_params):
        # self.augmentor = None
        # self.sparse = sparse
        self.augmentor = A.Compose([A.RandomCrop(width=aug_params['crop'][0], height=aug_params['crop'][1], p=1.0),
                                    A.HorizontalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    ])
        self.is_test = False
        self.is_validate = False
        self.init_seed = False
        self.data = []

    def __getitem__(self, index):
        # print('Index is {}'.format(index))
        # sys.stdout.flush()
        data = self.data[index]
        img0_path = data.image0
        img1_path = data.image1
        flow_path = data.flow

        img1 = cv.imread(img0_path)
        img2 = cv.imread(img1_path)
        flow, valid = frame_utils.readFlowKITTI(flow_path)
        flow = flow.astype(np.float32)
        flow[:, :, 0] = flow[:, :, 0] * 256
        flow[:, :, 1] = flow[:, :, 1] * 256
        # flow = flow[:,:,:2]
        # valid = flow[:, :, 2] > 1

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            # print(img1.shape,img2.shape, flow.shape, valid.shape)
            data = {"image": img1, "masks": [img2, flow, valid]}
            augmented = self.augmentor(**data)
            img1 = augmented["image"]
            img2 = augmented["masks"][0]
            flow = augmented["masks"][1]
            # print(flow.shape)
            valid = augmented["masks"][2]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        # print(flow.shape)
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.is_validate:
            return img1, img2, flow, valid.float(), img0_path, img1_path, flow_path
        else:
            return img1, img2, flow, valid.float()

    def __len__(self):
        return len(self.data)


class data_association():
    def __init__(self, image0, image1, flow):
        self.image0 = image0
        self.image1 = image1
        self.flow = flow


class Carla_Dataset(FlowDataset):
    def __init__(self, aug_params, split='training', root=r'C:\Users\Machine Learning GPU\Desktop\sushlok\new_approach\datasets\carla',
                 seq=[
                     "SoftRainNight",
                     "ClearNoon",
                     "CloudyNoon"
                 ],
                 setup_type=['camera_0', 'camera_-1', 'camera_1'], is_validate=False):
        super(Carla_Dataset, self).__init__(aug_params)
        if split == 'testing':
            self.is_test = True

        self.is_validate = is_validate
        self.data = []
        for s in seq:
            for t in setup_type:
                data_path = os.path.join(root, s, t)
                images = os.listdir(os.path.join(data_path, 'image_0'))
                print('ATTENTION: changes might be required in the data loader for the new dataset')
                for im in images:
                    img0 = os.path.join(data_path, 'image_0', im)
                    img1 = os.path.join(data_path, 'image_1', im)
                    flow = os.path.join(data_path, 'flow_data', im)
                    self.data.append(data_association(img0, img1, flow))

        # for debugging purposes
        with open('data_list.txt', 'w') as f:
            for i, image_pair_flow in enumerate(zip(self.image_list, self.flow_list)):
                image_pair = image_pair_flow[0]
                flow = image_pair_flow[1]
                f.write('{} {} {}\n'.format(i, image_pair, flow))


def fetch_dataloader(args, config=None, TRAIN_DS='C+T+K/S'):
    """ Create the data loader for the corresponding trainign set """

    # if args.stage == 'chairs':
    #     aug_params = {'crop_size': args.image_size,
    #                   'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
    #     train_dataset = FlyingChairs(aug_params, split='training')

    # elif args.stage == 'things':
    #     aug_params = {'crop_size': args.image_size,
    #                   'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
    #     clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
    #     final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
    #     train_dataset = clean_dataset + final_dataset

    # elif args.stage == 'sintel':
    #     print('Training Sintel Stage...')
    #     sys.stdout.flush()
    #     aug_params = {'crop_size': args.image_size,
    #                   'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    #     things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
    #     sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
    #     sintel_final = MpiSintel(aug_params, split='training', dstype='final')

    #     if TRAIN_DS == 'C+T+K+S+H':
    #         kitti = KITTI({'crop_size': args.image_size,
    #                       'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
    #         hd1k = HD1K({'crop_size': args.image_size,
    #                     'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
    #         train_dataset = 100*sintel_clean + 100 * \
    #             sintel_final + 200*kitti + 5*hd1k + things

    #     elif TRAIN_DS == 'C+T+K+H':
    #         hd1k = HD1K({'crop_size': args.image_size,
    #                     'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
    #         train_dataset = 100*sintel_clean + 100*sintel_final + 5*hd1k + things

    #     elif TRAIN_DS == 'C+T+S+K':
    #         kitti = KITTI({'crop_size': args.image_size,
    #                       'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
    #         train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + things

    #     elif TRAIN_DS == 'C+T+K/S':
    #         train_dataset = 100*sintel_clean + 100*sintel_final + things

    # elif args.stage == 'kitti':
    #     aug_params = {'crop_size': args.image_size,
    #                   'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    #     train_dataset = KITTI(aug_params, split='training')

    # elif args.stage == 'vkitti':
    #     aug_params = {'crop_size': args.image_size,
    #                   'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
    #     train_dataset = VirtualKITTI(aug_params, split='training', seq=[
    #                                  "0001", "0002", "0003"], setup_type=["fog", "morning"])

    if args.stage == 'carla':
        if config is not None:
            # aug_params = {'crop_size': config.image_size,
            #               'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            # , 'camera_1','camera_2','camera_3', 'camera_4'
            train_dataset = Carla_Dataset(split='training', seq=args.seq_list, setup_type=args.setup_list)
        else:
            # aug_params = {'crop_size': args.image_size,
            #               'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            # , "camera_-1", "camera_1", "camera_-10", "camera_10", "camera_-11", "camera_11", "camera_12", "camera_-2", "camera_2",
            train_dataset = Carla_Dataset(
                split='training', seq=["ClearNoon"],
                setup_type=["camera_0", "camera_-1", "camera_1", "camera_-10", "camera_10", "camera_-11", "camera_11", "camera_12",
                            "camera_-2", "camera_2", "camera_-3", "camera_3", "camera_-4", "camera_4", "camera_-5", "camera_5", "camera_-6",
                            "camera_6", "camera_-7", "camera_7", "camera_-8", "camera_8", "camera_-9", "camera_9"])
            # "camera_-3", "camera_3", "camera_-4", "camera_4", "camera_-5", "camera_5", "camera_-6", "camera_6", "camera_-7", "camera_7", "camera_-8", "camera_8", "camera_-9", "camera_9"])  # , 'camera_1','camera_2','camera_3', 'camera_4'

    if config is not None:
        train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                       pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    else:
        print("me")
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


# class MpiSintel(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='./datasets/Sintel', dstype='clean', is_validate=False):
#         super(MpiSintel, self).__init__(aug_params)
#         flow_root = osp.join(root, split, 'flow')
#         image_root = osp.join(root, split, dstype)

#         self.is_validate = is_validate
#         if split == 'test':
#             self.is_test = True

#         for scene in os.listdir(image_root):
#             image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
#             for i in range(len(image_list)-1):
#                 self.image_list += [[image_list[i], image_list[i+1]]]
#                 self.extra_info += [(scene, i)]  # scene and frame_id

#             if split != 'test':
#                 self.flow_list += sorted(glob(osp.join(flow_root,
#                                          scene, '*.flo')))


# class FlyingChairs(FlowDataset):
#     def __init__(self, aug_params=None, split='train', root='./datasets/FlyingChairs_release/data'):
#         super(FlyingChairs, self).__init__(aug_params)

#         images = sorted(glob(osp.join(root, '*.ppm')))
#         flows = sorted(glob(osp.join(root, '*.flo')))
#         assert (len(images)//2 == len(flows))

#         split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
#         for i in range(len(flows)):
#             xid = split_list[i]
#             if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
#                 self.flow_list += [flows[i]]
#                 self.image_list += [[images[2*i], images[2*i+1]]]


# class FlyingThings3D(FlowDataset):
#     def __init__(self, aug_params=None, root='./datasets/FlyingThings3D', dstype='frames_cleanpass'):
#         super(FlyingThings3D, self).__init__(aug_params)

#         for cam in ['left']:
#             for direction in ['into_future', 'into_past']:
#                 image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
#                 image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

#                 flow_dirs = sorted(
#                     glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
#                 flow_dirs = sorted([osp.join(f, direction, cam)
#                                    for f in flow_dirs])

#                 for idir, fdir in zip(image_dirs, flow_dirs):
#                     images = sorted(glob(osp.join(idir, '*.png')))
#                     flows = sorted(glob(osp.join(fdir, '*.pfm')))
#                     for i in range(len(flows)-1):
#                         if direction == 'into_future':
#                             self.image_list += [[images[i], images[i+1]]]
#                             self.flow_list += [flows[i]]
#                         elif direction == 'into_past':
#                             self.image_list += [[images[i+1], images[i]]]
#                             self.flow_list += [flows[i+1]]


# class KITTI(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='../dataset/data_scene_flow'):
#         super(KITTI, self).__init__(aug_params, sparse=True)
#         if split == 'testing':
#             self.is_test = True

#         root = osp.join(root, split)
#         images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
#         images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

#         for img1, img2 in zip(images1, images2):
#             frame_id = img1.split('/')[-1]
#             self.extra_info += [[frame_id]]
#             self.image_list += [[img1, img2]]

#         if split == 'training':
#             self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


# class HD1K(FlowDataset):
#     def __init__(self, aug_params=None, root='./datasets/hd1k_full_package'):
#         super(HD1K, self).__init__(aug_params, sparse=True)

#         seq_ix = 0
#         while 1:
#             flows = sorted(
#                 glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
#             images = sorted(
#                 glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

#             if len(flows) == 0:
#                 break

#             for i in range(len(flows)-1):
#                 self.flow_list += [flows[i]]
#                 self.image_list += [[images[i], images[i+1]]]

#             seq_ix += 1


# class VirtualKITTI(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='../datasets/vkitti', seq=["0001", "0002", "0003"], setup_type=["fog", "morning"], is_validate=False):
#         super(VirtualKITTI, self).__init__(aug_params, sparse=True)
#         if split == 'testing':
#             self.is_test = True

#         self.is_validate = is_validate
#         image_dirs = []
#         # datasets/vkitti/vkitti_1.3.1_rgb
#         for s in seq:
#             for t in setup_type:
#                 image_dirs += sorted(glob(osp.join(root,
#                                      'vkitti_1.3.1_rgb', '%s' % (s), '%s/*.png' % (t))))

#         for i in range(len(image_dirs)-1):
#             img1 = image_dirs[i]
#             img2 = image_dirs[i+1]
#             self.image_list += [[img1, img2]]
#             self.extra_info += [[img1.split('/')[-1]]]

#         if split == 'training':
#             for s in seq:
#                 for t in setup_type:
#                     self.flow_list += sorted(
#                         glob(osp.join(root, 'vkitti_1.3.1_flogt', '%s' % (s), '%s/*.png' % (t))))
