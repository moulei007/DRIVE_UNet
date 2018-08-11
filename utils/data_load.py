# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 上午9:21
# @Author  : Lei Mou
# @File    : data_load.py
# @Software: PyCharm
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')


# image_dir = 'images'
# groundtruth_dir = '1st_manual'

def load_dataset(root_dir, train=True):
    images = []
    groundtruth = []
    masks = []
    if train:
        sub_dir = 'training'
    else:
        sub_dir = 'test'
    images_path = os.path.join(root_dir, sub_dir, 'images')
    groundtruth_path = os.path.join(root_dir, sub_dir, '1st_manual')
    masks_path = os.path.join(root_dir, sub_dir, 'mask')

    for file in glob.glob(os.path.join(images_path, '*.tif')):
        image_name = os.path.basename(file)
        index_name = image_name[0:3]
        groundtruth_name = index_name + 'manual1.gif'
        masks_name = index_name + 'training_mask.gif'

        images.append(os.path.join(images_path, image_name))
        groundtruth.append(os.path.join(groundtruth_path, groundtruth_name))
        masks.append(os.path.join(masks_path, masks_name))

    return images, groundtruth


class DRIVEDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        :param root_dir: dataset root dir
        :param transform: wether to apply transform
        """
        self.root_dir = root_dir
        self.train = train
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(512),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]

        img = Image.open(img_path)
        gt = Image.open(gt_path)

        img = self.transform(img)
        gt = self.transform(gt)
        gt = np.squeeze(gt, axis=-1)

        return (img, gt)

    def __len__(self):
        return len(self.images)
