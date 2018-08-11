# -*- coding: utf-8 -*-
# @Time    : 2018/8/1 下午4:52
# @Author  : Lei Mou
# @File    : augumentation.py
# @Software: PyCharm

"""
This script is designed to enhance data
"""
import os
import glob
from PIL import Image
import numpy as np


def image_flip(iamge_path):
    for file in glob.glob(os.path.join(iamge_path, '*.*')):
        img_name = os.path.basename(file)
        index_name = img_name[:-4]
        ext = img_name[-4:]
        img = Image.open(file)

        tmp1_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        tmp2_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tmp3_img = img.transpose(Image.ROTATE_90)

        img1_save_name = os.path.join(iamge_path, index_name + '_flip_tb' + str(ext))
        img2_save_name = os.path.join(iamge_path, index_name + '_flip_lr' + str(ext))
        img3_save_name = os.path.join(iamge_path, index_name + '_rotate_90' + str(ext))

        tmp1_img.save(img1_save_name)
        tmp2_img.save(img2_save_name)
        tmp3_img.save(img3_save_name)


def image_crop(image_path):
    for file in glob.glob(os.path.join(image_path, '*.*')):
        img_name = os.path.basename(file)
        index_name = img_name[:-4]
        ext = img_name[-4:]
        img = Image.open(file)

        w, h = img.size
        center_x, center_y = int(w / 2), int(h / 2)
        len_w, len_h = int(w / 3), int(h / 3)
        size_w, size_h = int(w / 4), int(h / 4)
        centers = [[center_x, center_y],
                   [center_x - len_w, center_y - len_h],
                   [center_x, center_y - len_h],
                   [center_x + len_w, center_y - len_h],
                   [center_x - len_w, center_y],
                   [center_x + len_w, center_y],
                   [center_x - len_w, center_y + len_h],
                   [center_x, center_y + len_h],
                   [center_x + len_w, center_y + len_h]]
        for i in range(len(centers)):
            x, y = centers[i]
            box = (x - size_w, y - size_h, x + size_w, y + size_h)
            region = img.crop(box)
            region = region.resize((w, h))
            region_name = os.path.join(image_path, index_name + '_crop_' + str(i) + str(ext))
            region.save(region_name)


if __name__ == '__main__':
    root = '../DRIVE/'
    args = ['training', 'test']
    for arg in args:
        image_path = os.path.join(root, arg, 'images')
        gt_path = os.path.join(root, arg, '1st_manual')
        mask_path = os.path.join(root, arg, 'mask')
        # start augumentation
        print("starting flip :{}".format(image_path))
        image_flip(image_path)
        print('starting flip :{}'.format(gt_path))
        image_flip(gt_path)
        # image_flip(mask_path)

        print('starting crop :{}'.format(image_path))
        image_crop(image_path)
        print('starting crop :{}'.format(gt_path))
        image_crop(gt_path)
