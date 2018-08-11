# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 下午8:59
# @Author  : Lei Mou
# @File    : test.py
# @Software: PyCharm

import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from unet.model import *
import scipy.misc as misc
from utils.accuracy import *

root = './DRIVE/'
test_path = './DRIVE/test/'


def load_data():
    """
    Load testing images
    :return: list of image names
    """
    test_images = []
    test_gt = []
    for i in range(1, 21):
        if i < 10:
            index = '0' + str(i)
        else:
            index = str(i)
        image_name = test_path + 'images/' + index + '_test.tif'
        gt_name = test_path + '1st_manual/' + index + '_manual1.gif'
        test_images.append(image_name)
        test_gt.append(gt_name)
    return test_images, test_gt


def load_net():
    """
    Load trained model.
        The model trained on multi-gpus
    :return:
    """
    net_dict = torch.load('./checkpoint/unet_1000_.pkl',
                          map_location=lambda storage, loc: storage).module.state_dict()
    # net_dict = torch.load('./checkpoint/unet_100_.pkl').module.state_dict()
    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(net_dict)
    return net


def save_pred(mask, binary=False, filename=''):
    """
    Save predicted mask to gif file, and return mask to future use
    :param mask: predicted mask
    :param binary:
    :param filename:
    :return: pred mask with shape of [512,512]
    """
    img = np.transpose(np.squeeze(mask, axis=0), [1, 2, 0])
    img = np.clip(img, 0, 255)
    img = np.squeeze(img, axis=-1)
    if filename:
        misc.imsave('assets/' + filename + '.gif', img)
    return img


def mask_overlay(image, mask):
    """
    Plot predicted mask on input image
    :param image: PIL image, also input image: [512, 512, 3]
    :param mask: predicted mask: [512, 512]
    :return: image array, BGR format to RGB format
    """
    img_array = np.asarray(image)
    img_array.setflags(write=1)
    w, h = img_array.shape[0], img_array.shape[1]
    for i in range(w):
        for j in range(h):
            if mask[i, j] > 128:
                img_array[i, j, 1] = 255
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


def resize(img):
    re_size = transforms.Resize((512, 512), interpolation=Image.BILINEAR)
    image = re_size(img)
    return image


def _test():
    """
    Model test
    :return: None
    """
    net = load_net()
    net.eval()
    images, gts = load_data()

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BILINEAR),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])

    acc = 0
    with torch.no_grad():
        for i in range(len('1')):
            print(images[i])
            img = Image.open(images[i])
            ground_truth = Image.open(gts[i])

            # 转换成数组，用于测试准确率
            ground_truth = resize(ground_truth)
            ground_truth = np.asarray(ground_truth)

            image = transform(img)
            image = image.unsqueeze_(0)

            pred = net(image)

            mask_to_overlay = save_pred(pred.data.cpu().numpy(), binary=False, filename=images[i][-11:-8] + 'pred')
            # 显示原图
            img = resize(img)
            img.save('assets/' + images[i][-11:])
            # 在原图上显示mask
            img_mask = mask_overlay(img, mask_to_overlay)
            cv2.imwrite('assets/' + images[i][-11:-8] + 'mask_over.tif', img_mask)

            acc += pixel_accuracy(mask_to_overlay, ground_truth)

        print("\n acc:\t{0:.4f}".format(acc / len('1')))


if __name__ == '__main__':
    _test()
