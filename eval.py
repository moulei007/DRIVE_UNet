# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 上午10:25
# @Author  : Lei Mou
# @File    : eval.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from utils.dice_loss import dice_coeff


def eval_net(net, dataset):
    """在一定迭代次数之后，对训练的模型进行测试"""
    total = 0
    for idx, batch in enumerate(dataset):
        image = batch[0]
        ground_truth = batch[1]

        # image and ground_truth are already torch.Tensors
        if torch.cuda.is_available():
            image = image.cuda()
            ground_truth = ground_truth.cuda()
            net = net.cuda()

        mask_pred = net(image)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        total += dice_coeff(mask_pred, ground_truth).item()
    return total / idx
