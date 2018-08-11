# -*- coding: utf-8 -*-
# @Time    : 2018/7/30 下午4:14
# @Author  : Lei Mou
# @File    : train.py
# @Software: PyCharm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from unet.model import UNet
from utils.data_load import DRIVEDataset
from eval import eval_net

root = './DRIVE/'

# ------------------------------------------------------------
epochs = 2000
lr = 0.001
snapshot = 500
batch_size = 8
# ------------------------------------------------------------

train_data = DRIVEDataset(root, transform=None, train=True)
data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)


def save_ckpt(net, iter):
    ckpt_path = './checkpoint/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    torch.save(net, ckpt_path + 'unet_' + str(iter) + '_.pkl')


def train():
    net = UNet(n_channels=3, n_classes=1)
    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=0.0005)

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.MSELoss().cuda()

    curr_epoch = 1
    for epoch in range(epochs):
        print("Start training...")
        print('Starting epoch {}/{}'.format(epoch + 1, epochs))

        epoch_loss = 0
        tmp = 0
        for idx, batch in enumerate(data_loader):
            image = batch[0].cuda()
            gt = batch[1].float().cuda()

            mask_pred = net(image)
            mask_pred = mask_pred.squeeze(1)
            mask_pred = mask_pred.clamp(0, 255)
            # print(mask_pred.size())
            # print(gt.size())

            loss = criterion(mask_pred, gt)

            epoch_loss += loss.item()

            print('{0:d} --- loss:{1:.4f}'.format(curr_epoch, loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            curr_epoch += 1
            tmp += 1
            if epoch % snapshot == 0 and epoch > 0:
                save_ckpt(net, epoch)

        print('    epoch loss: {}'.format(epoch_loss / tmp))
        # eval_dice = eval_net(net, data_loader)
        # print("Validation Dice Coeff:{}".format(eval_dice))
        print('-' * 100)

    save_ckpt(net, epochs)


if __name__ == '__main__':
    train()
