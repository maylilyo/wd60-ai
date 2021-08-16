import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from icecream import ic

from vimeo_dataset import Vimeo
from main_net import Main_net
from loss_f import LapLoss


current_dir = os.path.dirname(os.path.abspath(__file__))
vimeo_data_dir = os.path.join(current_dir, 'data/vimeo_triplet')

seed = 42
batch_size = 1
max_epochs = 100
num_workers = 0
crop = False
if crop:
    W = 256
    H = 256
else:
    W = 448
    H = 256
lr = 1e-4


def set_seed(seed):
    pl.seed_everything(seed)


def train():
    criteration = LapLoss()
    vimeo_dataset = Vimeo(base_dir=vimeo_data_dir)
    train_loader = DataLoader(
        dataset=vimeo_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    shape = [batch_size, 3, H, W]
    model = Main_net(shape)
    model = model.train(True)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=4e-4,
    )

    model = model.cuda()

    for step in range(max_epochs):
        total_loss = 0
        train_length = len(train_loader)
        for ix, data in enumerate(train_loader):
            img1, img2, tar = data

            img1 = img1.cuda()
            img2 = img2.cuda()
            tar = tar.cuda()

            img_out = model(img1, img2)
            # loss = torch.nn.functional.l1_loss(img_out,tar)
            loss = criteration(img_out, tar)

            t.zero_grad()

            loss.backward()
            optimizer.step()
            if ix % 5 == 0:
                print(f'Epoch: {ix:3d}/{train_length} - lr: {lr}')
                print(f'loss value: {loss.item()}')
            total_loss += loss
        print(f'epoch: {step:3d} avg loss: {total_loss.item()/train_length}')
        if (step+1) % 3 == 0:
            torch.save(model, f'./weights/model_weight_{step:02d}.pt')


if __name__ == '__main__':
    set_seed(seed)
    # train()
