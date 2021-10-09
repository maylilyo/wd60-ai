from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from vimeo_dataset import Vimeo
from main_net import Main_net
from loss_f import LapLoss


PROJECT_DIR = Path(__file__).absolute().parent.parent
WEIGHT_DIR = PROJECT_DIR / 'weights'
DATA_DIR = PROJECT_DIR / 'data'
VIMEO_DIR = DATA_DIR / 'vimeo_triplet'

seed = 42
batch_size = 8
max_epochs = 10
num_workers = 0
crop = False
lr = 1e-4

if crop:
    width = 256
    height = 256
else:
    width = 448
    height = 256

model_option = {
    'shape': [height, width],
}


def set_seed(seed):
    pl.seed_everything(seed)


def train():
    criteration = LapLoss()
    vimeo_dataset = Vimeo(base_dir=VIMEO_DIR)
    train_loader = DataLoader(
        dataset=vimeo_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = Main_net(model_option)
    model = model.train(True)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=4e-4,
    )

    model = model.cuda()

    for epoch in range(max_epochs):
        total_loss = 0
        with tqdm(train_loader) as pbar:
            for data in pbar:
                img1, img2, tar = data

                img1 = img1.cuda()
                img2 = img2.cuda()
                tar = tar.cuda()

                img_out = model(img1, img2)
                # loss = torch.nn.functional.l1_loss(img_out,tar)
                loss = criteration(img_out, tar)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                pbar.set_postfix({'train_loss': loss.item()})
                total_loss += loss
        print(f'Epoch: {epoch:02d} loss: {total_loss.item() / len(train_loader)}')
        if (epoch+1) % 3 == 0:
            torch.save(model, WEIGHT_DIR / f'epoch_{epoch:02d}.pt')


if __name__ == '__main__':
    set_seed(seed)
    train()
