# Standard

# PIP
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, StepLR
import pytorch_lightning as pl

# Custom
from softcode.main_net import Main_net as CustomModel
import helper.loss as c_loss
import helper.lr_scheduler as lr_scheduler


class CustomModule(pl.LightningModule):
    def __init__(
        self,
        model_option,
        max_epochs,
        learning_rate=1e-2,
        criterion_name='RMSE',
        optimizer_name='Adam',
        lr_scheduler_name='StepLR',
        momentum=0.9,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.model = CustomModel(model_option)

        self.criterion = self.get_loss_function(criterion_name)
        self.metric_function = self.get_loss_function('SSIM')
        self.optimizer = self.get_optimizer(optimizer_name)
        self.lr_scheduler = self.get_lr_scheduler(lr_scheduler_name)

    @staticmethod
    def get_loss_function(loss_function_name):
        name = loss_function_name.lower()

        if name == 'RMSE'.lower():
            return c_loss.RMSELoss()
        elif name == 'MSE'.lower():
            return nn.MSELoss()
        elif name == 'MAE'.lower():
            return nn.L1Loss()
        elif name == 'CrossEntropy'.lower():
            return nn.CrossEntropyLoss()
        elif name == 'BCE'.lower():
            return nn.BCEWithLogitsLoss()
        elif name == 'LapLoss'.lower():
            return c_loss.LapLoss()
        elif name == 'SSIM'.lower():
            return c_loss.SSIM()

        raise ValueError(f'{loss_function_name} is not on the custom criterion list!')

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def get_lr_scheduler(self, scheduler_name):
        name = scheduler_name.lower()

        if name == 'OneCycleLR'.lower():
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.max_epochs * 2297,
                anneal_strategy='cos',
            )
        elif name == 'CosineAnnealingWarmUpRestarts'.lower():
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-9

            return lr_scheduler.CosineAnnealingWarmUpRestarts(
                optimizer=self.optimizer,
                T_0=10,
                T_mult=2,
                eta_max=self.learning_rate,
                T_up=5,
                gamma=0.5,
            )
        elif name == 'StepLR'.lower():
            return StepLR(
                optimizer=self.optimizer,
                step_size=20,
                gamma=0.5,
            )

        raise ValueError(f'{scheduler_name} is not on the custom scheduler list!')

    def forward(self, img1, img2):
        # img1: (batch_size, ???)
        # img2: (batch_size, ???)

        out = self.model(img1, img2)
        # out : (batch_size, ???)

        return out

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }

    def common_step(self, batch, state):
        img1, img2, y = batch
        # img1: (batch_size, ???)
        # img2: (batch_size, ???)
        # y: (batch_size, ???)

        y_hat = self(img1, img2)
        # y_hat: (batch_size, ???)

        loss = self.criterion(y_hat, y)
        loss /= len(y)

        metric = self.metric_function(y_hat, y)

        return loss, metric

    def training_step(self, batch, batch_idx):

        loss, metric = self.common_step(batch, state='train')

        self.log('train_loss', loss)
        self.log('train_ssim', metric)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, metric = self.common_step(batch, state='valid')

        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_ssim', metric, sync_dist=True)

    def test_step(self, batch, batch_idx):

        loss, metric = self.common_step(batch, state='test')

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_ssim', metric, sync_dist=True)
