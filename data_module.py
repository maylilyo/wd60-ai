# Standard

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Custom
from softcode.vimeo_dataset import Vimeo


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        vimeo_dir,
        batch_size=1,
        num_workers=0,
    ):
        super().__init__()
        self.vimeo_dir = vimeo_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.set_datasets()

    def set_datasets(self):
        self.train_dataset = Vimeo(base_dir=self.vimeo_dir, state='train')  # length: 51313
        self.valid_dataset = Vimeo(base_dir=self.vimeo_dir, state='test')  # length: 3783
        self.test_dataset = self.valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
