import lightning as li
from torch.utils.data import DataLoader
from os import sched_getaffinity
import torch


class DataModule(li.LightningDataModule):
    def __init__(
        self, 
        config, 
        batch_size=2,
        verbose=False,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.verbose = verbose        
        # check if running on a cpu or a gpu
        if torch.cuda.is_available():
            self.num_workers = len(sched_getaffinity(0))
        else:
            self.num_workers = 0

 
    def setup(self, stage):
        # TODO: implement setup for datasets
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1, # TODO: change to self.batch_size if no issues arise
            num_workers=1, # TODO: change to self.num_workers if no issues arise
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1, # TODO: change to self.batch_size if no issues arise
            num_workers=1, # TODO: change to self.num_workers if no issues arise
            shuffle=False,
        )