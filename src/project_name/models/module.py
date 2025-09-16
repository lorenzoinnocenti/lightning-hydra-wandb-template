import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import optim
from project_name.models.upernet import UperNet
import rasterio
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from pathlib import Path
import threading


model_dict = {
    "upernet": UperNet,
    # add other models here
}

class Module(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        jobid: str,
    ):
        super().__init__()
        self.lr = config['min_lr']
        self.size = config['size']
        self.stride = config['stride']
        self.val_step_outputs = [] # needed for val_epoch_end to work
        self.oem_batch_fetched = False
        self.oem_batch = None
        self.oem_loc_pred = None
        self.oem_dam_pred = None
        self.sysu_batch_fetched = False
        self.sysu_batch = None
        self.sysu_loc_pred = None
        self.sysu_dam_pred = None
        self.config = config
        self.batch_size = config['batch_size']
        self.test_plot_count = 0
        self.val_in_cpu = True 
        self.loss = # TODO: define loss function
        self.model = model_dict[config['name']]()
        self.threads = []
        self.jobid = jobid
        self.plot_path = f"logs/{jobid}/plots/"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        # TODO: implement training step
        return loss

    def test_step(self, batch: dict, batch_idx):
        # TODO: implement test step
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        thread = threading.Thread(
            target=self.plot, 
            args=(
                batch, 
                pred,
                True, 
                False, 
                self.test_plot_count, 
                batch['dataset'],
            )
        )
        thread.start()
        self.threads.append(thread)
        self.test_plot_count += 1
        return torch.tensor(0.0)

    def validation_step(self, batch, batch_idx):
        # TODO: implement validation step
        if not self.batch_fetched:
            # TODO: store one batch for plotting
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, batch_size=pre.size(0))
        return loss

    def on_test_epoch_end(self):
        for thread in self.threads:
            thread.join()
        self.threads = []
        
    def configure_optimizers(self):
        if self.config.lr_schedule == 'lin':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
            return {
                "optimizer": optimizer,
            }
        elif self.config.lr_schedule == 'rop':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=self.lr_decay_patience)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        elif self.config.lr_schedule == 'cos':
            # cosine annealing scheduler
            final_div_factor = self.config.max_lr / self.config.min_lr
            optimizer = optim.AdamW(self.parameters(), lr=self.config.min_lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.config.max_lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.,
                final_div_factor=final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.config.lr_schedule == 'exp':
            #check if max_lr is more than min_lr
            if self.config.max_lr < self.config.min_lr:
                raise ValueError("max_lr must be greater than min_lr for exponential decay")
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            return {
                "optimizer": optimizer,
            }

    def on_train_epoch_start(self):
        # TODO: move self.loss to same device as model it it has weights
        if self.config.lr_schedule == 'exp':
            # adjust lr
            min_lr = self.config.min_lr
            max_lr = self.config.max_lr
            max_epochs = self.config.max_epochs
            # same but for the exponent 
            min_exp = np.log2(min_lr)
            max_exp = np.log2(max_lr)
            exp = min_exp + (max_exp - min_exp) * (1 - self.current_epoch / max_epochs)
            lr = 2**exp
            self.trainer.optimizers[0].param_groups[0]['lr'] = lr     
   
    def on_validation_epoch_end(self):
        epoch_number = self.trainer.current_epoch
        # plot the fetched batch 
        if self.batch_fetched:
            thread = threading.Thread(
                target=self.plot, 
                args=(
                    self.batch, 
                    self.pred,
                    False, 
                    True, 
                    epoch_number,
                )
            )
            thread.start()
            self.batch_fetched = False

    def plot(self, batch, loc_pred, dam_pred, plot_raster=False, plot_logger=False, epoch_number=None, suffix=None):
        # TODO: implement plotting function
        print("plot number ", epoch_number, " done")
            
    def overlapping_predictions(self, pre, post, size, stride, val_in_cpu=False, verbose=False):
        # cat pre e post
        input = torch.cat([pre, post], dim=1)
        # store original input and target shape
        device = pre.device
        tile_list = []
        coords_list = []
        # create a cosine window
        han = torch.from_numpy(np.hanning(size))
        window = torch.outer(han, han) + 1e-6
        # compute the padding so that it is both bigger than stride and the padded image is divisible by stride
        pad_left = size//2
        pad_right = size//2 + stride + ((input.shape[2]) % stride)  # FIXME: probably could be reduced
        pad_top = size//2
        pad_bottom = size//2 + stride + ((input.shape[2]) % stride)
        # pad the image and label, the image has 3 channels
        input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        input = input.squeeze(0)
        # iterate over the image
        for i in range(0, input.shape[1]-size, stride):
            for j in range(0, input.shape[2]-size, stride):
                tile = input[:, i:i+size, j:j+size]
                tile = tile.cpu()
                # append the tile to the list
                tile_list.append(tile)
                coords_list.append((i, j))
        input = input.cpu()
        # initiate canvas as zero tensor with dimension dim_out x input.shape[1] x input.shape[2]
        canvas = torch.zeros(3, input.shape[1], input.shape[2])
        weight_canvas = torch.zeros(input.shape[1], input.shape[2])
        if verbose:
            loop = tqdm(range(0, len(tile_list), self.batch_size))
        else:
            loop = range(0, len(tile_list), self.batch_size)
        for k in loop:
            tiles = tile_list[k:k+self.batch_size]
            tiles = torch.stack(tiles)
            tiles = tiles.to(device)
            pred = self(tiles)
            pred = pred.cpu()
            # for each prediction, multiply by the window and sum to the pred_canvas
            for l, _ in enumerate(pred):
                i, j = coords_list[k+l]
                canvas[:, i:i+size, j:j+size] += pred[l] * window
                weight_canvas[i:i+size, j:j+size] += window
        # divide the pred_canvas by the weight_canvas
        canvas = canvas / weight_canvas
        # crop the pred_canvas to the original size
        canvas = canvas[:, pad_top:-pad_bottom, pad_left:-pad_right]
        # expand the pred_canvas to 4 dimensions
        canvas = canvas.unsqueeze(0)
        if not val_in_cpu:
            # to device
            canvas = canvas.to(device)
        return canvas
