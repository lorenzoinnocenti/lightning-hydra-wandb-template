import segmentation_models_pytorch as smp
import torch.nn as nn


class UperNet(nn.Module):
    def __init__(self, dim_in=3, dim_out=1):
        super(UperNet, self).__init__()
        self.model = smp.UPerNet(
            encoder_name="tu-convnextv2_base",
            encoder_weights="imagenet",
            in_channels=dim_in,
            classes=dim_out,
            drop_path_rate=0.1,
            # activation=None,
        )

    def forward(self, x):
        x = self.model(x)
        