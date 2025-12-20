import torch
import torch.nn as nn
from torchvision import models


class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.GoogLeNet_Weights.DEFAULT 
        base = models.googlenet(weights=weights)

        # [N, 3, 64, 64] -> [N, 1024]
        self.encoder = nn.Sequential(
            base.conv1,
            base.maxpool1,
            base.conv2,
            base.conv3,
            base.maxpool2,
            base.inception3a,
            base.inception3b,
            base.maxpool3,
            base.inception4a, 
            base.inception4b, 
            base.inception4c, 
            base.inception4d, 
            base.inception4e, 
            base.maxpool4,
            base.inception5a,
            base.inception5b,
            base.avgpool,
            torch.nn.Flatten(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)