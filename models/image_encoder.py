import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, emb_dim)

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(img_features)