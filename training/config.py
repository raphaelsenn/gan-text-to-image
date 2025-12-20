from dataclasses import dataclass
from typing import Tuple, Optional

import torch


class Preview:
    def __init__(self) -> None:
        raise NotImplementedError

    def preview(self) -> None:
        raise NotImplementedError


@dataclass
class TrainConfig:
    # Core training settings 
    dataset: str
    epochs: int = 200
    batch_size: int = 64

    lr: float = 0.0002
    lr_decay: float = 0.5
    decay_every: int = 100
    betas: Tuple[float, float] = (0.5, 0.999)

    save_every: int = 5
    eval_every: int = 1

    # Runtime settings
    device: torch.device = torch.device("cuda")
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    verbose: bool = True
    preview: bool = True

    # Text encoder settings 
    img_feat_dim: Optional[int] = None
    cnn_dim: Optional[int] = None
    emb_dim: Optional[int] = None
    context_length: Optional[int] = None
    average: bool = True
    grad_norm_clip: float = 5.0

    # GAN settings 
    nt: Optional[int] = None
    nz: Optional[int] = None
    emb_in: Optional[int] = None
    emb_out: Optional[int] = None
    ngf: Optional[int] = None
    ndf: Optional[int] = None
    interpolation_weight: float = 1.0