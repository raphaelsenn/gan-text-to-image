import os
import textwrap
import random
from typing import Any

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils.utils import filter_ascii


# =============================================================================
# Preview settings
# =============================================================================

ROWS: int = 3
COLS: int = 9
DPI: int = 100

N_IMAGES = ROWS * COLS
W, H = 64, 64 

TEXT_WIDTH = 25
TEXT_HEIGHT = 110
SCALE: float = 5.0

FIGSIZE = ((COLS * W + TEXT_WIDTH) * SCALE / DPI, (ROWS * H + TEXT_HEIGHT) * SCALE / DPI)

# =============================================================================
# GAN-Preview
# =============================================================================


class Preview: 
    def __init__(
            self,
            cfg: Any,
            generator: nn.Module, 
            dataset: Any,
            indices: list[int],
            mode: str,  # cls or int
            k_imgs_per_class: int=5 
    ) -> None:
        
        if not hasattr(cfg, "dataset"):
            raise AttributeError("Config must have attribute `dataset: str`")
        if not hasattr(cfg, "device"):
            raise AttributeError("Config must have attribute `device`")
        if not hasattr(cfg, "nz"):
            raise AttributeError("Config must have attribute `nz`")

        self.cfg = cfg 
        self.generator = generator
        self.mode = mode
        self.dataset = dataset
        self.indices = indices
        self.k_imgs_per_class = k_imgs_per_class

        self._prepare_preview()

    @torch.no_grad()
    def _prepare_preview(self) -> None:
        cfg = self.cfg
        save_dir = f"{cfg.dataset}-preview" 

        os.makedirs(save_dir, exist_ok=True) 

        indices = random.sample(self.indices, k=N_IMAGES)
        device = cfg.device
        nz = cfg.nz

        captions_lst, embeddings_lst = [], []
        for id in indices:
            if self.mode == "cls":
                _, _, _, text_str, _ = self.dataset[id]
                text_emb = self.dataset.get_all_embeddings(id) 
            elif self.mode == "int-cls" or self.mode == "cls-int": 
                _, _, _, text_str, _ = self.dataset[id]
                text_emb = self.dataset.get_all_embeddings(id) 
            else:
                _, _, text_str, _ = self.dataset[id]
                text_emb = self.dataset.get_all_embeddings(id) 

            text_emb = text_emb.mean(0)

            embeddings_lst.append(text_emb)
            captions_lst.append(filter_ascii(text_str))

        self.noise = torch.randn(size=(N_IMAGES, nz, 1, 1), device=device)
        self.embeddings = torch.stack(embeddings_lst, dim=0).to(device)
        self.captions = captions_lst 
        self.save_dir = save_dir 

    @torch.no_grad()
    def preview(self, epoch: int) -> None:
        generator = self.generator
        generator.eval()

        captions = self.captions
        noise = self.noise
        text = self.embeddings
        
        # Generate fake images 
        fake_imgs = generator(noise, text)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(ROWS, COLS, dpi=DPI, figsize=FIGSIZE)
        axes = axes.flatten()
        
        for ax, fake_img, caption in zip(axes, fake_imgs, captions):
            fake_img = fake_img.detach().cpu().numpy().transpose(1, 2, 0)
            fake_img = ((fake_img + 1) / 2).clip(0, 1)

            wrapped = "\n".join(textwrap.wrap(caption, width=TEXT_WIDTH))

            ax.imshow(fake_img)
            ax.axis("off")
            ax.set_title(wrapped)

        save_dir = self.save_dir
        save_name = f"{self.cfg.dataset}_{epoch}"
        out_path = os.path.join(save_dir, save_name)

        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)