import os
import textwrap
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import COCO2014ImgEmb, Tokenizer

from utils.utils import add_padding


class Visualizer:
    def __init__(
            self,
            text_encoder: nn.Module,
            dataset: COCO2014ImgEmb,
            tokenier: Tokenizer,
            ctx_len: int, 
            device: torch.device,
            figsize: tuple[int, int]=(18, 10),
            dim_noise: int=100,
            rows: int=4,
            cols: int=6,
            save_dir: str='./fake_coco/',
        ) -> None:
        self.text_encoder = text_encoder 
        self.dataset = dataset
        self.tokenizer = tokenier 
        self.ctx_len = ctx_len 
        self.pad_val = tokenier.pad_token_id
        self.device = device 
        self.dim_noise = dim_noise
        
        self.figsize = figsize
        self.rows = rows
        self.cols = cols
        self.save_dir = save_dir

        self._prepare_preview()

    @torch.no_grad()
    def _prepare_preview(self) -> None:
        self.text_encoder.eval()
        os.makedirs(self.save_dir, exist_ok=True) 
        n_images = self.rows * self.cols

        idx = torch.randint(0, len(self.dataset), size=(n_images,)) 
        captions_lst, tokens_lst = [], []
        for id in idx:
            img_id = self.dataset.coco.imgs[self.dataset.img_ids[id]]['id']
            captions = self.dataset.coco.imgToAnns[img_id]
            caption = random.choice(captions)
            caption = caption['caption'].lower()
            tokens = self.tokenizer.encode(caption)
            tokens = torch.Tensor(tokens).long()
            tokens = add_padding(tokens, self.pad_val, self.ctx_len) 
            tokens = F.one_hot(tokens, num_classes=len(self.tokenizer)).float()
            tokens = tokens.transpose(1, 0)
            
            tokens_lst.append(tokens)
            captions_lst.append(caption)

        tokens = torch.stack(tokens_lst)
        self.noise = torch.randn(size=(n_images, self.dim_noise, 1, 1), device=self.device)
        self.tokens = tokens.to(self.device) 
        self.captions = captions_lst 
        self.embs = self.text_encoder(self.tokens)

    @torch.no_grad()
    def _preview(self, G: nn.Module, epoch: int) -> None:
        G.eval()
        x_hat = G(self.noise, self.embs)

        fig, ax = plt.subplots(self.rows, self.cols, figsize=self.figsize)
        ax = ax.flatten()
        
        n_images = self.rows * self.cols
        for i in range(n_images):
            img = x_hat[i].cpu().numpy().transpose(1, 2, 0)
            img = ((img + 1) / 2).clip(0, 1)
            text = self.captions[i] 
            text = "\n".join(textwrap.wrap(text, width=25))
            ax[i].set_title(text, fontsize=6) 
            ax[i].imshow(img)
            ax[i].axis('off')
        fig.tight_layout(pad=0.5)
        fig.savefig(os.path.join(self.save_dir, f'coco_{epoch}'))
        plt.close(fig)