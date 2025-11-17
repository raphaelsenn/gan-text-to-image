import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from torchvision.transforms import transforms

from models.dcgan import Generator, Discriminator
from models.cnn_rnn import CNNRNNEncoder
from models.objectives import GeneratorLoss, DiscriminatorLoss

from data.dataset import COCO2014ImgEmb
from data.tokenizer import Tokenizer

from utils.visualize import Visualizer
from utils.utils import set_seeds
from utils.helpers import checkpoint_dcgan


@dataclass
class TrainConfig:
    epochs: int=200
    decay_every: int=40
    batch_size: int=64
    dim_noise: int=100
    emb_in: int=1024
    emb_out: int=128
    cnn_dim: int=512
    context_length: int=201
    ngf: int=196
    ndf: int=196
    lr: float=0.0002
    lr_decay: float=0.5
    betas: tuple[float, float]=(0.5, 0.999)
    device: torch.device = torch.device('cuda')
    seed: int=42
    num_workers: int=4
    verbose: bool=True
    preview: bool=True
    preview_figsize: tuple[int, int]=(12, 8)
    preview_rows: int=4
    preview_cols: int=6
    image_dim: tuple[int, int]= (3, 64, 64)
    path_report_csv: str='report.csv'
    checkpoint_path: str='./checkpoints/'
    path_coco_ann_json: str='../../datasets/annotations_trainval2014/annotations/captions_train2014.json'
    path_coco_imgs: str='../../datasets/train2014/'
    path_emb_dir: str='./embeddings/'
    path_cnn_rnn_pth: str='./cnn_rnn_checkpoints/weights_cnn_rnn.pth'


class Trainer:
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            loader: DataLoader,
            criterion_g: nn.Module,
            criterion_d: nn.Module,
            optimizer_g: Optimizer,
            optimizer_d: Optimizer,
            scheduler_g: LRScheduler,
            scheduler_d: LRScheduler,
            cfg: TrainConfig,
            vis: Visualizer
        ) -> None:
        self.G = generator.to(cfg.device)
        self.D = discriminator.to(cfg.device)
        
        self.loader = loader
        
        self.crit_g = criterion_g
        self.crit_d = criterion_d
        self.opt_g = optimizer_g
        self.opt_d = optimizer_d
        self.scheduler_g = scheduler_g 
        self.scheduler_d = scheduler_d
        
        self.cfg = cfg
        self.vis = vis

        if not os.path.exists(cfg.checkpoint_path):
            os.mkdir(cfg.checkpoint_path)    

    def train(self) -> None:
        if self.cfg.verbose:
            print(f'\n======= Training settings ======\n'
                  f'Epochs:         {cfg.epochs}\n'
                  f'Num batches:    {len(self.loader)}\n'
                  f'Device:         {cfg.device}\n'
                  f'Learning rate:  {cfg.lr}\n'
                  f'Betas:          {cfg.betas}\n'
                  f'Decay factor:   {cfg.lr_decay}\n'
                  f'Decay every:    {cfg.decay_every}\n'
                  f'==================================\n'
            )  
        N_samples = len(self.loader.dataset)
        losses_d, losses_g = [], []
        device = self.cfg.device

        for epoch in range(1, self.cfg.epochs+1):
            self.G.train()
            self.D.train()

            total_loss_g, total_loss_d = 0.0, 0.0
            epoch_time = time.monotonic() 
            for imgs, wrong_imgs, embs in self.loader:
                imgs = imgs.to(device)              # [N, 3, 64, 64]
                wrong_imgs = wrong_imgs.to(device)  # [N, 3, 64, 64]
                embs = embs.to(device)              # [N, 1024]

                # =============================================================
                # ================== Update discriminator =====================
                # =============================================================
                noise = torch.randn(size=(imgs.shape[0], self.cfg.dim_noise, 1, 1)).to(device)
                with torch.no_grad(): 
                    fake_imgs = self.G(noise, embs)

                sr = self.D(imgs, embs)                   # real image, right text
                sw = self.D(wrong_imgs, embs)             # real image, wrong text
                sf = self.D(fake_imgs.detach(), embs)     # fake image, right text

                # Minimize -[log(D(real img, right text)) + 0.5*log(1 - D(real img, wrong text)) + 0.5*log(1 - D(fake img, right text))]
                loss_d = self.crit_d(sr, sw, sf)

                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()
                
                # =============================================================
                # ==================== Update generator =======================
                # =============================================================
                noise = torch.randn(size=(imgs.shape[0], self.cfg.dim_noise, 1, 1)).to(device)
                fake_imgs = self.G(noise, embs)
 
                sf = self.D(fake_imgs, embs)             # fake image, right text

                # Minimize -[log(D(G(noise), right text))]
                loss_g = self.crit_g(sf)

                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()

                # =============================================================
                # ================= Tracking stats and stuff ==================
                # =============================================================
                total_loss_d += loss_d.item() * imgs.shape[0]
                total_loss_g += loss_g.item() * imgs.shape[0]
            epoch_time = time.monotonic() - epoch_time
            self.scheduler_g.step()
            self.scheduler_d.step()

            losses_d.append(total_loss_d / N_samples)
            losses_g.append(total_loss_g / N_samples)
            checkpoint_dcgan(self.cfg, self.G, self.D, losses_g, losses_d)
            if self.cfg.verbose: 
                print(
                    f'epoch: {epoch}\t' 
                    f'generator loss: {(total_loss_g/N_samples):.4f}\t'
                    f'discriminator loss: {(total_loss_d/N_samples):.4f}\t'
                    f'time: {epoch_time:.4f}s\t'
                ) 
            if self.cfg.preview: 
                self.vis._preview(self.G, epoch)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog='Training a text-conditioned DCGAN.'
    )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    # === Get arguments, set seeds, and init training setup ===
    args = parse_args() 
    set_seeds(args.seed)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device)
    )
    
    # === Transformations and tokenizer ===
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=64, 
            scale=(0.5, 1.0), 
            ratio=(0.9,1.1), 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tokenizer = Tokenizer() 
    
    # === Load COCO training set === 
    dataset = COCO2014ImgEmb(
        root=cfg.path_coco_imgs, 
        coco_ann_json=cfg.path_coco_ann_json, 
        emb_dir=cfg.path_emb_dir, 
        tokenizer=tokenizer, 
        transform=transform,
        context_length=cfg.context_length
    )
    dataloader = DataLoader(
        dataset, 
        cfg.batch_size, 
        True, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    # === Preparing generator and discriminator ===
    G = Generator(cfg.dim_noise, cfg.emb_in, cfg.emb_out, cfg.ngf).to(cfg.device)
    D = Discriminator(cfg.emb_in, cfg.emb_out, cfg.ndf).to(cfg.device)
    
    cnn_rnn = CNNRNNEncoder(len(tokenizer), cfg.cnn_dim, cfg.emb_in).to(cfg.device)
    cnn_rnn.load_state_dict(torch.load(cfg.path_cnn_rnn_pth, weights_only=True))
    cnn_rnn.eval()
    
    # === Objective, optimizer and lr scheduler  ===
    crit_g = GeneratorLoss()
    crit_d = DiscriminatorLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, cfg.decay_every, cfg.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, cfg.decay_every, cfg.lr_decay)
    
    vis = Visualizer(cnn_rnn, dataset, tokenizer, cfg.context_length, cfg.device)
    trainer = Trainer(
        generator=G,
        discriminator=D,
        loader=dataloader,
        criterion_g=crit_g,
        criterion_d=crit_d,
        optimizer_g=opt_g,
        optimizer_d=opt_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        cfg=cfg,
        vis=vis
    ) 
    trainer.train()