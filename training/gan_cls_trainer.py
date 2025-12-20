import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from training.checkpoints import checkpoint_dcgan
from training.config import Preview, TrainConfig


class Trainer:
    def __init__(
            self,
            cfg: TrainConfig,
            preview: Preview,
            generator: nn.Module,
            discriminator: nn.Module,
            dataloader: DataLoader,
            criterion_g: nn.Module,
            criterion_d: nn.Module,
            optimizer_g: Optimizer,
            optimizer_d: Optimizer,
            scheduler_g: LRScheduler,
            scheduler_d: LRScheduler,
    ) -> None:
        self.cfg = cfg
        self.preview = preview
        
        self.generator = generator.to(cfg.device)
        self.discriminator = discriminator.to(cfg.device)

        self.dataloader = dataloader
        self.num_batches = len(dataloader)

        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.scheduler_g = scheduler_g 
        self.scheduler_d = scheduler_d

    def train(self) -> None:
        """GAN-CLS Algorithm""" 
        cfg = self.cfg
        epochs = cfg.epochs
        dataloader = self.dataloader
        n_samples = len(dataloader)
        device = self.cfg.device
        nz = cfg.nz

        generator = self.generator
        discriminator = self.discriminator

        optimizer_g = self.optimizer_g
        optimizer_d = self.optimizer_d

        criterion_g = self.criterion_g
        criterion_d = self.criterion_d

        scheduler_g = self.scheduler_g
        scheduler_d = self.scheduler_d

        report = []

        if cfg.verbose:
            print(
                f'======== Training settings ========\n'
                f'Epochs:         {cfg.epochs}\n'
                f'Num batches:    {self.num_batches}\n'
                f'Device:         {cfg.device}\n'
                f'Learning rate:  {cfg.lr}\n'
                f'Betas:          {cfg.betas}\n'
                f'Decay factor:   {cfg.lr_decay}\n'
                f'Decay every:    {cfg.decay_every}\n'
                f'===================================\n'
            )
        
        for epoch in range(0, epochs):
            generator.train()
            discriminator.train()

            n_samples = 0
            total_loss_g, total_loss_d = 0.0, 0.0
            epoch_time = time.monotonic() 
            for img, wrong_img, text, _, _ in dataloader:
                batch = img.size(0)                                     # N := batch

                img = img.to(device)                                    # [N, 3, 64, 64]
                wrong_img = wrong_img.to(device)                        # [N, 3, 64, 64]
                text = text.to(cfg.device)                              # [N, vocab_size, ctx_len]

                # =============================================================
                # ================== Update discriminator =====================
                # =============================================================
                noise = torch.randn(size=(batch, nz, 1, 1)).to(device)  # [N, 100]
                with torch.no_grad(): 
                    fake_img = self.generator(noise, text)              # [N, 3, 64, 64]

                real_score = discriminator(img, text)                   # real image, right text
                wrong_score = discriminator(wrong_img, text)            # real image, wrong text
                fake_score = discriminator(fake_img.detach(), text)     # fake image, right text

                loss_d = criterion_d(real_score, fake_score)
                loss_d = criterion_d(real_score, wrong_score, fake_score)

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
                
                # =============================================================
                # =========== Update generator and text encoder  ==============
                # =============================================================
                noise = torch.randn(size=(batch, nz, 1, 1)).to(device)  # [N, 100]
                fake_img = generator(noise, text)                       # [N, 3, 64, 64]
 
                fake_score = discriminator(fake_img, text)              # fake image, right text

                loss_g = criterion_g(fake_score)

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                # =============================================================
                # ================= Tracking stats and stuff ==================
                # =============================================================
                n_samples += batch 
                total_loss_d += loss_d.item() * batch
                total_loss_g += loss_g.item() * batch
            
            epoch_time = time.monotonic() - epoch_time

            scheduler_g.step()
            scheduler_d.step()

            loss_g = total_loss_g / n_samples
            loss_d = total_loss_d / n_samples
            report.append((epoch, loss_g, loss_d))
            
            if self.cfg.verbose:
                print(
                    f'epoch: {epoch}\t' 
                    f'generator loss: {loss_g:.4f}\t'
                    f'discriminator loss: {loss_d:.4f}\t'
                    f'time: {epoch_time:.4f}s\t'
                ) 

            if self.cfg.preview:
                self.preview.preview(epoch)

            if epoch % cfg.save_every == 0:
                checkpoint_dcgan(cfg, generator, discriminator, report, cfg.dataset)
        
        checkpoint_dcgan(cfg, generator, discriminator, report, cfg.dataset)