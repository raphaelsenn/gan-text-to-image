import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from training.config import TrainConfig
from training.checkpoints import checkpoint_txt_enc


class Trainer:
    def __init__(
            self,
            cfg: TrainConfig,
            txt_enc: nn.Module,
            img_enc: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler,
    ) -> None:
        self.cfg = cfg

        self.txt_enc = txt_enc
        self.img_enc = img_enc
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train(self) -> None:
        txt_enc = self.txt_enc 
        img_enc = self.img_enc 

        optimizer = self.optimizer
        scheduler = self.scheduler

        criterion = self.criterion
        dataloader = self.train_loader 

        cfg = self.cfg
        epochs = cfg.epochs
        grad_norm_clip = cfg.grad_norm_clip 
        n_batches = len(dataloader)
        device = cfg.device

        if self.cfg.verbose:
            print(f'\n======= Training settings =======\n'
                  f'Epochs:         {epochs}\n'
                  f'Num batches:    {n_batches}\n'
                  f'Device:         {device}\n'
                  f'Learning rate:  {cfg.lr}\n'
                  f'Decay factor:   {cfg.lr_decay}\n'
                  f'Decay every:    {cfg.decay_every}\n'
                  f'===================================\n'
            ) 

        # [(epoch, train-loss, train-acc, val-loss, val-acc)]
        report: list[tuple[int, float, float, float, float]] = []
        for epoch in range(0, epochs):
            txt_enc.train()
            img_enc.eval()

            epoch_time = time.monotonic()
            for img_feat, text, _ in dataloader:
                img_feat = img_feat.to(device)                  # [N, 1024] (GoogLeeNet image features)
                text = text.to(device)                          # [N, n_captions, vocab_size, ctx_len]

                batch, n_captions, vocab, ctx = text.shape
                text = text.view(batch*n_captions, vocab, ctx)  # [N * n_captions, vocab_size, ctx_len]

                txt_emb = txt_enc(text)                         # [N*n_captions, 1024]
                txt_emb = txt_emb.view(batch, n_captions, -1)   # [N, n_captions, 1024]
                txt_emb = txt_emb.mean(dim=1)                   # [N, 1024] 
                
                img_emb = img_enc(img_feat)                     # [N, 1024]

                loss_ij, _ = criterion(txt_emb, img_emb)
                loss_ji, _ = criterion(img_emb, txt_emb)
                loss = (loss_ij + loss_ji) / 2

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(txt_enc.parameters(), grad_norm_clip)
                optimizer.step()
            epoch_time = time.monotonic() - epoch_time
            
            scheduler.step()

            if epoch % self.cfg.eval_every == 0:
                train_loss, train_acc = self._validate(self.train_loader)
                val_loss, val_acc = self._validate(self.val_loader)
                report.append((epoch, train_loss, train_acc, val_loss, val_acc))
                
                if self.cfg.verbose: 
                    print(
                        f'epoch: {epoch}\t' 
                        f'loss (train): {train_loss:.4f}\t'
                        f'acc (train): {train_acc:.4f}\t'
                        f'loss (val): {val_loss:.4f}\t'
                        f'acc (val): {(val_acc):.4f}\t'
                        f'time: {(epoch_time):.4f}s\t'
                    ) 
            if epoch % cfg.save_every == 0:
                checkpoint_txt_enc(cfg, txt_enc, report)

        # Final save after training 
        checkpoint_txt_enc(cfg, txt_enc, report)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> tuple[float, float]:
        txt_enc = self.txt_enc 
        img_enc = self.img_enc 

        txt_enc.eval()
        img_enc.eval()

        criterion = self.criterion 
        device = self.cfg.device
        
        n_samples = 0
        total_loss, total_acc = 0.0, 0.0
        for img_feat, text, _ in dataloader:
            img_feat = img_feat.to(device)                              # [N, 2048]
            text = text.to(device)                                      # [N, 10, vocab_size, ctx_len]

            batch, n_captions, vocab_size, ctx_len = text.shape

            text = text.view(batch*n_captions, vocab_size, ctx_len)     # [10*N, vocab_size, ctx_len]
            txt_emb = txt_enc(text)                                     # [10*N, 1024]
            txt_emb = txt_emb.view(batch, n_captions, -1)               # [N, 10, 1024]
            txt_emb = txt_emb.mean(dim=1)                               # [N, 1024]

            img_emb = img_enc(img_feat)                                 # [N, 1024]
          
            loss_ij, acc_ij = criterion(img_emb, txt_emb)
            loss_ji, acc_ji = criterion(txt_emb, img_emb)

            loss = (loss_ij + loss_ji) / 2
            acc = (acc_ij + acc_ji) / 2

            total_loss += loss.item() * batch
            total_acc += acc.item() * batch
            n_samples += batch

        # NOTE: n_samples != len(dataloader.dataset) for CUB-200-2011 training, 
        # beause of the training procedure with distinctive classes
        return total_loss / n_samples, total_acc / n_samples