import os
import time
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from data.tokenizer import Tokenizer
from data.dataset import COCO2014MatchingCaptions

from models.cnn_rnn import CNNRNNEncoder
from models.objectives import JointEmbeddingLoss

from utils.utils import set_seeds
from utils.helpers import checkpoint_cnnrnn


@dataclass
class TrainConfig:
    epochs: int=200
    decay_every: int=50
    validate_every: int=10
    batch_size: int=64
    cnn_dim: int=512
    emb_dim: int=1024
    context_length: int=201
    lr: float=0.0002
    lr_decay: float=0.5
    betas: tuple[float, float]=(0.5, 0.999)
    device: torch.device = torch.device('cuda')
    seed: int=42
    verbose: bool=True
    path_report_csv: str='cnn_rnn_report.csv'
    checkpoint_path: str='./cnn_rnn_checkpoints/'
    path_coco_ann_json: str='../../datasets/annotations_trainval2014/annotations/captions_train2014.json'
    path_coco_val_ann_json: str='../../datasets/annotations_trainval2014/annotations/captions_val2014.json'


class Trainer:
    def __init__(
            self,
            cnn_rnn: nn.Module, 
            loader: DataLoader,
            valloader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            cfg: TrainConfig,
        ) -> None:
        self.cnn_rnn = cnn_rnn
        self.loader = loader
        self.valloader = valloader
        self.crit = criterion
        self.opt = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        os.makedirs(cfg.checkpoint_path, exist_ok=True)

    def train(self) -> None:
        if self.cfg.verbose:
            print(f'\n======= Training settings =======\n'
                  f'Epochs:         {cfg.epochs}\n'
                  f'Num batches:    {len(self.loader)}\n'
                  f'Device:         {cfg.device}\n'
                  f'Learning rate:  {cfg.lr}\n'
                  f'Betas:          {cfg.betas}\n'
                  f'Decay factor:   {cfg.lr_decay}\n'
                  f'Decay every:    {cfg.decay_every}\n'
                  f'===================================\n'
            ) 
        N_batches = len(self.loader) 
        device = self.cfg.device

        losses, accs = [], []
        for epoch in range(1, self.cfg.epochs+1):
            self.cnn_rnn.train()
            total_loss, total_acc = 0.0, 0.0 
            epoch_time = time.monotonic()               # in seconds
            for text, match_text in self.loader:
                text = text.to(device)                  # [N, vocab_size, 201]
                match_text = match_text.to(device)      # [N, vocab_size, 201]

                emb = self.cnn_rnn(text)                # [N, 1024]
                match_emb = self.cnn_rnn(match_text)    # [N, 1024]

                # Minimize sum_ij [max{0, s_ij - sii + margin}] for all i != j
                loss, acc = self.crit(emb, match_emb)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                total_acc += acc.item()

            losses.append(total_loss / N_batches)
            accs.append(total_acc / N_batches)
            epoch_time = time.monotonic() - epoch_time
            checkpoint_cnnrnn(self.cfg, self.cnn_rnn, losses, accs)

            if epoch % cfg.decay_every == 0:
                self.scheduler.step()

            val_report = epoch % cfg.validate_every == 0
            if val_report:
                val_loss, val_acc = self._validate()

            if self.cfg.verbose:
                val_report = f'acc (val): {val_acc:.4f}\tloss (val): {val_loss:.4f}\t' if val_report else ''
                print(
                    f'epoch: {epoch}\t' 
                    f'loss: {(total_loss/N_batches):.4f}\t'
                    f'acc: {(total_acc/N_batches):.4f}\t'
                    f'{val_report}'
                    f'time: {(epoch_time):.4f}s\t'
                ) 

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.cnn_rnn.eval()
        device = self.cfg.device
        N_batches = len(self.valloader) 
        total_loss, total_acc = 0.0, 0.0
        for text, match_text in self.valloader:
            text = text.to(device)
            match_text = match_text.to(device)

            emb1 = self.cnn_rnn(text)
            emb2 = self.cnn_rnn(match_text)

            loss, acc_batch = self.crit(emb1, emb2)
            total_loss += loss.item() 
            total_acc += acc_batch.item()
        return total_loss / N_batches, total_acc / N_batches


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog='Training a deep convolutional-recurrent text encoder.'
    )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    # === Training configs ===
    args = parse_args() 
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=torch.device(args.device), seed=args.seed)
    set_seeds(cfg.seed)

    # === Load COCO training set ===
    tokenizer = Tokenizer() 
    dataset = COCO2014MatchingCaptions(cfg.path_coco_ann_json, tokenizer, cfg.context_length)
    dataloader = DataLoader(dataset, cfg.batch_size, True, num_workers=4, persistent_workers=True, pin_memory=True)

    # === Load COCO validation set ===
    valset = COCO2014MatchingCaptions(cfg.path_coco_val_ann_json, tokenizer, cfg.context_length)
    valloader = DataLoader(dataset, cfg.batch_size, False)

    # === CNN-RNN-Encoder ===
    cnn_rnn = CNNRNNEncoder(vocab_size=len(tokenizer), cnn_dim=cfg.cnn_dim, emb_dim=cfg.emb_dim) 
    cnn_rnn.to(cfg.device)
    
    # === Objective, optimizer and scheduler ===
    crit = JointEmbeddingLoss()
    opt = torch.optim.Adam(cnn_rnn.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, cfg.lr_decay)

    # === Start training ===
    trainer = Trainer(
        cnn_rnn=cnn_rnn,
        loader=dataloader,
        valloader=valloader,
        criterion=crit,
        optimizer=opt,
        scheduler=scheduler,
        cfg=cfg,
    ) 
    trainer.train()