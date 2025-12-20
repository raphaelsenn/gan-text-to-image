import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.cub200 import CUBTextEnc, CUBSampler

from models.image_encoder import ImageEncoder
from models.text_encoder import CNNRNNEncoder

from models.objectives import JointEmbeddingLoss

from utils.utils import set_seeds

from training.config import TrainConfig
from training.img_txt_trainer import Trainer
from cub.config import (
    DATASET,
    ROOT_DIR,
    ROOT_IMG_FEATURES,
    
    CNN_DIM,
    TXT_EMB_DIM,
    IMG_EMB_DIM,
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    AVERAGE, 
    DEVICE,
    
    TRAIN_CLASSES,
    VAL_CLASSES
)


if __name__ == '__main__':
    cfg = TrainConfig(
        dataset=DATASET,
        device=DEVICE,

        epochs=25,
        batch_size=40,
        lr=0.0007,
        lr_decay=0.98,
        grad_norm_clip=5.0,

        decay_every=1,
        save_every=2,
        eval_every=1,
        
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,

        cnn_dim=CNN_DIM,
        emb_dim=TXT_EMB_DIM,
        img_feat_dim=IMG_EMB_DIM,
        context_length=CONTEXT_LENGTH,
        average=AVERAGE,
    ) 

    # Set seeds for reproducability 
    set_seeds(cfg.seed)

    # Load CUB dataset (googlenet features version)
    dataset = CUBTextEnc(ROOT_DIR, ROOT_IMG_FEATURES, CONTEXT_LENGTH)

    train_sampler = CUBSampler(dataset, TRAIN_CLASSES, cfg.batch_size)
    val_sampler = CUBSampler(dataset, VAL_CLASSES, cfg.batch_size)

    train_loader = DataLoader(
        dataset, 
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers, 
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        dataset, 
        batch_sampler=val_sampler, 
        num_workers=cfg.num_workers, 
        persistent_workers=cfg.persistent_workers, 
        pin_memory=cfg.pin_memory
    )

    # CNN-RNN text encoder
    txt_enc = CNNRNNEncoder(VOCAB_SIZE, CNN_DIM, TXT_EMB_DIM, AVERAGE)
    txt_enc.to(cfg.device)

    # Image encoder
    img_enc = ImageEncoder(cfg.img_feat_dim, cfg.emb_dim)
    img_enc.to(cfg.device)

    # Objective, optimizer and scheduler
    criterion = JointEmbeddingLoss()
    optimizer = torch.optim.RMSprop(txt_enc.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.decay_every, cfg.lr_decay)

    # Start training
    trainer = Trainer(
        txt_enc=txt_enc,
        img_enc=img_enc,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )
    trainer.train()