import os
import sys

import torch
from torch.utils.data import Subset, DataLoader

from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.dcgan import Generator, Discriminator
from models.objectives import GeneratorLoss, DiscriminatorLossCLS

from datasets.cub200 import CUBGanCls

from training.preview import Preview
from training.config import TrainConfig
from training.gan_int_cls_trainer import Trainer

from utils.utils import set_seeds

from cub.config import (
    DATASET,
    ROOT_DIR,
    ROOT_TXT_EMBEDDINGS, 

    DEVICE,
    SEED,
    TRAIN_CLASSES,
    VAL_CLASSES,

    TXT_EMB_DIM,
    NUM_CAPTIONS,

    NZ,
    NT,
    NGF,
    NDF,
    INT_WEIGHT,
    CLS_WEIGHT
)


if __name__ == '__main__':
    cfg = TrainConfig(
        dataset=DATASET,
        device=DEVICE,
        seed=SEED,

        epochs=600,
        decay_every=100,
        batch_size=64,
        lr=0.0002,
        lr_decay=0.5,
        betas=(0.5, 0.999),

        nz=NZ,
        nt=NT,
        ngf=NGF,
        ndf=NDF,

        emb_in=TXT_EMB_DIM,
        emb_out=NT, 
    
    ) 
    
    set_seeds(cfg.seed)
    
    transform = transforms.Compose([
        transforms.Resize(76),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = CUBGanCls(ROOT_DIR,
        root_cap_emb=ROOT_TXT_EMBEDDINGS, 
        num_captions=NUM_CAPTIONS,
        classes=TRAIN_CLASSES,
        transform=transform,
    )

    class_to_imgs = dataset.class_id_to_img_ids
    train_indices = [img_id for class_id in TRAIN_CLASSES for img_id in class_to_imgs[class_id]]
    train_set = Subset(dataset, train_indices)

    dataloader = DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory, 
    )

    generator = Generator(NZ, NT, TXT_EMB_DIM, NGF).to(cfg.device)
    discriminator = Discriminator(NT, TXT_EMB_DIM, NDF).to(cfg.device)
    
    criterion_g = GeneratorLoss()
    criterion_d = DiscriminatorLossCLS(CLS_WEIGHT)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
    
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, cfg.decay_every, cfg.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, cfg.decay_every, cfg.lr_decay)

    preview = Preview(cfg, generator, dataset, train_indices, mode="int-cls")
    trainer = Trainer(
        cfg=cfg,
        preview=preview,
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        criterion_g=criterion_g,
        criterion_d=criterion_d,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        interpolation_weight=INT_WEIGHT
    )
    trainer.train()