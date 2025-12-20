import torch
import pandas as pd

from training.config import TrainConfig

from models.dcgan import Generator, Discriminator
from models.text_encoder import CNNRNNEncoder


def checkpoint_dcgan(
        cfg: TrainConfig,
        generator: Generator,
        discriminator: Discriminator,
        report: list,
) -> None:
    
    model_name = f"generator_ngf{generator.ngf}_epochs{cfg.epochs}_lr{cfg.lr}_{cfg.dataset}.pth"
    torch.save(generator.state_dict(), model_name)
    
    model_name = f"discriminator_ndf{discriminator.ndf}_epochs{cfg.epochs}_lr_{cfg.lr}_{cfg.dataset}.pth"
    torch.save(discriminator.state_dict(), model_name)

    csv_name = f"gan_report_epochs{cfg.epochs}_lr{cfg.lr}_{cfg.dataset}.csv" 
    pd.DataFrame(report, columns=[
        "epochs", "loss-generator", "loss-discriminator", 
    ]).to_csv(csv_name, index=False)


def checkpoint_txt_enc(
        cfg: TrainConfig,
        txt_enc: CNNRNNEncoder,
        report: list,
) -> None:
    
    model_name = f"txt_enc_cnn{txt_enc.cnn_dim}_emb{txt_enc.emb_dim}_epochs{cfg.epochs}_lr{cfg.lr}_{cfg.dataset}.pth"
    torch.save(txt_enc.state_dict(), model_name)
    
    csv_name = f"txt_enc_report_epochs{cfg.epochs}_lr{cfg.lr}_{cfg.dataset}.csv" 
    pd.DataFrame(report, columns=[
        "epochs", "train-loss", "train-acc", 
        "val-loss", "val-acc"
    ]).to_csv(csv_name, index=False)