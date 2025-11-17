import os
from dataclasses import dataclass

import torch
import pandas as pd


@dataclass
class TrainConfig:
    checkpoint_path: str
    path_report_csv: str 


def checkpoint_dcgan(
        cfg: TrainConfig,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        losses_generator: list[int],
        losses_discriminator: list[int],
    ) -> None:
    torch.save(
        generator.state_dict(), 
        os.path.join(cfg.checkpoint_path, 'weights_generator.pth'))

    torch.save(
        discriminator.state_dict(), 
        os.path.join(cfg.checkpoint_path, 'weights_discriminator.pth'))
    
    report = pd.DataFrame({
        'generator_loss': losses_generator,
        'discriminator_loss': losses_discriminator,
    })
    report.to_csv(cfg.path_report_csv, index=False)


def checkpoint_cnnrnn(
        cfg: TrainConfig,
        cnn_rnn: torch.nn.Module,
        losses: list[int], 
        accs: list[int], 
    ) -> None:
    torch.save(
        cnn_rnn.state_dict(), 
        os.path.join(cfg.checkpoint_path, 'weights_cnn_rnn.pth')
    )

    report = pd.DataFrame({
        'cnn_rnn_loss': losses,
        'cnn_rnn_acc': accs,
    })
    report.to_csv(cfg.path_report_csv, index=False)