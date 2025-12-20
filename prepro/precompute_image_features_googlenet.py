import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.flowers102 import FlowersCaption
from datasets.cub200 import CUBCaption

from models.googlenet import GoogLeNet


@torch.no_grad()
def precompute_image_features_googlenet(
        dataset: FlowersCaption | CUBCaption,
        root_img_features: str,
        device: torch.device,
        verbose: bool=True
) -> None:
    """
    Simple script to extract and save 1024-d image features using GoogLeNet. 
    Designed to work on both: ./datasets/cub200 and ./datasets/flowers102.
    """ 
    model = GoogLeNet().to(device)
    model.eval()

    os.makedirs(root_img_features, exist_ok=True)

    n_samples = len(dataset)
    w = len(str(n_samples))

    for index in range(0, n_samples):
        imgs, _, _, _ = dataset[index]                  # [10, 3, 224, 224]
        imgs = imgs.to(device)

        features = model(imgs)                          # [10, 1024, 1, 1]
        features = torch.mean(features, dim=0).cpu()    # [1024,]

        img_name = dataset.img_id_to_img_name[index]
        
        feature_name = img_name.replace("jpg", "pth")
        save_path = os.path.join(root_img_features, feature_name)
        torch.save(features, save_path)

        if verbose and (index+1) % 256 == 0:
            pct = ((index+1) / n_samples) * 100
            print(f"[{(index+1):>{w}} / {n_samples}]\t{pct:.2f}% done")