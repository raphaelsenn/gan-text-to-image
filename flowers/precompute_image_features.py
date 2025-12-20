import os
import sys

import torch
from torchvision.transforms import transforms as TF

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.flowers102 import FlowersCaption

from prepro.precompute_image_features_googlenet import precompute_image_features_googlenet

from flowers.config import (
    ROOT_IMG_DIR,
    ROOT_CAP_DIR,
    ROOT_IMG_FEATURES,
    CONTEXT_LENGTH,
    DEVICE
)


if __name__ == "__main__":
    imgagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    transform = TF.Compose([
        TF.Resize(256),
        TF.TenCrop(224, vertical_flip=False),
        TF.Lambda(
            lambda crops: torch.stack([
                TF.Normalize(imgagenet_mean, imagenet_std)(
                    TF.ToTensor()(c)
                ) for c in crops
            ])
        ),
    ])

    dataset = FlowersCaption(ROOT_IMG_DIR, ROOT_CAP_DIR, CONTEXT_LENGTH, transform)
    precompute_image_features_googlenet(dataset, ROOT_IMG_FEATURES, DEVICE)