"""
Expected folder structure for 102flowers

|-- 102flowers
|   |-- images          // image folder
|   |-- captions        // caption folder
|   |   |-- class_0001  // captions of class_0001
|   |   |-- class_0002  // captions of class_0001
|   |   |-- class_0003  // captions of class_0001
.   .   .
.   .   .
.   .   .
"""
import torch


DATASET: str="102flowers"

ROOT_IMG_DIR: str="../datasets/102flowers/images/"
ROOT_CAP_DIR: str="../datasets/102flowers/captions"
ROOT_IMG_FEATURES: str="./102flowers-image-features"
ROOT_TXT_EMBEDDINGS: str="./102flowers-text-embeddings"

WEIGHTS_TEXT_ENCODER: str="txt_enc_cnn512_emb1024_epochs100_lr0.0007_102flowers.pth"
WEIGHTS_GENERATOR: str="generator_ngf128_epochs600_lr0.0002_102flowers.pth"
WEIGHTS_DISCRIMINATOR: str="discriminator_ndf128_epochs600_lr0.0002_102flowers.pth"

# Random seed for reproduction
SEED: int=42
torch.manual_seed(SEED)

# Computing device (GPU or CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Split dataset into training and validation set
N_CLASSES: int=102
N_TRAIN_CLASSES: int=82

CLASSES = torch.randperm(N_CLASSES)
TRAIN_CLASSES = CLASSES[:N_TRAIN_CLASSES]
VAL_CLASSES= CLASSES[N_TRAIN_CLASSES:]

torch.save(TRAIN_CLASSES, f"102flowers-train-split-classes{len(TRAIN_CLASSES)}.pth")
torch.save(VAL_CLASSES, f"102flowers-val-split_classes{len(VAL_CLASSES)}.pth")

TRAIN_CLASSES = TRAIN_CLASSES.tolist()
VAL_CLASSES = VAL_CLASSES.tolist()

# NLP settings
PAD_VALUE = 0
VOCAB_SIZE = 70
CONTEXT_LENGTH = 201
VOCAB_SIZE = 70

# Text encoder settings
CNN_DIM=512
TXT_EMB_DIM=1024
AVERAGE=True

# GoogLeNet feature dim
IMG_EMB_DIM=1024

# GAN settings
NUM_CAPTIONS=5
NZ=100
NT=128
NGF=128
NDF=128
CLS_WEIGHT=0.5
INT_WEIGHT=1.0