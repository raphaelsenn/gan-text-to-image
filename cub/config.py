"""
Expected folder structure for CUB-200-2011

|-- cub-200-2011
|   |-- images      // image folder
|   |-- captions    // caption folder
"""
import torch


DATASET: str="cub-200-2011"

ROOT_DIR: str="../datasets/cub-200-2011/"
ROOT_IMG_FEATURES: str="./cub-200-2011-image-features"
ROOT_TXT_EMBEDDINGS: str="./cub-200-2011-text-embeddings"
CUB_200_2011_IMG_TEXT_FILE: str="../datasets/cub-200-2011/images.txt"

WEIGHTS_TEXT_ENCODER: str="txt_enc_cnn512_emb1024_epochs25_lr0.0007_cub-200-2011.pth"
WEIGHTS_GENERATOR: str="generator_ngf128_epochs600_lr0.0002_cub-200-2011.pth"
WEIGHTS_DISCRIMINATOR: str="discriminator_ndf64_epochs600_lr0.0002_cub-200-2011.pth"

# Random seed for repdoducability
SEED: int=42
torch.manual_seed(SEED)

# Computing device (GPU or CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Split dataset into training and validation set
N_CLASSES: int=200
N_TRAIN_CLASSES: int=160

CLASSES = torch.randperm(N_CLASSES)
TRAIN_CLASSES = CLASSES[:N_TRAIN_CLASSES]
VAL_CLASSES= CLASSES[N_TRAIN_CLASSES:]

torch.save(TRAIN_CLASSES, f"cub_200-2011-train-split-classes{len(TRAIN_CLASSES)}.pth")
torch.save(VAL_CLASSES, f"cub-200-2011-val-split_classes{len(VAL_CLASSES)}.pth")

TRAIN_CLASSES = TRAIN_CLASSES.tolist()
VAL_CLASSES = VAL_CLASSES.tolist()

# NLP settings
PAD_VALUE = 0
VOCAB_SIZE = 70
CONTEXT_LENGTH = 201

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