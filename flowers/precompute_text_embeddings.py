import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.flowers102 import FlowersCaption

from models.text_encoder import CNNRNNEncoder

from prepro.precompute_text_embeddings_charcnn_rnn import precompute_text_embeddings_charcnn_rnn

from flowers.config import (
    ROOT_CAP_DIR,
    ROOT_IMG_DIR,
    ROOT_TXT_EMBEDDINGS,
    WEIGHTS_TEXT_ENCODER, 
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    PAD_VALUE,
    CNN_DIM,
    TXT_EMB_DIM,
    AVERAGE,
    DEVICE,
)


if __name__ == '__main__':
    dataset = FlowersCaption(ROOT_IMG_DIR, ROOT_CAP_DIR, CONTEXT_LENGTH)

    model = CNNRNNEncoder(VOCAB_SIZE, CNN_DIM, TXT_EMB_DIM, AVERAGE)
    model.load_state_dict(torch.load(WEIGHTS_TEXT_ENCODER, weights_only=True))
    model.to(DEVICE) 
    model.eval()

    precompute_text_embeddings_charcnn_rnn(
        model, 
        dataset, 
        ROOT_TXT_EMBEDDINGS, 
        PAD_VALUE, 
        CONTEXT_LENGTH, 
        DEVICE
    )