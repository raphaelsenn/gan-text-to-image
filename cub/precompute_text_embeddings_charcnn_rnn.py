import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.cub200 import CUBCaption

from models.text_encoder import CNNRNNEncoder

from cub.config import (
    ROOT_DIR,
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

from prepro.precompute_text_embeddings_charcnn_rnn import precompute_text_embeddings_charcnn_rnn


if __name__ == '__main__':
    dataset = CUBCaption(ROOT_DIR, CONTEXT_LENGTH)

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