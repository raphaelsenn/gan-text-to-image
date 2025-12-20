import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.flowers102 import FlowersCaption
from datasets.cub200 import CUBCaption

from models.text_encoder import CNNRNNEncoder
from utils.utils import pad_onehot_transpose



@torch.no_grad()
def precompute_text_embeddings_charcnn_rnn(
        model: CNNRNNEncoder,
        dataset: FlowersCaption | CUBCaption,
        root_txt_embeddings: str,
        pad_val: int,
        ctx_len: int,
        device: torch.device,
        verbose: bool=True
) -> None:
    """
    Simple script to extract and save 1024-d text embeddings using the trained Char-CNN-RNN.
    Designed to work on both: ./datasets/cub200 and ./datasets/flowers102.
    """ 
    os.makedirs(root_txt_embeddings, exist_ok=True)

    model.eval()
    tokenizer = dataset.tokenizer
    n_samples = len(dataset)
    vocab_size = len(tokenizer) 
    w = len(str(n_samples))

    # for index in range(n_samples):
    for i, (img_id, img_name) in enumerate(dataset.img_id_to_img_name.items()):
        captions = dataset.get_all_captions(img_id)

        embeddings = []
        for cap in captions:
            cap = cap.lower().strip()               # string of length n
            cap = tokenizer(cap)                    # list of n integers
            tok = pad_onehot_transpose(
                cap, pad_val, ctx_len, vocab_size
            )                                       # tensor of shape [vocab_size, ctx_len]  
            tok = tok.unsqueeze(0).to(device)       # tensor of shape [1, vocab_size, ctx_len]
            emb = model(tok)                        # tensor of shape [1, 1024]
            embeddings.append(emb.cpu())

        embeddings = torch.stack(embeddings)        # tensor of shape [len(E), 1, 1024]
        embeddings = embeddings.squeeze(1)          # tensor of shape [len(E), 1024]
        
        emb_name = img_name.replace("jpg", "pth") 
        save_path = os.path.join(root_txt_embeddings, emb_name)
        torch.save(embeddings, save_path)

        if verbose and (i+1) % 256 == 0:
            pct = ((i+1) / n_samples) * 100
            print(f"[{(i+1):>{w}} / {n_samples}]\t{pct:.2f}% done")