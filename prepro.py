import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pycocotools.coco import COCO

from data.dataset import Tokenizer
from models.cnn_rnn import CNNRNNEncoder
from utils.utils import add_padding


@dataclass
class Config:
    emb_dim: int=1024
    cnn_dim: int=512
    context_length: int=201
    device: torch.device = torch.device('cuda')
    path_coco_imgs: str='../../datasets/train2014/'
    path_coco_ann_json: str='../../datasets/annotations_trainval2014/annotations/captions_train2014.json'
    path_save_emb: str='./embeddings/'
    path_cnn_rnn_weights: str='./cnn_rnn_checkpoints/weights_cnn_rnn.pth'


@torch.no_grad()
def calculate_embeddings(
        coco_ann_json: str,
        path_save_emb: str,
        model: nn.Module, 
        tokenizer: Tokenizer, 
        context_length: int,
        device: torch.device,
    ) -> None:
    """
    Creates a folder called embeddings.
    Uses the trained text encoder to embed COCO caption. 
    All embeddings are saved in the folder ./embeddings.
    """ 
    assert coco_ann_json.endswith('.json'), f'File: {coco_ann_json} is not a .json file.'
    assert os.path.exists(coco_ann_json), f'File: {coco_ann_json} does not exist.' 
    os.makedirs(path_save_emb, exist_ok=True)

    coco = COCO(coco_ann_json)
    indices = coco.getImgIds()
    pad_value = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    for idx in range(len(indices)):
        img_id = coco.imgs[indices[idx]]['id']
        captions = coco.imgToAnns[img_id]

        embeddings = []
        for cap in captions:
            cap = cap['caption'].lower()                            # string of length n
            cap = tokenizer.encode(cap)                             # list of n integers
            cap = torch.Tensor(cap).long()                          # tensor of shape [n,]
            cap = add_padding(cap, pad_value, context_length)       # tensor of shape [ctx_len,]
            cap = F.one_hot(cap, num_classes=vocab_size).float()    # tensor of shape [ctx_len, vocab_size]
            cap = cap.transpose(1, 0)                               # tensor of shape [vocab_size, ctx_len]

            cap = cap.unsqueeze(0).to(device)                       # tensor of shape [1, vocab_size, ctx_len]
            emb = model(cap)                                        # tensor of shape [1, 1024]
            embeddings.append(emb.cpu())
        embeddings = torch.stack(embeddings)                        # tensor of shape [len(E), 1, 1024]
        embeddings = embeddings.squeeze(1)                          # tensor of shape [len(E), 1024]
        save_path = os.path.join(path_save_emb, f'{img_id}.pth')
        torch.save(embeddings, save_path)


if __name__ == '__main__':
    cfg = Config()
    tokenizer = Tokenizer()
    model = CNNRNNEncoder(len(tokenizer), cfg.cnn_dim, cfg.emb_dim)
    model.load_state_dict(torch.load(cfg.path_cnn_rnn_weights, weights_only=True))
    model.to(cfg.device) 
    calculate_embeddings(
        cfg.path_coco_ann_json,
        cfg.path_save_emb,
        model,
        tokenizer,
        cfg.context_length,
        cfg.device
    )