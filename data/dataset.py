import os
import random
from typing import Callable

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO

from data.tokenizer import Tokenizer
from utils.utils import (
    add_padding, 
    randint_except, 
    ensure_rgb, 
    pad_onehot_transpose
)


class COCO2014ImgEmb(Dataset):
    """
    COCO 2014 Images/Embedded-Captions dataset.
    This dataset is used to train the text-conditional DCGAN. 

    Requirements: 
        1. Coco needs to be installed 
        2. Execute the script prepro.py
    """
    def __init__(
            self,
            root: str,
            coco_ann_json: str,
            emb_dir: str,
            tokenizer: Tokenizer, 
            transform: Callable|None=None,
            context_length: int=201
        ) -> None:
        assert os.path.exists(root), f'Path: {root} does not exist.'
        assert os.path.exists(emb_dir), f'Path: {emb_dir} does not exist.' 
        assert os.path.exists(coco_ann_json), f'File: {coco_ann_json} does not exist.' 
        assert coco_ann_json.endswith('.json'), f'File: {coco_ann_json} is not a .json file.'

        self.root = root        # path to coco image folder
        self.emb_dir = emb_dir  # path to image embeddings
        self.coco = COCO(coco_ann_json)
        self.tokenizer = tokenizer
        self.transform = transform
        
        self.img_ids = self.coco.getImgIds()
        self.imgs = [self.coco.imgs[img_id] for img_id in self.img_ids] 
        self.n_samples = len(self.img_ids)
        self.ctx_len = context_length
        self.pad_val = tokenizer.pad_token_id

        self.embeddings = {}
        self._load_embeddings()
        assert len(self.embeddings) == self.n_samples, "Error"

    def _load_embeddings(self) -> None:
        for img_id in self.img_ids:
            emb_path = os.path.join(self.emb_dir, f'{img_id}.pth')
            embeddings = torch.load(emb_path, map_location="cpu")
            embeddings = embeddings.to(torch.float32)
            self.embeddings[img_id] = embeddings

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # --- Loading image ---
        img_name = self.imgs[index]['file_name']
        img_id = self.imgs[index]['id'] 
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)

        # --- Loading corresponding text embedding ---
        embeddings = self.embeddings[img_id]                    # [num_captions, 1024]
        rand_emb_id = torch.randint(0, len(embeddings), (1,)).item()   
        emb = embeddings[rand_emb_id]                           # [1, 1024] -> [1024,]

        # --- Loading random image ---
        rand_index = randint_except(0, self.n_samples, numb_not=index)
        rand_img_name = self.imgs[rand_index]['file_name']
        rand_img_path = os.path.join(self.root, rand_img_name)
        rand_img = Image.open(rand_img_path) 
        
        img = ensure_rgb(img)
        rand_img = ensure_rgb(rand_img)
        if self.transform: 
            img = self.transform(img)
            rand_img = self.transform(rand_img)
        return img, rand_img, emb
    

class COCO2014MatchingCaptions(Dataset):
    """
    COCO 2014 Text-description/Matching-text-description dataset.
    This dataset is used to train the deep convolutional-recurrent text encoder. 
    """
    def __init__(
            self,
            coco_ann_json: str, 
            tokenizer: Tokenizer, 
            context_length: int=201
        ) -> None:
        assert os.path.exists(coco_ann_json), f'File: {coco_ann_json} does not exist.' 
        assert coco_ann_json.endswith('.json'), f'File: {coco_ann_json} is not a .json file.'

        self.coco = COCO(coco_ann_json)
        self.tokenizer = tokenizer

        self.img_ids = self.coco.getImgIds() 
        self.imgs = [self.coco.imgs[i] for i in self.img_ids]
        self.imgs_to_anns = self.coco.imgToAnns

        self.ctx_len = context_length
        self.pad_val = tokenizer.pad_token_id
        self.num_classes = len(tokenizer)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # --- Loading two matching text descriptions ---
        img_id = self.imgs[index]['id'] 
        captions = self.imgs_to_anns[img_id]
        
        captions = np.random.permutation(captions)

        caption = captions[0]['caption'].lower()
        caption = self.tokenizer.encode(caption)
        caption = pad_onehot_transpose(caption, self.pad_val, self.ctx_len, self.num_classes)

        caption_match = captions[1]['caption'].lower()
        caption_match = self.tokenizer.encode(caption_match)
        caption_match = pad_onehot_transpose(caption_match, self.pad_val, self.ctx_len, self.num_classes)
        
        return caption, caption_match
    

class COCO2014ImgCap(Dataset):
    """
    COCO 2014 Images/Text-tokens dataset.
    NOTE: This dataset is not used inside this repo.
    """
    def __init__(
            self,
            root: str, 
            coco_ann_json: str,
            tokenizer: Tokenizer, 
            transform: Callable|None=None,
            context_length: int=201,
        ) -> None:
        assert os.path.exists(root), f'Path: {root} does not exist.'
        assert os.path.exists(coco_ann_json), f'File: {coco_ann_json} does not exist.' 
        assert coco_ann_json.endswith('.json'), f'File: {coco_ann_json} is not a .json file.'

        self.root = root
        self.coco = COCO(coco_ann_json)
        self.indices = self.coco.getImgIds()
        self.tokenizer = tokenizer
        self.transform = transform
        self.ctx_len = context_length
        self.pad_val = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # --- Loading a image with it's corresponding description ---
        img_name = self.coco.imgs[self.indices[index]]['file_name']
        img_id = self.coco.imgs[self.indices[index]]['id'] 
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)

        captions = self.coco.imgToAnns[img_id]
        caption = random.choice(captions)
        caption = caption['caption'].lower()
        caption = self.tokenizer.encode(caption)
        caption = torch.Tensor(caption).long()
        caption = add_padding(caption, self.pad_val, self.ctx_len)

        img = ensure_rgb(img)
        if self.transform: 
            img = self.transform(img)
        # NOTE: Before feeding caption into CNN-RNN, 
        # you need to do: caption = pad_onehot_transpose(caption)
        return img, caption
    
    def get_by_imgid(self, img_id: int|torch.Tensor) -> tuple:
        if isinstance(img_id, torch.Tensor):
            img_id = img_id.item()

        # --- Loading a image with it's corresponding description ---
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)

        captions = self.coco.imgToAnns[img_id]
        caption = random.choice(captions)
        caption = caption['caption'].lower()
        caption = self.tokenizer.encode(caption)
        caption = torch.Tensor(caption).long()
        caption = add_padding(caption, self.pad_val, self.ctx_len)

        img = ensure_rgb(img)
        if self.transform: 
            img = self.transform(img)
        # NOTE: Before feeding captions into the CNN-RNN do:
        # caption = pad_onehot_transpose(caption)
        return img, caption