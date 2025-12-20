import os
import random
from collections import defaultdict
from typing import Callable

import torch
from torch.utils.data import Dataset, Sampler

from PIL import Image

from datasets.utils import Tokenizer
from utils.utils import pad_onehot_transpose


class CUBCaption(Dataset):
    def __init__(
            self,
            root: str,
            ctx_len: int,
            transform: Callable|None=None,
    ) -> None:

        assert os.path.isdir(root), f"Directory: {root} does not exist."
        self.root = root

        images_txt = os.path.join(self.root, "images.txt")
        assert os.path.isfile(images_txt), f"File: {images_txt} does not exist."
        with open(images_txt, "r") as f:
            self.img_lines = f.readlines()

        img_id_to_class_id = {}
        img_id_to_img_name = {}
        class_name_to_class_id = {}
        class_id_to_img_ids = defaultdict(list)
        for img_index, line in enumerate(self.img_lines):
            _, img_path = line.strip().split()
            class_name, img_name = img_path.split("/")

            if class_name not in class_name_to_class_id:
                class_name_to_class_id[class_name] = len(class_name_to_class_id)

            class_index = class_name_to_class_id[class_name]
            class_id_to_img_ids[class_index].append(img_index)
            img_id_to_class_id[img_index] = class_index
            img_id_to_img_name[img_index] = img_name

        # NLP settings 
        self.tokenizer = Tokenizer()
        self.pad_val = self.tokenizer.pad_token_id 
        self.vocab_size = len(self.tokenizer) 
        self.ctx_len = ctx_len

        # Index mappings
        self.img_id_to_class_id = img_id_to_class_id
        self.img_id_to_img_name = img_id_to_img_name
        self.class_name_to_class_index = class_name_to_class_id
        self.class_id_to_img_ids = class_id_to_img_ids

        # Image transform
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_lines)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # Load image
        _, img_name = self.img_lines[index].strip().split()
        img_path = os.path.join(self.root, "images", img_name)
        img = Image.open(img_path).convert("RGB")

        # Load matching text description 
        cap_name = img_name.replace("jpg", "txt")
        cap_path = os.path.join(self.root, "captions", cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        text_str = random.choice(captions)

        text_tok = self.tokenizer(text_str)
        text_tok = pad_onehot_transpose(text_tok, self.pad_val, self.ctx_len, self.vocab_size)

        # Load corresponding image class 
        label = self.img_id_to_class_id[index]
        
        if self.transform:
            img = self.transform(img)

        return img, text_str, text_tok, label

    def get_img_name(self, index: int) -> str:
        _, img_name = self.img_lines[index].strip().split()
        return img_name

    def get_all_captions(self, index: int) -> list[str]:
        _, img_name = self.img_lines[index].strip().split()
        cap_name = img_name.replace("jpg", "txt")
        cap_path = os.path.join(self.root, "captions", cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        return captions


class CUBGanCls(Dataset):
    def __init__(
            self,
            root: str,
            root_cap_emb: str,
            num_captions: int,
            classes: int,
            transform: Callable|None=None,
    ) -> None:
        """
        CUB-200-2011 : returns (image, caption_str, caption_embedding, class_label) 
        Used for GAN-INT.
        """ 
     
        assert os.path.isdir(root), f"Directory: {root} does not exist."
        assert os.path.isdir(root_cap_emb), f"Directory: {root_cap_emb} does not exist."

        self.root = root
        self.root_emb = root_cap_emb

        images_txt = os.path.join(self.root, "images.txt")
        assert os.path.isfile(images_txt), f"File: {images_txt} does not exist."
        with open(images_txt, "r") as f:
            self.img_lines = f.readlines()

        img_id_to_class_id = {}
        class_name_to_class_id = {}
        class_id_to_img_ids = defaultdict(list)
        for img_index, line in enumerate(self.img_lines):
            _, img_path = line.strip().split()
            class_name, img_name = img_path.split("/")

            if class_name not in class_name_to_class_id:
                class_name_to_class_id[class_name] = len(class_name_to_class_id)

            class_index = class_name_to_class_id[class_name]
            class_id_to_img_ids[class_index].append(img_index)
            img_id_to_class_id[img_index] = class_index

        self.img_id_to_class_id = img_id_to_class_id
        self.class_name_to_class_id = class_name_to_class_id
        self.class_id_to_img_ids = class_id_to_img_ids

        # Number of captions 
        self.num_captions = num_captions
        self.classes = classes

        # Image transform
        self.transform = transform

        self._load_embeddings()

    def _load_embeddings(self) -> None:
        embeddings = {}
        for index in range(len(self.img_lines)):
            _, img_path = self.img_lines[index].strip().split()
            _, img_name = img_path.split("/")
            
            txt_emb_name = img_name.replace("jpg", "pth")
            txt_emb_path = os.path.join(self.root_emb, txt_emb_name)

            txt_emb = torch.load(txt_emb_path, map_location="cpu")
            ids = torch.arange(self.num_captions)
            txt_emb = txt_emb[ids, :] 

            embeddings[index] = txt_emb

        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.img_lines)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # Load image I
        _, img_name = self.img_lines[index].strip().split()
        img_path = os.path.join(self.root, "images", img_name)
        img = Image.open(img_path).convert("RGB")

        class_index = self.img_id_to_class_id[index]
        r1, r2 = random.sample(self.classes, 2)
        rnd_class = r1 if r1 != class_index else r2
        rnd_index = random.choice(self.class_id_to_img_ids[rnd_class])
        _, rnd_img_name = self.img_lines[rnd_index].strip().split()
        rnd_img_path = os.path.join(self.root, "images", rnd_img_name)
        rnd_img = Image.open(rnd_img_path).convert("RGB")

        # Load embedding of image I 
        embeddings = self.embeddings[index] # [n_captions, 1024]
        emb_id = random.randint(0, self.num_captions - 1)
        txt_emb = embeddings[emb_id] 

        # Load text description of image I 
        cap_name = img_name.replace("jpg", "txt")
        cap_path = os.path.join(self.root, "captions", cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        txt_str = captions[0]
        
        # Load label of image I 
        label = self.img_id_to_class_id[index]

        if self.transform:
            img = self.transform(img)
            rnd_img = self.transform(rnd_img)

        return img, rnd_img, txt_emb, txt_str, label

    def get_img_name(self, index: int) -> str:
        _, img_name = self.img_lines[index].strip().split()
        return img_name

    def get_all_embeddings(self, index: int) -> torch.Tensor:
        embedding = self.embeddings[index]
        return embedding


class CUBTextEnc(Dataset):
    def __init__(
            self,
            root: str,
            root_google_features: str, 
            ctx_len: int,
            transform: Callable|None=None,
    ) -> None:
        assert os.path.isdir(root), f"Directory: {root} does not exist."
        assert os.path.isdir(root_google_features), f"Directory: {root_google_features} does not exist."

        self.root = root
        self.root_features = root_google_features

        images_txt = os.path.join(self.root, "images.txt")
        assert os.path.isfile(images_txt), f"File: {images_txt} does not exist."
        with open(images_txt, "r") as f:
            self.img_lines = f.readlines()

        img_id_to_class_id = {}
        img_id_to_img_name = {} 
        class_name_to_class_id = {}
        class_id_to_img_ids = defaultdict(list)
        
        for img_index, line in enumerate(self.img_lines):
            _, img_path = line.strip().split()
            class_name, img_name = img_path.split("/")

            if class_name not in class_name_to_class_id:
                class_name_to_class_id[class_name] = len(class_name_to_class_id)

            img_id_to_img_name[img_index] = img_name
            class_index = class_name_to_class_id[class_name]
            class_id_to_img_ids[class_index].append(img_index)
            img_id_to_class_id[img_index] = class_index

        self.tokenizer = Tokenizer()
        self.pad_val = self.tokenizer.pad_token_id 
        self.vocab_size = len(self.tokenizer) 
        self.ctx_len = ctx_len

        self.transform = transform

        self.img_id_to_class_id = img_id_to_class_id
        self.img_id_to_img_name = img_id_to_img_name 
        self.class_name_to_class_id = class_name_to_class_id
        self.class_id_to_img_ids = class_id_to_img_ids

        self._load_features()
        self._load_captions()

    def _load_features(self) -> None:
        # (32 bit) * 1024 = 32'768 bit = 4096 byte
        # => (4096 byte) * 11788 = 48'283'648 byte = 48 megabyte
        self.features = {}
        for index in range(len(self.img_lines)):
            _, img_path = self.img_lines[index].strip().split()
            _, img_name = img_path.split("/") 

            img_feat_name = img_name.replace("jpg", "pth") 
            img_feat_path = os.path.join(self.root_features, img_feat_name)
            img_feat = torch.load(img_feat_path, map_location="cpu")
            self.features[index] = img_feat
    
    def _load_captions(self) -> None:
        # (32 bit) * 10 * 70 * 201  = 4'502'400 bit = 562'800 byte
        # => 562'800 byte * 11'877 = 6'634'286'400 byte = 6634'2864 megabyte = 6,6 gigabyte
        pad = self.pad_val 
        ctx = self.ctx_len
        n_vocab = self.vocab_size

        self.captions = {}
        for index in range(len(self.img_lines)):
            _, img_name = self.img_lines[index].strip().split()
            cap_name = img_name.replace("jpg", "txt")
            cap_path = os.path.join(self.root, "captions", cap_name)
            
            with open(cap_path) as f: 
                captions = f.readlines() 
            
            captions = [pad_onehot_transpose(self.tokenizer(cap), pad, ctx, n_vocab) for cap in captions]
            self.captions[index] = torch.stack(captions, dim=0)     # [n_captions, 70, 201]

    def __len__(self) -> int:
        return len(self.img_lines)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        _, img_name = self.img_lines[index].strip().split()
        img_feat = self.features[index]

        cap_name = img_name.replace("jpg", "txt")
        cap_path = os.path.join(self.root, "captions", cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        captions = self.captions[index]
        
        label = self.img_id_to_class_id[index]

        captions = self.captions[index]
        
        return img_feat, captions, label

    def get_img_name(self, index: int) -> str:
        _, img_name = self.img_lines[index].strip().split()
        return img_name        


class CUBSampler(Sampler):
    def __init__(
            self,
            dataset: CUBTextEnc,
            target_classes: list[int],
            batch_size: int
    ) -> None:
        super().__init__()
        assert len(target_classes) >= batch_size, (
            f"Need at least batch_size classes, got {len(target_classes)}"
        )
        self.dataset = dataset
        self.target_classes = target_classes
        self.batch_size = batch_size

        # class_idx -> [img_idx, imd_idx, img_idx, ...]
        self.class_to_indices = {
            c : dataset.class_id_to_img_ids[c]
            for c in target_classes
        } 

        self.num_items = sum(len(img_idx_lst) for img_idx_lst in self.class_to_indices.values())
        self.num_batches = self.num_items // self.batch_size

    def __iter__(self):
        class_to_ramining = {c : idxs.copy() for c, idxs in self.class_to_indices.items()}

        for idxs in class_to_ramining.values():
            random.shuffle(idxs)

        for _ in range(self.num_batches):
            available_classes = [c for c, idxs in class_to_ramining.items() if len(idxs) > 0]

            if len(available_classes) < self.batch_size:
                break
            
            chosen_classes = random.sample(available_classes, self.batch_size)

            batch = []

            for c in chosen_classes:
                batch.append(class_to_ramining[c].pop())
            
            yield batch

    def __len__(self) -> int:
        return self.num_batches