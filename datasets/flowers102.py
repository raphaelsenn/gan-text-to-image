import os
import random
from typing import Callable
from collections import defaultdict

from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler

from datasets.utils import Tokenizer
from utils.utils import pad_onehot_transpose


class FlowersCaption(Dataset):
    def __init__(
            self, 
            root_imgs: str, 
            root_captions: str,
            ctx_len: int, 
            transform: Callable|None=None
    ) -> None:

        assert os.path.isdir(root_imgs), f"Directory: {root_imgs} does not exist." 
        assert os.path.isdir(root_captions), f"Directory: {root_captions} does not exist." 

        self.root_imgs = root_imgs 
        self.root_captions = root_captions
        self.img_data = os.listdir(root_imgs)
        
        self.tokenizer = Tokenizer()
        self.pad_val = self.tokenizer.pad_token_id 
        self.vocab_size = len(self.tokenizer) 
        self.ctx_len = ctx_len
        self.transform = transform

        self.class_name_to_imgs = None
        self.img_id_to_class_name = None
        self._build_from_folder()

    def _build_from_folder(self) -> None:
        caption_folder = os.listdir(self.root_captions)

        class_name_to_imgs = defaultdict(list)  # class_name -> [image.jpg, ...]
        img_name_to_img_id = {}                 # image.jpg -> img_id, ...
        img_id_to_class_name = {}               # image.jpg -> class_name, ...

        img_id = 0
        for class_name in caption_folder:
            captions = os.listdir(os.path.join(self.root_captions, class_name))         
            for caption in captions:
                img = caption.replace(".txt", ".jpg")
                
                class_name_to_imgs[class_name].append(img)
                img_name_to_img_id[img] = img_id
                img_id_to_class_name[img_id] = class_name
                
                img_id += 1

        self.class_name_to_imgs = class_name_to_imgs
        self.img_name_to_img_id = img_name_to_img_id 
        self.img_id_to_class_name = img_id_to_class_name
        
        self.img_id_to_img_name = {val : key for key, val in self.img_name_to_img_id.items()} 
        self.class_id_to_class_name = {i : class_name for i, class_name in enumerate(self.class_name_to_imgs.keys())}
        self.class_name_to_class_id = {val : key for key, val in self.class_id_to_class_name.items()}

        class_id_to_img_ids = defaultdict(list)
        for class_name, img_lst in class_name_to_imgs.items():
            class_id = self.class_name_to_class_id[class_name]
            for img_name in img_lst:
                img_id = img_name_to_img_id[img_name]
                class_id_to_img_ids[class_id].append(img_id)

        self.class_id_to_img_ids = class_id_to_img_ids

    def __len__(self) -> None:
        return len(self.img_data)

    def __getitem__(self, index: int) -> tuple:
        img_name = self.img_id_to_img_name[index]
        img_path = os.path.join(self.root_imgs, img_name)
        img = Image.open(img_path).convert("RGB")

        img_class = self.img_id_to_class_name[index] 

        cap_name = img_name.replace(".jpg", ".txt") 
        cap_path = os.path.join(self.root_captions, img_class, cap_name)
        with open(cap_path, "r") as file:
            captions = file.readlines()
        text_str = random.choice(captions)

        if self.transform:
            img = self.transform(img)

        text_tok = self.tokenizer(text_str)
        text_tok = pad_onehot_transpose(text_tok, self.pad_val, self.ctx_len, self.vocab_size)

        return img, text_tok, text_str, img_class

    def get_all_captions(self, index: int) -> list[str]:
        img_name = self.img_id_to_img_name[index]
        cap_name = img_name.replace("jpg", "txt")
        class_name = self.img_id_to_class_name[index]
        cap_path = os.path.join(self.root_captions, class_name, cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        return captions


class FlowersTextEnc(Dataset):
    def __init__(
            self, 
            root_features: str, 
            root_captions: str,
            ctx_len: int, 
            transform: Callable|None=None
    ) -> None:

        assert os.path.isdir(root_features), f"Directory: {root_features} does not exist." 
        assert os.path.isdir(root_captions), f"Directory: {root_captions} does not exist." 

        self.root_features = root_features
        self.root_captions = root_captions
        self.img_data = os.listdir(root_features)
        
        self.tokenizer = Tokenizer()
        self.pad_val = self.tokenizer.pad_token_id
        self.vocab_size = len(self.tokenizer)
        self.ctx_len = ctx_len
        self.transform = transform

        self.class_name_to_imgs = None
        self.img_id_to_class_name = None
        self._build_from_folder()
        self._load_image_features()
        self._load_captions()

    def _build_from_folder(self) -> None:
        caption_folder = os.listdir(self.root_captions)

        class_name_to_imgs = defaultdict(list)  # class_name -> [image.jpg, ...]
        img_name_to_img_id = {}                 # image.jpg -> img_id, ...
        img_id_to_class_name = {}               # image.jpg -> class_name, ...

        img_id = 0
        for class_name in caption_folder:
            captions = os.listdir(os.path.join(self.root_captions, class_name))         
            for caption in captions:
                img = caption.replace(".txt", ".jpg")
                
                class_name_to_imgs[class_name].append(img)
                img_name_to_img_id[img] = img_id
                img_id_to_class_name[img_id] = class_name
                
                img_id += 1

        self.class_name_to_imgs = class_name_to_imgs
        self.img_name_to_img_id = img_name_to_img_id 
        self.img_id_to_class_name = img_id_to_class_name
        
        self.img_id_to_img_name = {val : key for key, val in self.img_name_to_img_id.items()} 
        self.class_id_to_class_name = {i : class_name for i, class_name in enumerate(self.class_name_to_imgs.keys())}
        self.class_name_to_class_id = {val : key for key, val in self.class_id_to_class_name.items()}

        class_id_to_img_ids = defaultdict(list)
        for class_name, img_lst in class_name_to_imgs.items():
            class_id = self.class_name_to_class_id[class_name]
            for img_name in img_lst:
                img_id = img_name_to_img_id[img_name]
                class_id_to_img_ids[class_id].append(img_id)

        self.class_id_to_img_ids = class_id_to_img_ids

    def _load_image_features(self) -> None:
        # (32 bit) * 1024 = 32'768 bit = 4096 byte
        # => (4096 byte) * 8189 = 33'542'144 byte ~ 33 megabyte
        self.features = {}
        for img_id, img_name in self.img_id_to_img_name.items():
            img_feat_name = img_name.replace("jpg", "pth") 
            img_feat_path = os.path.join(self.root_features, img_feat_name)
            img_feat = torch.load(img_feat_path, map_location="cpu")
            self.features[img_id] = img_feat

    def _load_captions(self) -> None:
        # (32 bit) * 10 * 70 * 201  = 4'502'400 bit = 562'800 byte
        # => 562'800 byte * 8'189 = 4'608'769'200 byte ~ 4,6 gigabyte
        pad = self.pad_val 
        ctx = self.ctx_len
        n_vocab = self.vocab_size

        self.captions = {}
        for img_id, img_name in self.img_id_to_img_name.items():
            class_name = self.img_id_to_class_name[img_id]
            cap_name = img_name.replace("jpg", "txt")
            cap_path = os.path.join(self.root_captions, class_name, cap_name)
            
            with open(cap_path) as f: 
                captions = f.readlines() 
            
            captions = [pad_onehot_transpose(self.tokenizer(cap), pad, ctx, n_vocab) for cap in captions]
            self.captions[img_id] = torch.stack(captions, dim=0)     # [n_captions, 70, 201]

    def __len__(self) -> None:
        return len(self.img_data)

    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        img_feat = self.features[index]
        class_name = self.img_id_to_class_name[index]
        img_class = self.class_name_to_class_id[class_name]
        text_tok = self.captions[index]

        return img_feat, text_tok, img_class


class FlowersGanCls(Dataset):
    def __init__(
            self, 
            root_imgs: str,
            root_captions: str,
            root_text_emb: str,
            classes: list[int],
            num_captions: int,
            transform: Callable|None=None
    ) -> None:

        assert os.path.isdir(root_imgs), f"Directory: {root_imgs} does not exist." 
        assert os.path.isdir(root_captions), f"Directory: {root_captions} does not exist." 
        assert os.path.isdir(root_text_emb), f"Directory: {root_text_emb} does not exist." 

        self.root_imgs = root_imgs
        self.root_captions = root_captions
        self.root_text_emb = root_text_emb 

        self.img_data = os.listdir(root_imgs)

        self.classes = classes
        self.num_captions = num_captions
        self.transform = transform

        self.class_name_to_imgs = None
        self.img_id_to_class_name = None
        self._build_from_folder()
        self._load_embeddings()

    def _build_from_folder(self) -> None:
        caption_folder = os.listdir(self.root_captions)

        class_name_to_imgs = defaultdict(list)  # class_name -> [image.jpg, ...]
        img_name_to_img_id = {}                 # image.jpg -> img_id, ...
        img_id_to_class_name = {}               # image.jpg -> class_name, ...

        img_id = 0
        for class_name in caption_folder:
            captions = os.listdir(os.path.join(self.root_captions, class_name))         
            for caption in captions:
                img = caption.replace(".txt", ".jpg")
                
                class_name_to_imgs[class_name].append(img)
                img_name_to_img_id[img] = img_id
                img_id_to_class_name[img_id] = class_name
                
                img_id += 1

        self.class_name_to_imgs = class_name_to_imgs
        self.img_name_to_img_id = img_name_to_img_id 
        self.img_id_to_class_name = img_id_to_class_name
        
        self.img_id_to_img_name = {val : key for key, val in self.img_name_to_img_id.items()} 
        self.class_id_to_class_name = {i : class_name for i, class_name in enumerate(self.class_name_to_imgs.keys())}
        self.class_name_to_class_id = {val : key for key, val in self.class_id_to_class_name.items()}

        class_id_to_img_ids = defaultdict(list)
        for class_name, img_lst in class_name_to_imgs.items():
            class_id = self.class_name_to_class_id[class_name]
            for img_name in img_lst:
                img_id = img_name_to_img_id[img_name]
                class_id_to_img_ids[class_id].append(img_id)

        self.class_id_to_img_ids = class_id_to_img_ids

    def _load_embeddings(self) -> None:
        embeddings = {}
        for img_id, img_name in self.img_id_to_img_name.items():
            txt_emb_name = img_name.replace("jpg", "pth")
            txt_emb_path = os.path.join(self.root_text_emb, txt_emb_name)

            txt_emb = torch.load(txt_emb_path, map_location="cpu")
            ids = torch.arange(self.num_captions)
            txt_emb = txt_emb[ids, :] 

            embeddings[img_id] = txt_emb

        self.embeddings = embeddings 

    def __len__(self) -> None:
        return len(self.img_data)


    def __getitem__(self, index: int|torch.Tensor) -> tuple:
        if isinstance(index, torch.Tensor):
            index = index.item()

        # Real image
        img_name = self.img_id_to_img_name[index]
        class_name = self.img_id_to_class_name[index]
        img_path = os.path.join(self.root_imgs, img_name)
        img = Image.open(img_path).convert("RGB")

        # Wrong image 
        class_id = self.class_name_to_class_id[class_name]
        r1, r2 = random.sample(self.classes, 2)
        rnd_class = r1 if r1 != class_id else r2
        rnd_index = random.choice(self.class_id_to_img_ids[rnd_class])
        rnd_img_name = self.img_id_to_img_name[rnd_index] 
        rnd_img_path = os.path.join(self.root_imgs, rnd_img_name)
        rnd_img = Image.open(rnd_img_path).convert("RGB")

        # Matching text (embedding)
        embeddings = self.embeddings[index]                 # [n_captions, 1024]
        emb_id = random.randint(0, self.num_captions - 1)
        txt_emb = embeddings[emb_id]

        # Matching text (string) 
        cap_name = img_name.replace("jpg", "txt")
        cap_path = os.path.join(self.root_captions, class_name, cap_name)
        with open(cap_path) as f: 
            captions = f.readlines()
        txt_str = captions[emb_id]
        
        # Matching class label 
        img_class = self.class_name_to_class_id[class_name] 

        if self.transform:
            img = self.transform(img)
            rnd_img = self.transform(rnd_img)

        return img, rnd_img, txt_emb, txt_str, img_class

    def get_all_embeddings(self, index: int) -> torch.Tensor:
        embedding = self.embeddings[index]
        return embedding


class FlowersSampler(Sampler):
    def __init__(
            self,
            dataset: FlowersCaption,
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