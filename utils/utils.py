import random
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from PIL import Image


def add_padding(x: torch.Tensor, padding_value: int, max_seq_len: int) -> torch.Tensor:
    """Pad or truncate a tensor of shape [seq_len,] -> [max_seq_len,]."""
    seq_len = x.shape[0]
    if seq_len > max_seq_len:
        return x[:max_seq_len]
    padding_length = max(0, max_seq_len - seq_len)
    x = F.pad(x, pad=(0, padding_length), value=padding_value)
    return x


def randint_except(low: int, high: int, numb_not: int) -> int:
    """Returns a random uniform-integer in [0, high) (except the element num_not)."""
    assert low <= numb_not < high, "numb_not must be inside [low, high)" 
    number = torch.randint(low, high-1, (1,)).item()
    if number >= numb_not:
        return number + 1
    return number


def ensure_rgb(img: Image.Image) -> Image:
    """Convertes non RGB images to RGB.""" 
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def pad_onehot_transpose(tokens: torch.Tensor | list[int], pad_val: int, ctx_len: int, num_classes: int) -> torch.Tensor:
    """Adds padding, performs one_hot encoding. Outputs a tensor of shape [1, vocab_size, ctx_len]""" 
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens, dtype=torch.long) # [n,]
    tokens = add_padding(tokens, pad_val, ctx_len)      # [ctx_len,]
    tokens = F.one_hot(tokens, num_classes).float()     # [ctx_len, vocab_size] 
    tokens = tokens.transpose(1, 0)                     # [vocab_size, ctx_len]
    return tokens


def filter_ascii(s: str) -> str:
    ascii_pattern = re.compile(r'[^\x00-\x7F]')
    return ascii_pattern.sub(" ", s)


def set_seeds(seed: int, deterministic: bool=False) -> None:
    """Sets random seeds for reproducability for both CPU and GPU.""" 
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic: 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True