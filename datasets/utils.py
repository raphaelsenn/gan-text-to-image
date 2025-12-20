import torch


class Tokenizer:
    """Character-level tokenizer."""
    def __init__(self) -> None:
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@$%^&~`*#+=<>()[]{} "
        self.pad_token_id = 0
        self.stoi = {c : i+1 for i, c in enumerate(alphabet)}
        self.itos = {val : key for key, val in self.stoi.items()}

    def encode(self, string: str) -> list[int]:
        """
        >>> tokenizer = Tokenizer()
        >>> tokenizer.encode('abcdefg')
        [1, 2, 3, 4, 5, 6, 7]
        """ 
        tokens = []
        for s in string.lower():
            if s not in self.stoi:
                continue
            tokens.append(self.stoi[s])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        >>> tokenizer = Tokenizer()
        >>> tokenizer.decode([1, 2, 3, 4, 5, 6, 7])
        'abcdefg'
        """ 
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join(self.itos.get(tok, "") for tok in tokens)

    def __len__(self) -> int:
        """
        >>> tokenizer = Tokenizer()
        >>> len(tokenizer)
        70
        """
        return len(self.stoi) + 1   # silently counting the padding token 0 :D

    def __call__(self, string: str) -> list[int]:
        """
        >>> tokenizer = Tokenizer()
        >>> tokenizer('abcdefg')
        [1, 2, 3, 4, 5, 6, 7]
        """  
        return self.encode(string) 