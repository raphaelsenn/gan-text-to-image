import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEmbeddingLoss(nn.Module):
    """
    Deep Structured Joint Embedding Loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    https://github.com/reedscot/icml2016
    """ 
    def __init__(self, margin: float=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_1: torch.Tensor, emb_2: torch.Tensor) -> torch.Tensor:
        # input shapes:                                         # emb: [N, 1024], match_emb: [N, 1024]
        batch_size = emb_1.size(0)

        score = emb_2 @ emb_1.T                                 # [N, N] 
        label_score = torch.diag(score).view(batch_size, 1)     # [N, 1]

        # thresh[i, j] = (s(i, j) - s(i, i)) + margin
        thresh = (score - label_score) + self.margin            # [N, N]
        
        # Exclude (i == j) 
        mask = ~torch.eye(batch_size, dtype=bool, device=score.device)

        # 1/N^2 * sum_ij [ max{0, s_ij - s_sii + margin} ] where i != j
        loss = F.relu(thresh)[mask].sum() / (batch_size * (batch_size - 1))

        # Calculating top-1 accuracy
        with torch.no_grad(): 
            pred = score.argmax(dim=1)                                # [N,]
            target = torch.arange(score.size(0), device=emb_1.device) # [N,]
            acc = (pred == target).float().mean()                     # [1,]

        return loss, acc                                              # [1,], [1,]


class GeneratorLoss(nn.Module):
    """
    Generator loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661  
    """  
    def __init__(self) -> None:
        super().__init__() 
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, fake_score: torch.Tensor) -> torch.Tensor:
        loss_g = self.bce_with_logits(fake_score, torch.ones_like(fake_score)) 
        return loss_g


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661  
    """ 
    def __init__(self, cls_weight: float=0.5) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
        bce_real = self.bce_with_logits(real_score, torch.ones_like(real_score))
        bce_fake = self.bce_with_logits(fake_score, torch.zeros_like(fake_score))
        loss_d = bce_real + bce_fake
        return loss_d


class DiscriminatorLossCLS(nn.Module):
    """
    Discriminator cls-loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    """ 
    def __init__(self, cls_weight: float=0.5) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, real_score: torch.Tensor, wrong_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
        bce_real = self.bce_with_logits(real_score, torch.ones_like(real_score))
        bce_wrong = self.bce_with_logits(wrong_score, torch.zeros_like(wrong_score))
        bce_fake = self.bce_with_logits(fake_score, torch.zeros_like(fake_score))
        loss_d = bce_real + self.cls_weight*bce_wrong + (1-self.cls_weight)*bce_fake
        return loss_d
    

