import torch
import torch.nn as nn


class JointEmbeddingLoss(nn.Module):
    """
    Joint embedding loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    https://github.com/reedscot/icml2016
    """ 
    def __init__(self, margin: float=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, match_emb: torch.Tensor) -> torch.Tensor:
        # input shapes:                                             # emb: [N, 1024], match_emb: [N, 1024]
        batch_size = emb.size(0)
        score = match_emb @ emb.T                                   # [N, N] 
        label_score = torch.diagonal(score).unsqueeze(1)            # [N, 1]

        # thresh[i, j] = (s(i, j) - s(i, i)) + margin
        thresh = (score - label_score) + self.margin                # [N, N]
        
        # Exclude (i == j) 
        mask = ~torch.eye(batch_size, dtype=bool, device=score.device)

        # 1/N^2 * sum_ij [ max{0, s_ij - s_sii + margin} ] where i != j
        loss = torch.relu(thresh)[mask].sum() / (batch_size * batch_size)

        # Calculating top-1 accuracy
        with torch.no_grad(): 
            pred = score.argmax(dim=1)                              # [N,]
            target = torch.arange(score.size(0), device=emb.device) # [N,]
            acc = (pred == target).float().mean()                   # [1,]

        return loss, acc                                            # [1,], [1,]


class GeneratorLoss(nn.Module):
    """
    Generator loss.

    Reference:
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    """  
    def __init__(self) -> None:
        super().__init__() 
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, sf: torch.Tensor) -> torch.Tensor:
        loss_g = self.bce_with_logits(sf, torch.ones_like(sf)) 
        return loss_g


class DiscriminatorLoss(nn.Module):
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

    def forward(self, sr: torch.Tensor, sw: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
        # sr = D(real image, right text)
        # sw = D(real image, wrong text)
        # sf = D(fake image, right text)
        bce_sr = self.bce_with_logits(sr, torch.ones_like(sr))
        bce_sw = self.bce_with_logits(sw, torch.zeros_like(sw))
        bce_sf = self.bce_with_logits(sf, torch.zeros_like(sf))
        loss_d = bce_sr + self.cls_weight*bce_sw + self.cls_weight*bce_sf
        return loss_d