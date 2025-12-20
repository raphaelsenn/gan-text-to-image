import torch
import torch.nn as nn


class CNNRNNEncoder(nn.Module):
    """
    Deep convolutional-recurrent text encoder network.

    Reference: 
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 

    Learning Deep Representations of Fine-grained Visual Descriptions, Reed et al., 2016 
    https://arxiv.org/abs/1605.05395 
    """ 
    def __init__(
            self, 
            vocab_size: int, 
            cnn_dim: int=512, 
            emb_dim: int=1024, 
            avg: bool=True,
            bidirectional: bool=False
    ) -> None:
        super().__init__()
        self.cnn_dim = cnn_dim
        self.emb_dim = emb_dim
        self.avg = avg
        self.bidirectional = bidirectional

        self.cnn = nn.Sequential(
            # [N, vocab_size, 201] -> [N, 384, 66]
            nn.Conv1d(vocab_size, 384, kernel_size=4),
            nn.ReLU(),
            # nn.Threshold(1e-6, 0.0), 
            nn.MaxPool1d(3, 3),
            
            # [N, 384, 66] -> [N, cnn_dim, 21]
            nn.Conv1d(384, 512, kernel_size=4),
            nn.ReLU(),
            # nn.Threshold(1e-6, 0.0), 
            nn.MaxPool1d(3, 3),
            
            # [N, cnn_dim, 21] -> [N, cnn_dim, 8]
            nn.Conv1d(512, cnn_dim, kernel_size=4),
            # nn.Threshold(1e-6, 0.0), 
            nn.ReLU(),
        )

        hidden_dim = 2*cnn_dim if bidirectional else cnn_dim

        # [N, cnn_dim, 18] -> [N, hidden_dim] 
        self.gru = nn.GRU(cnn_dim, cnn_dim, batch_first=True, bidirectional=bidirectional)

        # [N, hidden_dim] -> [N, emb_dim]
        self.out = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                 # [N, cnn_dim, 18]
        x = x.transpose(1, 2)           # [N, 18, cnn_dim]
        out, _ = self.gru(x)            # [N, 18, cnn_dim]
        if self.avg: 
            h = out.mean(dim=1)         # [N, cnn_dim]
        else:
            h = out[:, -1, :]           # [N, cnn_dim]
        return self.out(h)              # [N, emb_dim]