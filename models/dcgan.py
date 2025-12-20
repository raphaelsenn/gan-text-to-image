import torch
import torch.nn as nn


class ProjectManifold(nn.Module):
    """Projects the text-emedding (output from CNN-RNN) to a lower dimension.""" 
    def __init__(self, emb_in: int=1024, emb_out: int=128) -> None:
        super().__init__()

        # [N, emb_in] -> [N, emb_out] 
        self.proj = nn.Sequential(
            nn.Linear(emb_in, emb_out, bias=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ProjectManifoldBN(nn.Module):
    """Projects the text-emedding (output from CNN-RNN) to a lower dimension.""" 
    def __init__(self, emb_in: int=1024, emb_out: int=128) -> None:
        super().__init__()

        # [N, emb_in] -> [N, emb_out] 
        self.proj = nn.Sequential(
            nn.Linear(emb_in, emb_out, bias=True),
            nn.BatchNorm1d(emb_out), 
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ResidualBottleneck4x4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, act: str='leaky') -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # [N, in_ch, 4, 4] -> [N, mid_ch, 4, 4]
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            self._make_act(act),

            # [N, mid_ch, 4, 4] -> [N, mid_ch, 4, 4]
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            self._make_act(act),

            # [N, mid_ch, 4, 4] -> [N, in_ch, 4, 4]
            nn.Conv2d(mid_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),

        )
        self.act = self._make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x) 
        return self.act(x + y) 

    @staticmethod
    def _make_act(act: str='leaky') -> nn.Module:
        return nn.LeakyReLU(0.2, True) if act.lower() == 'leaky' else nn.ReLU(True)


class ResidualBottleneck8x8(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # [N, in_ch, 8, 8] -> [N, mid_ch, 8, 8]
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(True),

            # [N, mid_ch, 8, 8] -> [N, mid_ch, 8, 8]
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(True),

            # [N, mid_ch, 8, 8] -> [N, in_ch, 8, 8]
            nn.Conv2d(mid_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.act = nn.ReLU(True) 
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x) 
        return self.act(x + y)


class Generator(nn.Module):
    """
    Text-conditioned deep convolutional generater.

    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661 
    """
    def __init__(
            self, 
            nz: int=100,
            nt: int=128,
            emb_in: int=1024,
            ngf: int=64,
            channels_img: int=3,
    ) -> None:
        super().__init__() 
        self.ngf = ngf

        # [N, 1024] -> [N, 128]
        self.proj_manifold = ProjectManifoldBN(emb_in, nt)
        
        # [N, 100 + 128, 1, 1] -> [N, 8*ngf, 4, 4] 
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nz + nt, 8*ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * ngf),
        ) 

        # [N, 8*ngf, 4, 4] -> [N, 2*ngf, 4, 4] ... [N, 2*ngf, 4, 4]-> [N, 8*ngf, 4, 4]
        self.res4x4 = ResidualBottleneck4x4(8*ngf, 2*ngf, act='relu')

        # [N, ngf*8, 4, 4] -> [N, ngf*4, 8, 8]
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 4*ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*ngf),
        )

        # [N, 4*ngf, 8, 8] -> [N, ngf, 8, 8] ... [N, ngf, 8, 8]-> [N, 4*ngf, 8, 8]
        self.res8x8 = ResidualBottleneck8x8(4*ngf, ngf)

        self.out = nn.Sequential(
            # [N, 4*ngf, 8, 8] -> [N, 2*ngf, 16, 16]
            nn.ConvTranspose2d(4*ngf, 2*ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(True),

            # [N, 2*ngf, 16, 16] -> [N, ngf, 32, 32]
            nn.ConvTranspose2d(2*ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [N, ngf, 32, 32] -> [N, channels_img, 64, 64]
            nn.ConvTranspose2d(ngf, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def forward(self, noise: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        text = self.proj_manifold(text)
        text = text.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([noise, text], dim=1)

        x = self.up1(x)
        x = self.res4x4(x) 
        
        x = self.up2(x)
        x = self.res8x8(x)

        return self.out(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Text-conditioned deep convolutional discriminator.
 
    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    
    Generative Adversarial Text to Image Synthesis, Reed et al., 2016
    https://arxiv.org/pdf/1605.05396 
    
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661 
    """ 
    def __init__(
            self,
            nt: int=128, 
            emb_in: int=1024,
            ndf: int=64,
            channels_img: int=3,
    ) -> None:
        super().__init__()
        self.ndf = ndf

        self.convD = nn.Sequential(
            # [N, channels_img, 64, 64] -> [N, ndf, 32, 32]
            nn.Conv2d(channels_img, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            
            # [N, ndf, 32, 32] -> [N, 2*ndf, 16, 16]
            nn.Conv2d(ndf, 2*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2, True),
            
            # [N, 2*ndf, 16, 16] -> [N, 4*ndf, 8, 8]
            nn.Conv2d(2*ndf, 4*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2, True),

            # [N, 4*ndf, 8, 8] -> [N, 8*ndf, 4, 4]
            nn.Conv2d(4*ndf, 8*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*ndf),
        )

        # [N, 8*ndf, 4, 4] -> [N, 2*ndf, 4, 4] ... [N, 2*ndf, 4, 4]-> [N, 8*ndf, 4, 4]
        self.res4x4 = ResidualBottleneck4x4(8*ndf, 2 * ndf, act='leaky') 
        
        # [N, 1024] -> [N, 128]
        self.proj_manifold = ProjectManifoldBN(emb_in, nt)

        self.fuse = nn.Sequential(
            # [N, 8*ndf + 128, 4, 4] -> [N, 8*ndf, 4, 4]
            nn.Conv2d(ndf*8 + nt, ndf*8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(0.2, True),

            # [N, 8*ndf, 4, 4] -> [N, 1, 1, 1]
            nn.Conv2d(8*ndf, 1, 4, 1, 0, bias=False)
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.convD(x)
        x = self.res4x4(x) 

        t = self.proj_manifold(t)
        t = t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3]).contiguous() 
        x = torch.cat([x, t], dim=1)
        return self.fuse(x).view(-1, 1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)