import torch
import torch.nn as nn
import numpy as np

class SubnetST(nn.Module):
    """
    Lightweight Depthwise Separable Convolutional Subnetwork s(.) and t(.).
    Generates non-linear transformation shifts for INN coupling layers.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale output to integer shift range [0, 255]
        out = (self.net(x) + 1.0) * 127.5
        return torch.floor(out)

class IntegerCouplingBlock(nn.Module):
    """
    Bit-Exact Invertible Neural Coupling Block (Z -> Z).
    Guarantees exact integer reversibility under clean transmission.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        assert in_channels % 2 == 0, "Input channels must be even"
        half_ch = in_channels // 2
        self.s1 = SubnetST(half_ch)
        self.t1 = SubnetST(half_ch)
        self.s2 = SubnetST(half_ch)
        self.t2 = SubnetST(half_ch)

    def forward(self, u: torch.Tensor, chaos_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass (Encryption).
        u: torch.Tensor of shape (B, C, H, W), dtype uint8 or float32 (values in [0, 255])
        chaos_matrix: (B, C, H, W) integer byte matrix from chaotic generator
        """
        u1, u2 = torch.chunk(u, 2, dim=1)
        c1, c2 = torch.chunk(chaos_matrix, 2, dim=1)
        
        # Step 1: v1 = u1 XOR floor(c1)
        v1 = (u1.long() ^ c1.long()).float()
        
        # Step 2: v2 = (u2 + floor(s1(v1))) mod 256
        shift1 = self.s1(v1)
        v2 = torch.remainder(u2.long() + shift1.long(), 256).float()
        
        # Step 3: w1 = (v1 + floor(t1(v2))) mod 256
        shift2 = self.t1(v2)
        w1 = torch.remainder(v1.long() + shift2.long(), 256).float()
        
        # Step 4: w2 = v2 XOR floor(c2)
        w2 = (v2.long() ^ c2.long()).float()
        
        w = torch.cat([w1, w2], dim=1)
        return w

    def reverse(self, w: torch.Tensor, chaos_matrix: torch.Tensor) -> torch.Tensor:
        """
        Reverse Pass (Decryption).
        Reconstructs exact original input u.
        """
        w1, w2 = torch.chunk(w, 2, dim=1)
        c1, c2 = torch.chunk(chaos_matrix, 2, dim=1)
        
        # Step 4 reverse: v2 = w2 XOR floor(c2)
        v2 = (w2.long() ^ c2.long()).float()
        
        # Step 3 reverse: v1 = (w1 - floor(t1(v2))) mod 256
        shift2 = self.t1(v2)
        v1 = torch.remainder(w1.long() - shift2.long(), 256).float()
        
        # Step 2 reverse: u2 = (v2 - floor(s1(v1))) mod 256
        shift1 = self.s1(v1)
        u2 = torch.remainder(v2.long() - shift1.long(), 256).float()
        
        # Step 1 reverse: u1 = v1 XOR floor(c1)
        u1 = (v1.long() ^ c1.long()).float()
        
        u = torch.cat([u1, u2], dim=1)
        return u
