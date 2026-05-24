import math
import torch
import torch.nn as nn
from torchaudio.functional import lfilter
from .autograd import Iso_SRS



class CudaSRS(nn.Module):
    def __init__(self, freqs: torch.Tensor, damping=0.03, fs=32768):
        super().__init__()

        freqs = freqs.double()
        self.fs = fs
        self.damping = damping

        Q = 1.0 / (2.0 * damping)
        omegas = 2.0 * math.pi * freqs

        A = omegas * (1.0 / (2.0 * Q * fs))
        B = (omegas / fs) * math.sqrt(1.0 - 1.0 / (4.0 * Q**2))

        exp_A = torch.exp(-A)
        exp_2A = torch.exp(-2.0 * A)
        cos_B = torch.cos(B)

        sinB_over_B = torch.where(
            torch.abs(B) < 1e-12,
            torch.ones_like(B),
            torch.sin(B) / B
        )

        # We don't need the a0 = 1 for the kernel, we can do it implicitly
        As = torch.stack([
            -2.0 * exp_A * sinB_over_B,
            exp_2A,
        ], dim=1)

        Bs = torch.stack([
            1.0 - exp_A * sinB_over_B,
            2.0 * exp_A * (sinB_over_B - cos_B),
            -2.0 * exp_A * cos_B,
        ], dim=1)


        self.register_buffer("As", As.contiguous())
        self.register_buffer("Bs", Bs.contiguous())

    def forward(self, x):
        return Iso_SRS.apply(x, self.As, self.Bs)
    






class TorchSRS(nn.Module):
    def __init__(self, freqs: torch.Tensor, damping: float = 0.03, fs: int = 32768):
        super().__init__()

        freqs = freqs.double()
        self.fs = fs
        self.damping = damping

        Q = 1.0 / (2.0 * damping)
        omegas = 2.0 * math.pi * freqs

        A = omegas * (1.0 / (2.0 * Q * fs))
        B = (omegas / fs) * math.sqrt(1.0 - 1.0 / (4.0 * Q**2))

        exp_A = torch.exp(-A)
        exp_2A = torch.exp(-2.0 * A)
        cos_B = torch.cos(B)

        sinB_over_B = torch.where(
            torch.abs(B) < 1e-12,
            torch.ones_like(B),
            torch.sin(B) / B,
        )
 

        As = torch.stack([
            torch.ones_like(freqs),
            -2.0 * exp_A * sinB_over_B,
            exp_2A,
        ], dim=1)

        Bs = torch.stack([
            1.0 - exp_A * sinB_over_B,
            2.0 * exp_A * (sinB_over_B - cos_B),
            -2.0 * exp_A * cos_B,
        ], dim=1)

        self.register_buffer("As", As.contiguous())
        self.register_buffer("Bs", Bs.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.double()
        B, T = x.shape
        F = self.As.size(0)

        # (B, 1, T) -> (B, F, T)
        x_f = x.unsqueeze(1).expand(B, F, T).contiguous()

        return lfilter(x_f, self.As, self.Bs, clamp=False).abs().amax(dim=-1)
