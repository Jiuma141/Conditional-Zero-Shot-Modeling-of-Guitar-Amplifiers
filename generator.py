import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

class FiLMGCNBlock(nn.Module):
    def __init__(self, C, dilation):
        super().__init__()
        self.filt = nn.Conv1d(C, C, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(C, C, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.to_gamma = nn.Linear(128, C, bias=True)
        self.to_beta  = nn.Linear(128, C, bias=True)
        self.res_conv = nn.Conv1d(C, C, kernel_size=1)
    def forward(self, x, φ):
        # 打印一下进入和离开这个 block 的形状
        # print("  FiLM in:", x.shape, φ.shape)
        h = torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))
        γ = self.to_gamma(φ).unsqueeze(-1)
        β = self.to_beta(φ).unsqueeze(-1)
        h = γ * h + β
        out = self.res_conv(h) + x[..., :h.size(-1)]
        # print("  FiLM out:", out.shape)
        return out

class GeneratorFiLMGCN(nn.Module):
    def __init__(self, in_ch=1, C=32, L=16):
        super().__init__()
        self.init_conv = nn.Conv1d(in_ch, C, kernel_size=1)
        self.blocks = nn.ModuleList([
            FiLMGCNBlock(C, dilation=2**i) for i in range(L)
        ])
        self.final_conv = nn.Conv1d(C*L, in_ch, kernel_size=1)
    def forward(self, x, φ):
        h = self.init_conv(x)
        # print("after init_conv:", h.shape)  # 应该是 [B, C, T]
        outs = []
        for blk in self.blocks:
            # print("  block input h:", h.shape, "φ:", φ.shape)
            h = blk(h, φ)
            outs.append(h)
        cat = torch.cat(outs, dim=1)  # [B,16*12,T]
        return self.final_conv(cat)
    
# -----------------------------
# 2) 条件编码网络
# -----------------------------
class CondNet(nn.Module):
    def __init__(self, in_dim=1, hid=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, c):
        return self.net(c)  # [B, 128]

    import torch.nn as nn

class FusionMLP(nn.Module):
    """
    Fuse tone embedding φ_z and knob embedding φ_c by concatenation → MLP projection.
    Input dim: Dz + Dc (e.g. 128+128=256)
    Output dim: Dout (e.g. 128)
    """
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, phi_cat):
        """
        phi_cat: Tensor of shape [B, in_dim]
        returns:  Tensor of shape [B, out_dim]
        """
        return self.net(phi_cat)

class ResidualFusionMLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, z, c):
        fused = self.net(torch.cat([z, c], dim=-1))
        return z + self.norm(fused)

class DeepFusionMLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim, eps=1e-6)
        )
        # 如果 in_dim == out_dim，就可以加残差，否则略去
        if in_dim == out_dim:
            self.residual = True
        else:
            self.residual = False

    def forward(self, phi_cat):
        y = self.net(phi_cat)
        if self.residual:
            # 残差只对维度匹配的情况
            y = y + phi_cat
        return y
