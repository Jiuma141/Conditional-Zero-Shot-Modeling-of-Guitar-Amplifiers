# train_simclr_with_tb.py
import os, torch, random
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms as T
import pandas as pd
from torch.utils.data import Dataset
import torchaudio


# ---------------------------
# 2) Encoder + Projection
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.models import resnet18

class ToneEncoder(nn.Module):
    def __init__(self, projection_dim=128, n_fft=1024, hop_length=256, sr=44100):
        super().__init__()
        # -----------------------------
        # 1) STFT 频谱前端 (线性频谱)
        # -----------------------------
        # power=None 表示输出复数谱，如果想直接拿幅度可以用 power=2 或 power=1
        # 这里我们先计算幅度谱再做 dB 变换
        self.spec = nn.Sequential(
            T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                power=2.0,          # 2.0 -> 输出幅度平方谱，后面做 db 会正确
                center=True,
                pad_mode="reflect",
                normalized=False,
            ),
            T.AmplitudeToDB(stype='power')   # 将 power 谱 -> dB
        )
        
        # -----------------------------
        # 2) ResNet-18 Backbone
        # -----------------------------
        resnet = resnet18(weights=None)     # torchvision>=0.13 用 weights=None；<0.13 用 pretrained=False
        # 把第一层 conv 的 in_channels 改成 1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉原来最后的 avgpool+fc，保留到倒数第二层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features

        # -----------------------------
        # 3) Projection Head
        # -----------------------------
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim)
        )

    def forward(self, x):
        """
        x: Tensor [B, 1, T]  time-domain waveform
        """
        # [B,1,T] -> [B,T]
        x = x.squeeze(1)

        # 1) 计算线性 STFT power 谱，并转 dB
        #    输出 shape = [B, n_fft//2+1, T']
        spec = self.spec(x)

        # 2) 增加 channel 维度 -> [B,1, freq, time]
        spec = spec.unsqueeze(1)

        # 3) 送入 CNN backbone -> [B, C, 1, 1]
        h = self.backbone(spec)

        # 4) flatten -> [B, C]
        h = h.flatten(1)

        # 5) projection + L2 normalize
        z = self.projector(h)
        z = F.normalize(z, dim=1)

        return z

# # ---------------------------
# # 3) NT-Xent Loss
# # ---------------------------
# def nt_xent_loss(z1, z2, tau=0.5):
#     """
#     z1, z2: [B, D] (already normalized)
#     """
#     B = z1.size(0)
#     z = torch.cat([z1,z2], dim=0)          # [2B, D]
#     sim = torch.matmul(z, z.t())           # [2B,2B] 余弦等价
#     sim = sim / tau

#     # mask out self similarities
#     mask = (~torch.eye(2*B, device=sim.device).bool()).float()

#     # numerator: exp(sim(i, pos(i)))
#     pos = torch.cat([torch.arange(B, 2*B), torch.arange(0,B)]).to(sim.device)
#     nom = torch.exp(sim[torch.arange(2*B), pos])

#     den = (mask * torch.exp(sim)).sum(dim=1)

#     loss = -torch.log(nom/den).mean()
#     return loss
