import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

def loudness_match(a, b, eps=1e-8):
    # a, b : [B, 1, T] float
    rms_a = (a**2).mean(dim=-1, keepdim=True).sqrt()   # [B,1,1]
    rms_b = (b**2).mean(dim=-1, keepdim=True).sqrt()
    b_adj = b * (rms_a / (rms_b + eps))
    return b_adj.clamp(-1, 1)

# -----------------------------------
# 1) –12 dBFS 峰值归一化
# -----------------------------------
def peak_normalize(wav, target_dbfs=-12.0):
    # wav: Tensor [1, T], 假定振幅在 [-1,1]
    peak = wav.abs().max()
    if peak > 0:
        # target linear = 10^(dB/20)
        tgt = 10.0 ** (target_dbfs / 20.0)
        wav = wav * (tgt / peak)
    return wav
