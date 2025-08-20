import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from helper import peak_normalize, loudness_match

class CSVSupToneDataset(Dataset):
    """
    读取一个 CSV，必须包含列：
      - 'filename'：wav 文件路径
      - 'model'：该 wav 对应的音箱／型号标签
      - 'gain','treble','mid','bass'：四个参数列
    仅保留这四列都非空的行。
    """
    def __init__(self, csv_path, sample_rate=44100, clip_sec=1.0):
        df = pd.read_csv(csv_path)
        # 确保必要列都存在
        required = ['filename', 'model', 'gain', 'treble', 'mid', 'bass']
        for c in required:
            if c not in df.columns:
                raise ValueError(f"CSV 中缺少列：{c}")

        # 过滤：只保留 gain/treble/mid/bass 都不为空的行
        df = df[df[['gain','treble','mid','bass']].notnull().all(axis=1)].reset_index(drop=True)
        self.df = df

        self.sr = sample_rate
        self.clip_len = int(clip_sec * sample_rate)

        # 如果 model 是字符串，就映射到数字标签
        if self.df['model'].dtype == object:
            models = sorted(self.df['model'].unique().tolist())
            self.model2idx = {m: i for i, m in enumerate(models)}
            # 新增一列 'label_idx'
            self.df['label_idx'] = self.df['model'].map(self.model2idx)
        else:
            # 已经是数值型的话直接转 int
            self.df['label_idx'] = self.df['model'].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 1. 读 wav
        wav, sr = torchaudio.load(row['filename'])
        # 2. 重采样
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        # 3. 转单通道
        wav = wav.mean(dim=0, keepdim=True)  # [1, L]
        L = wav.size(1)
        # 4. 随机裁两段
        if L >= self.clip_len:
            s1 = random.randint(0, L - self.clip_len)
            s2 = random.randint(0, L - self.clip_len)
            c1 = wav[:, s1:s1+self.clip_len]
            c2 = wav[:, s2:s2+self.clip_len]
        else:
            pad = self.clip_len - L
            c1 = F.pad(wav, (0, pad))
            c2 = c1

        label = row['label_idx']
        return c1, c2, label

class AmpPairDataset(Dataset):
    def __init__(self, df, sr=48000, clip_sec=1, device='cuda'):
        self.df      = df.reset_index(drop=True)
        # self.encoder = tone_encoder.eval()        # **不 to(device)**
        self.sr      = sr
        self.clip_len= int(sr*clip_sec)
        self.device  = device                     # 存一下

        # 在 __init__ 里，计算每个 model 的 min/max
        # self.knob_cols = ['gain','mid','bass','treble']
        self.knob_cols = ['gain']
        group = self.df.groupby('model')[self.knob_cols]
        # 得到两个 DataFrame：索引是 model，列是 knob
        self.knob_min_by_model = group.min()     # shape [num_models × 4]
        self.knob_max_by_model = group.max()
        self.knob_range_by_model = (self.knob_max_by_model - self.knob_min_by_model).replace(0,1e-6)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # 读取 wav
        x, sr1 = torchaudio.load(row['di_file'])
        y, sr2 = torchaudio.load(row['filename'])
        z, sr3 = torchaudio.load(row['filename'])
        # 单通道
        x = x.mean(0, keepdim=True)
        y = y.mean(0, keepdim=True)
        z = z.mean(0, keepdim=True)
        # 重采样
        if sr1 != self.sr: x = torchaudio.functional.resample(x, sr1, self.sr)
        if sr2 != self.sr: y = torchaudio.functional.resample(y, sr2, self.sr)
        if sr3 != self.sr: z = torchaudio.functional.resample(z, sr3, self.sr)
        # 峰值归一化
        x = peak_normalize(x)
        y = peak_normalize(y)
        y = loudness_match(x, y)  
        # 随机裁剪 3.5s
        def crop(w):
            L = w.size(1)
            if L >= self.clip_len:
                s = torch.randint(0, L - self.clip_len + 1, ())
                return w[:, s:s+self.clip_len]
            else:
                return F.pad(w, (0, self.clip_len - L))
        x = crop(x); y = crop(y); z = crop(z)
        
        row = self.df.iloc[i]
        model = row['model']  
        knob_min   = torch.tensor(self.knob_min_by_model.loc[model].values,
                          device=self.device).float()
        knob_range = torch.tensor(self.knob_range_by_model.loc[model].values,
                                  device=self.device).float()
        raw = row[self.knob_cols].astype(float).to_numpy()
        knobs = (torch.tensor(raw, device=self.device).float() - knob_min) / knob_range
        # # 计算 tone embedding φ (512 → 128)
        # # 先把 z 也搬到 device 上
        # z = z.to(self.device)
        # with torch.no_grad():
        #     φ = self.encoder(z)
        # φ = φ.squeeze(0)
        # # ——3. φ 和 x,y 一并搬到 GPU——
        # x = x.to(self.device)
        # y = y.to(self.device)

        return x, y, z, knobs
    
class AmpPairDatasetCPU(Dataset):
    def __init__(self, df, sr=48000, clip_sec=1.0):
        self.df       = df.reset_index(drop=True)
        self.sr       = sr
        self.clip_len = int(sr * clip_sec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, sr1 = torchaudio.load(row["di_file"])      # [1, T]
        y, sr2 = torchaudio.load(row["filename"])     # [1, T]
        # mono
        x = x.mean(0, True);  y = y.mean(0, True)
        # resample
        if sr1 != self.sr: x = torchaudio.functional.resample(x, sr1, self.sr)
        if sr2 != self.sr: y = torchaudio.functional.resample(y, sr2, self.sr)
        # normalize & match
        x = peak_normalize(x);  y = peak_normalize(y)
        y = loudness_match(x, y)
        # random crop to clip_len
        L = x.size(-1)
        if L >= self.clip_len:
            s = torch.randint(0, L - self.clip_len + 1, ())
            x = x[:, s : s+self.clip_len]
            y = y[:, s : s+self.clip_len]
        else:
            pad = self.clip_len - L
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))
        # 返回 CPU tensors + 路径用于加载 reference
        return x, y, row["filename"]

