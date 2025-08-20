"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    
class SpectralLoss(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop_length
    def forward(self, y_hat, y):
        win = torch.hann_window(self.n_fft,
                                     device=y_hat.device,
                                     dtype=y_hat.dtype)
        # [B,1,T] → complex STFT
        S1 = torch.stft(y_hat.squeeze(1), self.n_fft, self.hop, window=win,
                        return_complex=True)
        S2 = torch.stft(y      .squeeze(1), self.n_fft, self.hop, window=win,
                        return_complex=True)
        return F.mse_loss(S1.real, S2.real) + F.mse_loss(S1.imag, S2.imag)
    
    
class MRSTFTLoss(nn.Module):
    def __init__(self, ffts=(256, 512, 1024, 2048), hop_ratio=0.25, w_log=True, alpha_spec=0.8, alpha_time=0.2):
        super().__init__()
        self.ffts = ffts
        self.hops = [int(n * hop_ratio) for n in ffts]
        self.w_log = w_log        # True=log1p, False=linear
        self.alpha_spec = alpha_spec
        self.alpha_time = alpha_time

    def stft_mag(self, x, n_fft, hop):
        win = torch.hann_window(n_fft, device=x.device)
        S = torch.stft(x.squeeze(1), n_fft, hop, window=win, return_complex=True)
        mag = S.abs()
        return torch.log1p(mag) if self.w_log else mag

    def forward(self, y_hat, y):
        loss = 0.
        for n, h in zip(self.ffts, self.hops):
            M1 = self.stft_mag(y_hat, n, h)
            M2 = self.stft_mag(y,     n, h)
            loss += self.alpha_spec * F.l1_loss(M1, M2)
        # 叠加少量时域 L1
        loss += self.alpha_time * F.l1_loss(y_hat, y)
        return loss / len(self.ffts)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexSpectralLoss(nn.Module):
    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = None,
                 window: torch.Tensor = None,
                 reduction: str = 'mean',
                 alpha_time: float = 0.1,
                 alpha_rms:  float = 0.1):
        """
        Args:
            n_fft, hop_length, win_length, window, reduction: 同你之前的定义
            alpha_time: 时域 L1 损失的权重
            α_rms:  RMS 能量（响度）对齐损失权重
        """
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.reduction  = reduction
        self.alpha_time     = alpha_time
        self.alpha_rms      = alpha_rms

        # 如果没传 window，就在 forward 时按 device/dtype 动态创建
        self.register_buffer('window', window, persistent=False)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat, y: [B,1,T] or [B,T]
        Returns:
            加权后的标量损失
        """
        # ——— 1. 复谱 L1 ———
        # shape -> [B,T]
        if y_hat.dim()==3 and y_hat.size(1)==1:
            y_hat = y_hat.squeeze(1)
        if y.dim()==3 and y.size(1)==1:
            y = y.squeeze(1)
        B, T = y_hat.shape

        # window
        if self.window is None:
            win = torch.hann_window(self.win_length,
                                     device=y_hat.device,
                                     dtype=y_hat.dtype)
        else:
            win = self.window.to(y_hat.device, y_hat.dtype)

        # STFT (complex)
        S_hat = torch.stft(y_hat, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=win,
                           center=True,
                           return_complex=True)  # [B, F, N]
        S_ref = torch.stft(y,     n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=win,
                           center=True,
                           return_complex=True)

        re_hat, im_hat = S_hat.real, S_hat.imag
        re_ref, im_ref = S_ref.real, S_ref.imag

        spec_diff = torch.abs(re_hat - re_ref) + torch.abs(im_hat - im_ref)
        if self.reduction=='sum':
            L_spec = spec_diff.sum()
        else:
            L_spec = spec_diff.mean()

        # ——— 2. 时域 L1 ———
        # 保证长度一致
        if y_hat.size(-1)!=y.size(-1):
            Lmin = min(y_hat.size(-1), y.size(-1))
            y_hat_t = y_hat[..., :Lmin]
            y_t     = y[..., :Lmin]
        else:
            y_hat_t, y_t = y_hat, y
        L_time = F.l1_loss(y_hat_t, y_t, reduction=self.reduction)

        # ——— 3. RMS（响度）对齐损失 ———
        # 计算每条的 rms
        # Eg. RMS = sqrt(mean(x^2))
        rms_hat = torch.sqrt((y_hat_t**2).mean(dim=-1) + 1e-9)  # [B]
        rms_ref = torch.sqrt((y_t**2).mean(dim=-1)     + 1e-9)
        L_rms   = F.mse_loss(rms_hat, rms_ref, reduction=self.reduction)

        # ——— 4. 组合加权 ———
        loss = L_spec + self.alpha_time * L_time + self.alpha_rms * L_rms
        return loss
