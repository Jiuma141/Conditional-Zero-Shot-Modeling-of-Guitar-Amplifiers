import torchaudio
import numpy as np

def load_mono(path, sr=44100):
    x, _ = torchaudio.load(path)
    x = x.mean(0).numpy()
    # 可加重采样
    return x

def energy_decay_curve(sig):
    # sig: 一维 numpy array, 假定已对齐到冲击起始帧
    e = np.cumsum(sig[::-1]**2)[::-1]
    return 10*np.log10(e / np.max(e))

def estimate_rt60(decay_curve, t):
    # decay_curve: dB 标度的能量衰减对数曲线
    # t: 每帧对应的时间戳数组
    # 在 –5 dB 到 –35 dB 范围拟合直线，斜率求 RT60
    import numpy as np
    idx = np.where((decay_curve <= -5) & (decay_curve >= -35))[0]
    if len(idx) < 2: return None
    p = np.polyfit(t[idx], decay_curve[idx], 1)
    # 斜率 p[0] 单位 dB/s, RT60 = –60 / slope
    return -60.0 / p[0]


def compute_rt60(sig, sr=44100):
    # 假定冲击起点在 0，或可先用能量最大帧对齐
    dec = energy_decay_curve(sig)
    t = np.arange(len(dec)) / sr
    return estimate_rt60(dec, t)

def compute_elr(sig, sr=48000, early_ms=50, n_fft=1024, hop=256):
    import numpy as np
    S = np.abs(np.stack([np.fft.rfft(sig[i:i+n_fft])
                          for i in range(0, len(sig)-n_fft, hop)], axis=1))
    n_early = int((early_ms/1000) * sr / hop)
    early_e = (S[:,:n_early]**2).sum()
    late_e  = (S[:,n_early:]**2).sum()
    return 10*np.log10(early_e / (late_e+1e-8))

# 加载
orig = load_mono('Jiaming Generated Fractal - FM3 - USA Lead (1).wav')
imp  = load_mono('Fractal - FM3 - USA Lead.wav')

# 计算并打印
rt60_orig = compute_rt60(orig)
rt60_imp  = compute_rt60(imp)
elr_orig  = compute_elr(orig)
elr_imp   = compute_elr(imp)

print(f"RT60: orig={rt60_orig:.3f}s, improved={rt60_imp:.3f}s")
print(f"ELR (dB): orig={elr_orig:.1f}, improved={elr_imp:.1f}")
