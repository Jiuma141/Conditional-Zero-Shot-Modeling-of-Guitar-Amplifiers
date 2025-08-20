# Conditional Zero-Shot Modeling of Guitar Amplifiers 🎸

This project explores **conditional zero-shot modeling of guitar amplifiers** using deep learning.
It aims to **replicate the sonic characteristics** of real guitar amplifiers while allowing **control via tone and knob embeddings**, enabling flexible tone cloning and parameter-conditioned audio generation.

---

## 🚀 Features

* **Zero-Shot Tone Cloning**
  Capture the timbre of unseen guitar amplifiers without retraining.

* **Conditional Control**
  Supports **knob-parameter conditioning** (e.g., gain, EQ) for realistic amplifier behavior.

* **Learned Tone Embedding**
  A contrastive-trained **tone encoder** maps reference audio into a compact embedding space for conditioning.

* **High-Fidelity Audio Modeling**
  Generator network trained with **multi-resolution STFT loss** and **spectral convergence loss** for perceptual quality.

---

## 🛠️ Methodology

* **Tone Encoder (E):**
  A SimCLR-inspired encoder projects audio into a 128-D tone embedding.
  Trained on AMP-SPACE dataset for supervised tone similarity.

* **Conditional Generator (G):**
  Implemented with **FiLM-GCN blocks**, modulated by both **tone embedding** and **knob parameters**.

* **Training Losses:**

  * Multi-Resolution STFT Loss
  * Spectral Convergence Loss
  * Complex Spectral Loss

---

## 📊 Dataset

* **AMP-SPACE Dataset**
  Contains dry/wet guitar signal pairs across diverse amplifiers and settings.
  Preprocessing includes:

  * Loudness matching
  * Peak normalization
  * Resampling
  * Knob parameter normalization

---

## 📂 Project Structure

```
.
├── datasets/          # Custom Dataset classes (AmpPairDataset, CSVSupToneDataset)
├── generator/         # FiLM-GCN based generator
├── toneEncoder/       # Tone encoder implementation
├── losses/            # Custom loss functions (STFT, spectral)
├── scripts/           # Training and evaluation scripts
└── examples/          # Audio demos and case studies
```

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/conditional-amp-modeling.git
cd conditional-amp-modeling
pip install -r requirements.txt
```

---

## 📖 Usage

### Train Tone Encoder

```bash
python train_tone_encoder.py --data path/to/csv --epochs 100
```

### Train Conditional Generator

```bash
python train_generator.py --train_csv data/train.csv --val_csv data/val.csv \
    --tone_ckpt checkpoints/tone_encoder.pt --epochs 50
```

### Inference

```bash
python inference.py --input dry_guitar.wav --amp example_amp --tone_ref tone.wav
```

---

## 🎧 Audio Examples

| Task         | Example        |
| ------------ | -------------- |
| Tone Cloning | [Demo Link](#) |
| Gain Control | [Demo Link](#) |

---

## 📑 Case Studies

* **Tone Clone:**
  Successfully reproduces amplifier timbre in low/mid frequency ranges, with slight reduction in high-frequency detail.

* **Gain Control:**
  Accurately models knob-conditioned dynamics, preserving harmonic structure across gain settings.

---

## 📝 Citation

If you use this project in research, please cite:

```
@misc{wang2025conditionalamp,
  title={Conditional Zero-Shot Modeling of Guitar Amplifiers},
  author={Jiaming Wang},
  year={2025},
  note={Master Thesis, Carnegie Mellon University}
}
```

---

## 📬 Contact

Author: **Jiaming Wang**

* 🎓 Master of Music & Technology, Carnegie Mellon University
* ✉️ Email: [jiamingw@cmu.edu](mailto:jiamingw@cmu.edu)
* 🌐 GitHub: [xhn2333](https://github.com/xhn2333)

---

要不要我帮你再写一个**精简版（350字以内的 GitHub description）**，像你之前要的那种简短介绍，放在 repo 的简介和开头？
