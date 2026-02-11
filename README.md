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
├── Con_Model.ipynb # Main notebook (conditional model training)
├── Models.ipynb # Tone encoder / generator experiments
├── datasets.py # Custom dataset definitions
├── generator.py # FiLM-GCN generator network
├── toneEncoder.py # Contrastive tone encoder
├── losses.py # Loss functions (MRSTFT, spectral, etc.)
├── helper.py # Utility functions (normalization, matching)
├── load_data.py # Data loading utilities
├── split_all_data.py # Dataset split scripts
├── reverb_test.py # Testing with reverb augmentation
├── count.py # Data statistics
└── README.md # Project documentation
```

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
