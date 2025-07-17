# Multi-Teacher Knowledge Distillation for Accented English Speech Recognition

- Master's Thesis: 
- Presentation Slides: [./MTKD for AccentedASR.pdf](https://github.com/aalto-speech/MTKD4AccentedASR/blob/main/%5BMaster's%20Thesis%20Presentation%20Slides%5D%20MTKD%20for%20AccentedASR.pdf)
- Oral Presentation: https://www.youtube.com/watch?v=n5lFmgq_Fy0
- Publication I `@Interspeech 2025`: https://arxiv.org/abs/2506.08717
- Publication II:

<br><br>
## 🖼️ System Diagram of the Proposed MTKD Method
![Image](https://github.com/user-attachments/assets/ded9ca62-9e49-4f28-b08c-83bc46cebb5f)

<br><br>
## ✨ Features
- Multi-Teacher Distillation for robust ASR
- Support for various teacher weighting strategies:
  - Average (`mtkd_avg`)
  - Top-1 (`mtkd_top1`)
  - Rank-Weighted (`mtkd_rankweighted`)
- Accent specific fine-tuning baselines
- Logits extraction from teacher models
- Easy-to-configure training and evaluation pipeline

<br><br>
## ⚙️ Installation

1. **Clone the repository**
```bash
   git clone https://github.com/aalto-speech/MTKD4AccentedASR.git
   cd MTKD4AccentedASR
```

2. **Create and activate the virtual environment**
```bash
   conda env create -f environment.yml
   conda activate accented_asr
```

<br><br>
## 🚀 Usage
⚠️ Please ensure that all file paths in the code are correctly set according to your local machine before running any scripts. ⚠️

#### 🪄 Extract logits
```
python main.py \
  --TRAIN_ACCENT "American" \
  --DEVEL_ACCENT "Indian" \
  --TEST_ACCENT "Canadian" \
  --BATCH_SIZE 16 \
  --LEARNING_RATE 0.0001 \
  --N_EPOCHS 20 \
  --TRAINING 0
```

<br>
🧪 Train and Evaluate the Fine-Tuned Baseline

```
python ./ft/main.py \
  --TRAIN_ACCENT "American" \
  --DEVEL_ACCENT "Indian" \
  --TEST_ACCENT "Canadian" \
  --MODEL_CKP "facebook/wav2vec2-base-960h" \
  --BATCH_SIZE 8 \
  --LEARNING_RATE 0.0001 \
  --N_EPOCHS 20 \
  --TRAINING 1
```

```
python ./ft/main.py \
  --TRAIN_ACCENT "American" \
  --DEVEL_ACCENT "Indian" \
  --TEST_ACCENT "Canadian" \
  --MODEL_CKP "facebook/wav2vec2-base-960h" \
  --TRAINING 0
```


<br>
🧪 Train and Evaluate the proposed MTKD Method

```
python ./mtkd_avg/main.py \\
  --TRAIN_ACCENT "American" \\
  --DEVEL_ACCENT "Indian" \\
  --TEST_ACCENT "Canadian" \\
  --BATCH_SIZE 8 \\
  --LEARNING_RATE 0.0001 \\
  --N_EPOCHS 20 \\
  --TRAINING 1
```

```
python ./mtkd_avg/main.py \\
  --TRAIN_ACCENT "American" \\
  --DEVEL_ACCENT "Indian" \\
  --TEST_ACCENT "Canadian" \\
  --TRAINING 0
```

<br> <br>
## 🛠️ Customization
- --TRAIN_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC200H`, `EDACCdevel`, `AESRC200H+EDACCdevel`
- --DEVEL_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`
- --TEST_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC20H`, `AESRC10Accents`, `AESRC10Accents+AESRC20H`, `EDACCtest`
- --MODEL_CKP:
  - `facebook/wav2vec2-base-960h`
  - `facebook/wav2vec2-xls-r-300m`
  - `facebook/wav2vec2-base-100k-voxpopuli`
  - `elgeish/wav2vec2-base-timit-asr`
  - `patrickvonplaten/wav2vec2-base-timit-demo-colab`
  - `AKulk/wav2vec2-base-timit-epochs20`
- --N_EPOCHS: **Integers > 0**: `z ∈ ℤ, z > 0`
- --BATCH_SIZE: **Powers of Two**: `z = 2ᵏ where k ∈ ℤ, k ≥ 0`
- --LEARNING_RATE: **Positive Reals**: `x ∈ ℝ, x > 0.0`

<br> <br>
## 📚 Citation
**Publication I**
```bibtex
@article{bijoy2025multi,
  title={Multi-Teacher Language-Aware Knowledge Distillation for Multilingual Speech Emotion Recognition},
  author={Bijoy, Mehedi Hasan and Porjazovski, Dejan and Gr{\'o}sz, Tam{\'a}s and Kurimo, Mikko},
  journal={arXiv preprint arXiv:2506.08717},
  year={2025}
}
```

**Publication II**
```bibtex
...
```





