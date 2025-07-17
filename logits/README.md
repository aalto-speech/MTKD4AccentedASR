# Logits Extraction
This module extracts logits from a pretrained or fine-tuned Wav2Vec2 CTC model on accented speech datasets. These logits are useful for knowledge distillation, teacher-student training, or alignment evaluation across multiple accents. It supports various training/test accent pairs and outputs HuggingFace datasets with saved logits.


## 🧠 What It Does
- Loads a pretrained or fine-tuned Wav2Vec2 model.
- Loads the speech datasets for training, development, and testing.
- Processes and tokenizes the audio using Wav2Vec2Processor.
- Trains the model using CTC loss (optional).
- Evaluates on a dev set (optional).
- Extracts logits from the test set and saves them to disk.


## 🚀 How to Run (Logits Extraction Only)
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
Or, submit to SLURM with the job script: `sbatch job.slrm`


## 💾 Saved Output
Logits will be saved as a HuggingFace dataset at: `.../ft_on_{TRAIN_ACCENT}_Accent/tested_on_{TEST_ACCENT}`
Each example includes:
- Original input features
- Transcription
- Logits (model output before softmax)


## 🛠️ Customization
- --TRAIN_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC200H`, `EDACCdevel`, `AESRC200H+EDACCdevel`
- --DEVEL_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`
- --TEST_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC20H`, `AESRC10Accents`, `AESRC10Accents+AESRC20H`, `EDACCtest`
- --N_EPOCHS: **Integers > 0**: `z ∈ ℤ, z > 0`
- --BATCH_SIZE: **Powers of Two**: `z = 2ᵏ where k ∈ ℤ, k ≥ 0`
- --LEARNING_RATE: **Positive Reals**: `x ∈ ℝ, x > 0.0`











