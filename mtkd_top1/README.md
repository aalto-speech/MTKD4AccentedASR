# MTKD_top1
A variant of **Multi-Teacher Knowledge Distillation (MTKD)** where **only the best-aligned (Top-1) teacher model** is used to guide the student model. The approach uses **meta-learned objecctive adapter** to dynamically balance **CTC loss** and **KL divergence** during training.


## 🚀 How to Train?
```
python main.py \\
  --TRAIN_ACCENT "American" \\
  --DEVEL_ACCENT "Indian" \\
  --TEST_ACCENT "Canadian" \\
  --BATCH_SIZE 8 \\
  --LEARNING_RATE 0.0001 \\
  --N_EPOCHS 20 \\
  --TRAINING 1
```
Or, submit to SLURM with the job script: `sbatch job.slrm`


## 🧪 How to Evaluate?
```
python main.py \\
  --TRAIN_ACCENT "American" \\
  --DEVEL_ACCENT "Indian" \\
  --TEST_ACCENT "Canadian" \\
  --TRAINING 0
```
The script will:
- Load the saved checkpoint
- Evaluate on the specified test set
- Print WER, CER
- Save predictions to CSV


## 🧠 Model Overview
- Backbone: `facebook/wav2vec2-base-960h`
- Loss Function: Combination of CTC + KL-divergence
- Optimizer: AdamW
- Scheduler: Linear warmup and decay


## 🛠️ Customization
Available accents:
- --TRAIN_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC200H`, `EDACCdevel`, `AESRC200H+EDACCdevel`
- --DEVEL_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`
- --TEST_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC20H`, `AESRC10Accents`, `AESRC10Accents+AESRC20H`, `EDACCtest`
- --N_EPOCHS: **Integers > 0**: `z ∈ ℤ, z > 0`
- --BATCH_SIZE: **Powers of Two**: `z = 2ᵏ where k ∈ ℤ, k ≥ 0`
- --LEARNING_RATE: **Positive Reals**: `x ∈ ℝ, x > 0.0`













