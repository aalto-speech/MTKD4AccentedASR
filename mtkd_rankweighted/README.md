# MTKD_RankWeighted
 A variant of Multi-Teacher Knowledge Distillation (MTKD) that utilizes all teacher models, but weights their contributions based on a rank-based strategy. Each teacher’s influence is determined by its relative alignment with the student, as computed by the **neural teacher mixer**. Higher-ranked teachers contribute more to the final supervision signal. This method also incorporates the **meta-learned objective adapter** to dynamically balance the CTC loss and KL divergence.

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


## 🛠️ Customization
- --TRAIN_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC200H`, `EDACCdevel`, `AESRC200H+EDACCdevel`
- --DEVEL_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`
- --TEST_ACCENT: `American`, `Canadian`, `Indian`, `Korean`, `Russian`, `British`, `Chinese`, `Japanese`, `Portuguese`, `Spanish`, `AESRC20H`, `AESRC10Accents`, `AESRC10Accents+AESRC20H`, `EDACCtest`
- --N_EPOCHS: **Integers > 0**: `z ∈ ℤ, z > 0`
- --BATCH_SIZE: **Powers of Two**: `z = 2ᵏ where k ∈ ℤ, k ≥ 0`
- --LEARNING_RATE: **Positive Reals**: `x ∈ ℝ, x > 0.0`


