## Fine-Tuning
A baseline method for accented speech recognition that fine-tunes a pretrained Wav2Vec2 model (e.g., facebook/wav2vec2-base-960h) directly on labeled speech data from a specific accent. No teacher models or distillation strategies are used. The method relies solely on supervised learning using CTC loss, and serves as a reference point for evaluating more complex knowledge distillation techniques.


## 🚀 How to Train?
```
python main.py \
  --TRAIN_ACCENT "American" \
  --DEVEL_ACCENT "Indian" \
  --TEST_ACCENT "Canadian" \
  --MODEL_CKP "facebook/wav2vec2-base-960h" \
  --BATCH_SIZE 8 \
  --LEARNING_RATE 0.0001 \
  --N_EPOCHS 20 \
  --TRAINING 1
```
Or, submit to SLURM with the job script: `sbatch job.slrm`


## 🧪 How to Evaluate?
```
python main.py \
  --TRAIN_ACCENT "American" \
  --DEVEL_ACCENT "Indian" \
  --TEST_ACCENT "Canadian" \
  --MODEL_CKP "facebook/wav2vec2-base-960h" \
  --TRAINING 0
```


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






