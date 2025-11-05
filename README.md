# DIME: Deterministic Information Maximizing Autoencoders
_This is the official repository of our paper "DIME: Deterministic Information Maximizing Autoencoder". [Paper Link](https://iclr.cc/virtual/2025/35364)
._

> This repo demonstrates end-to-end ownership of our paper DIME, from objective design to clean, reproducible training, generation, and evaluation. The code is modular, readable, and built for fast ablations.

---

## Highlights
- **Deterministic, informative latents:** Autoencoder with an information-maximizing regularizer over normalized latents.
- **Stable training, sharp reconstructions:** No variational sampling or adversarial critics.
- **Turn‑key pipeline:** Train → Generate (reconstructions, interpolations, MVG/GMM sampling) → Evaluate (Improved Precision/Recall) → Linear eval.
- **Modular design:** Encoders/decoders and heads are decoupled for rapid iteration.

---

## Repository Layout
```
.
├── classification.py              # Linear eval / supervised head on top of encoders
├── generation_o.py                # Reconstructions, interpolations, MVG/GMM sampling, KDE/TSNE plots
├── improved_precision_recall.py   # Improved Precision & Recall evaluation utilities
├── model.py                       # Core wrappers (AE, heads)
├── models.py                      # Encoders/decoders for MNIST/CelebA and variants
├── models_test.py                 # Sanity tests for model building blocks
├── train.py                       # Training loop with MI-style uniformity regularizer
└── utils.py                       # Datasets, image IO, plotting helpers
```

---

## Installation

> Python 3.9+ recommended

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scriptsctivate

pip install -U pip

# Install PyTorch (choose the right CUDA/CPU build as needed)
# Example CUDA 12.1 build:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Common deps
pip install numpy matplotlib pillow tqdm scikit-learn plotly pandas torchmetrics
```

---

## Datasets

- **MNIST / FashionMNIST / CIFAR-10**: auto-download via `torchvision`.
- **CelebA**: expected at `data/celeba/{train,val,test}/...` (center-crop and resize handled in code).
- **Custom folders**: use `utils.ImageFolder("<path>/*")` for your own images.

**Paths & defaults**
- Training uses `--data-path` (default `./data/`).
- Generation uses `--data-path` and `--save-path` for outputs.
- Checkpoints saved under `--checkpoint` (e.g., `./checkpoint/mnist`).

---

## Replicating Results (Copy‑Paste)

### 1) Train an Autoencoder (DIME)
Latent size and hyperparameters can be dataset‑specific. Typical batch size is `100`.

**MNIST**
```bash
python train.py   --dataset mnist   --n 64   --epochs 100   --batch_size 100   --lr 0.005   --unif_lambda 1e-2   --t 0.005   --optimizer adam   --data-path ./data/   --checkpoint ./checkpoint/
```

**CelebA**
```bash
python train.py   --dataset celeba   --n 128   --epochs 100   --batch_size 100   --lr 0.005   --unif_lambda 1e-2   --t 0.005   --optimizer adam   --data-path ./data/   --checkpoint ./checkpoint/
```

_What this does:_ trains AE with reconstruction loss + information‑maximizing uniformity on normalized latent codes; writes checkpoints to `./checkpoint/<dataset>`.

---

### 2) Generate: Reconstructions, Interpolations, Sampling
Use trained checkpoints to analyze latents and visualize outputs. Supports **MVG** and **GMM** sampling, **interpolations**, **reconstructions**, and **KDE/TSNE** plots.

**Interpolations (MNIST, 10×10)**
```bash
python generation_o.py   --dataset mnist   --task interpolation   -X 10 -Y 10   --test-size 100   --model huae   --checkpoint ./checkpoint/mnist   --data-path ./data/   --save-path ./results/
```

**MVG sampling (CelebA)**
```bash
python generation_o.py   --dataset celeba   --task mvg   -X 10 -Y 10   --model huae   --checkpoint ./checkpoint/celeba   --data-path ./data/   --save-path ./results/
```

**Latent KDE visualization**
```bash
python generation_o.py   --dataset mnist   --task kde   --model huae   --checkpoint ./checkpoint/mnist   --data-path ./data/
```

Outputs are written under `--save-path` (e.g., `./results/Huae/<dataset>/...`).

---

### 3) Evaluate: Improved Precision & Recall (IPR)
Compare generated folders vs. real folders to assess generative quality. You can precompute manifold features for the real set and reuse them.

**Direct IPR comparison**
```bash
python improved_precision_recall.py   --path_real ./test_image_folder/celeba/real/   --path_fake ./results/Huae/celeba/   --batch_size 100   --k 3   --num_samples 10000
```

**Precompute “real” manifold**
```bash
python improved_precision_recall.py   --path_real ./test_image_folder/celeba/real/   --path_fake dummy   --fname_precalc celeba_real_manifold.npz
```

**Reuse the precomputed file later**
```bash
python improved_precision_recall.py   --path_real celeba_real_manifold.npz   --path_fake ./results/Huae/celeba/   --batch_size 100   --k 3   --num_samples 10000
```

The script writes out precision/recall logs and optional realism scores.

---

### 4) Linear Evaluation / Classification
Run a simple head on top of the learned encoder. Works with vanilla AE, VAE, or the DIME encoder.

**MNIST linear eval**
```bash
python classification.py   --dataset mnist   --batch-size 100   --train-size 60000   --epochs 100   -n 128   --optimizer adam   --checkpoint ./checkpoint/   --data-path ./data/   --gpu
```

Useful toggles:
- `--supervised` to jointly fine‑tune.
- `--vae` to use a VAE encoder.
- `--vanilla` to use a simple AE.

---

## Example: End‑to‑End on MNIST
```bash
# 1) Train
python train.py --dataset mnist --n 64 --epochs 50 --unif_lambda 1e-2 --t 0.005   --checkpoint ./checkpoint/ --data-path ./data/

# 2) Generate (interpolations + samples)
python generation_o.py --dataset mnist --task interpolation -X 8 -Y 8   --checkpoint ./checkpoint/mnist --save-path ./results/ --data-path ./data/

# 3) Evaluate (IPR)
python improved_precision_recall.py --path_real ./test_image_folder/mnist/real/   --path_fake ./results/Huae/mnist/ --batch_size 100 --num_samples 10000

# 4) Linear Eval
python classification.py --dataset mnist --epochs 100   --checkpoint ./checkpoint/ --data-path ./data/ --gpu
```

---

## Practical Tips

- **Batch size & workers:** Default batch size `100`. If you have fewer CPU cores, reduce `num_workers` in data loaders.
- **Determinism:** For strict reproducibility across hardware, fix seeds (`torch`, `numpy`) and enable deterministic CuDNN.
- **Checkpoints & results:** Use separate folders per dataset/config to keep runs organized (`./checkpoint/<run>`, `./results/<run>`).
- **Ablations:** Swap encoders/decoders in `models.py`, try different latent sizes with `--n`, and vary `--unif_lambda` / `--t`.

---

## Extending the Codebase

- **Architectures:** Add encoders/decoders in `models.py` and reference them in `model.py`.
- **Priors for generation:** Add new latent samplers in `generation_o.py` (follow MVG/GMM patterns).
- **Metrics:** Extend `improved_precision_recall.py` or plug in additional metrics (e.g., FID) using the generated folders.

---

## License
For research and evaluation purposes. Please adapt a license appropriate to your organization or release.
