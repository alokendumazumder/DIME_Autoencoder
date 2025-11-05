# DIME: Disentangled & Uniform Manifold Autoencoders
_A compact research repo for training autoencoders, generating images, and evaluating precision/recall — designed to be easy to read, run, and extend._

## Why this repo stands out (for hiring managers)
- **Research-y yet pragmatic:** Clean PyTorch code that’s easy to follow, with reproducible training + evaluation scripts.
- **Multiple datasets & tasks:** MNIST, FashionMNIST, CIFAR-10, CelebA, Intel Scenes, plus synthetic shapes.
- **Clear metrics & visuals:** Precision/Recall (IPR) for generative quality, FID-ready outputs, latent space visualizations.
- **Modular design:** Encoders/decoders & heads are decoupled, enabling quick ablations and new backbones.

---

## Repository at a glance
- `train.py` — Train our AutoEncoder with an additional **uniformity** regularizer and rich logging (losses, gradient norms). fileciteturn1file6
- `generation_o.py` — Generate samples via **MVG**, **GMM**, **STDN** priors, run **interpolations**, **reconstructions**, FID-ready dumps, KDE/TSNE plots. fileciteturn1file1
- `classification.py` — Linear evaluation head on top of frozen (or optionally supervised) encoders for MNIST. fileciteturn1file0
- `improved_precision_recall.py` — Compute **Improved Precision & Recall (IPR)** using VGG16 features; also supports realism score & precalculation. fileciteturn1file2
- `model.py`, `models.py`, `models_test.py` — Encoders/decoders (MNIST, CelebA), MLP heads, plus a lightweight AE wrapper. fileciteturn1file3 fileciteturn1file4 fileciteturn1file5
- `utils.py` — Minimal datasets (e.g., glob-based `ImageFolder`) and helpers used by training and generation. fileciteturn1file7

> Tip: Start with `train.py` → `generation_o.py` → `improved_precision_recall.py` for a full end‑to‑end run.

---

## Installation
```bash
# Python 3.9+ recommended
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip

# Core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA/CPU build
pip install numpy matplotlib pillow tqdm scikit-learn plotly pandas

# Optional: pythae (used by some benchmarks in generation)
pip install pythae

# If you plan to compute IPR with VGG16 on GPU
pip install torchmetrics
```

---

## Datasets
- **MNIST / FashionMNIST / CIFAR-10:** auto-downloaded by torchvision.
- **CelebA:** expects folder `data/celeba/{train,val,test}/...` (center‑crop then resize used in code). fileciteturn1file6
- **Intel Scenes:** expects `data/intel/{train,test}/...` folders. fileciteturn1file6
- **Custom folders:** globbed by `utils.ImageFolder("<path>/*")`. fileciteturn1file7

Set the base with `--data-path` (default `./data/` in `train.py`, `../data/` in `generation_o.py`). fileciteturn1file6 fileciteturn1file1

---

## Quick Start (Replicate our results)

### 1) Train an AutoEncoder
Trains AE with MSE + **uniformity** regularization (controlled by `--unif_lambda` and `--t`). Logs include batch‑wise gradient norms for analysis. fileciteturn1file6

**MNIST example:**
```bash
python train.py   --dataset mnist   --n 64   --epochs 100   --batch_size 100   --lr 0.005   --unif_lambda 1e-2   --t 0.005   --optimizer adam   --data-path ./data/   --checkpoint ./checkpoint/
```

**CelebA example:**
```bash
python train.py   --dataset celeba   --n 128   --epochs 100   --batch_size 100   --lr 0.005   --unif_lambda 1e-2   --t 0.005   --data-path ./data/   --checkpoint ./checkpoint/
```

_What this does internally:_ computes AE reconstruction loss and a uniformity term on latent codes; saves checkpoints into `./checkpoint/`. Batch‑wise gradient norms and losses are tracked per epoch. fileciteturn1file6

---

### 2) Generate & Visualize
Use the trained encoder/decoder to **reconstruct**, **interpolate**, or **sample** from latent distributions (MVG/GMM/STDN). Also supports **FID‑ready** image dumps and **KDE plots** of the latent space. fileciteturn1file1

**Interpolation on MNIST (10×10 grid):**
```bash
python generation_o.py   --dataset mnist   --task interpolation   -X 10 -Y 10   --test-size 100   --model huae   --checkpoint ./checkpoint/mnist   --data-path ./data/   --save-path ./results/
```
This saves images under `./results/Huae/mnist/…` with sensible filenames. fileciteturn1file1

**MVG sampling (CelebA):**
```bash
python generation_o.py   --dataset celeba   --task mvg   -X 10 -Y 10   --model huae   --checkpoint ./checkpoint/celeba   --data-path ./data/   --save-path ./results/
```

**KDE of latent codes (2D viz):**
```bash
python generation_o.py   --dataset mnist   --task kde   --model huae   --checkpoint ./checkpoint/mnist   --data-path ./data/
```
Produces `plots/image_huae_kde_2.png`. fileciteturn1file1

> The script has switches for VAEs, IRMAE, LoRAE, and GMM baselines if you wire them in; default is our HUAE (“huae”). See the argument block in `generation_o.py`. fileciteturn1file1

---

### 3) Evaluate Generative Quality (Improved Precision/Recall)
We compute **IPR** using VGG16 features. You can precalculate the real manifold once and reuse it. fileciteturn1file2

**Compute precision/recall given two folders:**
```bash
python improved_precision_recall.py   --path_real ./test_image_folder/celeba/real/   --path_fake ./results/Huae/celeba/   --batch_size 100   --k 3   --num_samples 10000
```
Writes `precision_recall.txt` with per‑run entries and prints realism scores for sample images. fileciteturn1file2

**Precompute real manifold:**
```bash
python improved_precision_recall.py   --path_real ./test_image_folder/celeba/real/   --path_fake dummy   --fname_precalc celeba_real_manifold.npz
```
Then later: `--path_real celeba_real_manifold.npz` speeds up subsequent runs. fileciteturn1file2

---

### 4) Linear Eval / Downstream Classification (MNIST)
Attach a small MLP `Head` to frozen encoders (or jointly fine‑tune with `--supervised`). Supports **vanilla** AE, **VAE**, or our HUAE checkpoints. fileciteturn1file0

```bash
python classification.py   --dataset mnist   --batch-size 100   --train-size 60000   --epochs 100   -n 128   --optimizer adam   --checkpoint ./checkpoint/   --data-path ./data/   --gpu
```
Key toggles: `--supervised` (joint training), `--vae` (use VAE enc), `--vanilla` (simple AE). Validation runs every 10 epochs. fileciteturn1file0

---

## Models & Architecture
- **Encoders/Decoders:** MNIST and CelebA variants implemented with standard Conv/ConvTranspose stacks; code is modular and normalized via an `L2Norm` utility where appropriate. fileciteturn1file3 fileciteturn1file4
- **Head / MLP:** Lightweight MLP for classification or feature transforms. fileciteturn1file3
- **AE wrapper:** Encode/Decode helpers & reconstruction loss path for simpler training loops. fileciteturn1file4 fileciteturn1file5

---

## Reproducibility checklist
- **Hardware:** Code is CUDA‑aware and uses DataLoaders with `num_workers=32` by default (reduce on low‑core machines). fileciteturn1file0 fileciteturn1file1
- **Checkpoints:** All scripts read/write from `--checkpoint` (e.g., `./checkpoint/mnist`). fileciteturn1file0 fileciteturn1file1 fileciteturn1file6
- **Logging/Plots:** Latent 3D TSNE HTML and gradient‑variance PNGs are produced by `train.py` utilities and `generation_o.py` (KDE). fileciteturn1file6 fileciteturn1file1
- **Determinism:** For strict replication across hardware, set seeds (`torch`, `numpy`) and enable deterministic backends if needed (not enforced by default).

---

## Example: Full pipeline on MNIST
```bash
# 1) Train
python train.py --dataset mnist --n 64 --epochs 50 --unif_lambda 1e-2 --t 0.005 --checkpoint ./checkpoint/ --data-path ./data/

# 2) Generate (interpolation + sampling)
python generation_o.py --dataset mnist --task interpolation -X 8 -Y 8 --checkpoint ./checkpoint/mnist --save-path ./results/ --data-path ./data/

# 3) Evaluate IPR
python improved_precision_recall.py --path_real ./test_image_folder/mnist/real/ --path_fake ./results/Huae/mnist/ --batch_size 100 --num_samples 10000

# 4) Linear Eval
python classification.py --dataset mnist --epochs 100 --checkpoint ./checkpoint/ --data-path ./data/ --gpu
```

---

## Extending the code
- Plug in new encoders/decoders in `models.py` / `model.py`. fileciteturn1file4 fileciteturn1file3
- Add new generation priors in `generation_o.py` (follow MVG/GMM/STDN patterns). fileciteturn1file1
- Swap classification heads in `classification.py`. fileciteturn1file0

---

## Citation
If you use this codebase in academic or industrial work, please reference the repo and acknowledge the IPR metric implementation. fileciteturn1file2

---

**Maintainer note:** This README was generated to showcase clarity for reviewers and hiring teams while remaining faithful to the repo’s scripts and flags.
