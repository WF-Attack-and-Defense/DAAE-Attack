# Defense-Aware Autoencoder (DAAE)

Defense-Aware Autoencoder (DAAE) is a research framework that evaluates the robustness of website fingerprinting (WF) defenses. Many defenses inject **dummy packets** (padding, cover traffic) to obfuscate deep-learning WF classifiers. DAAE learns to **reverse** those perturbations using a small paired subset of original and defended traces, then recovers the remaining defended traces to improve WF attack accuracy.

:warning: :warning: :warning: This software is a research prototype intended solely for experimental and evaluation purposes. :warning: :warning: :warning:

## Auto–Decoder for Traffic Recovery

Auto–decoder models compress input into a latent space and reconstruct outputs:
  1. Encodes defended traces into latent representations.
  2. Predicts a **dummy-packet mask** (which positions are injected).
  3. Reconstructs original packet values at non-dummy positions.

With only **20–30%** paired traces per class for training, DAAE generalizes to recover all other instances—demonstrating that defense perturbations can be learned and reversed



## Project Structure

```
DAAE/
├── README.md
├── requirements.txt
├── run.py                  # End-to-end pipeline
└── daae/
    ├── config.py           # Hyperparameters only (no dataset paths)
    ├── cli.py              # Shared --original-root/--defended-root/--output-dir args
    ├── train.py            # Train DAAE / FS-DAAE
    ├── recover.py          # Remove dummy packets from defended traces
    ├── evaluate.py         # Recovery + WF attack evaluation
    ├── data/
    │   ├── cell_io.py      # .cell file parser
    │   ├── features.py     # Defense-agnostic multi-channel features
    │   ├── alignment.py    # Direction-based dummy mask labels
    │   ├── few_shot.py     # Episodic K-shot support sampling
    │   ├── support_bank.py # Per-class support memory for inference
    │   └── dataset.py      # Per-class train/val/test split
    └── models/
        ├── daae.py         # Baseline Defense-Aware Autoencoder
        ├── fs_daae.py      # Few-Shot DAAE with support-query attention
        ├── losses.py       # Focal BCE + reconstruction loss
        ├── factory.py      # Model creation / checkpoint loading
        └── classifier.py   # 1D-CNN WF attack model
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

All commands require these arguments:

| Argument | Description |
|----------|-------------|
| `--original-root` | Original (undefended) `.cell` directory |
| `--defended-root` | Defended `.cell` directory (same filenames) |
| `--output-dir` | Output directory for checkpoints and results |
| `--num-classes` | Number of classes to load (`0` .. `num_classes-1`) |
| `--num-instances` | Number of instances per class (`0` .. `num_instances-1`) |
| `--train-ratio` | Fraction of per-class instances used for DAAE training |

Dataset discovery constructs paths directly as `{class}-{instance}.cell` (fast) instead of scanning the full directory.

## GPU / high-memory server (RTX PRO 6000, 192GB RAM)

Defaults are tuned for this hardware:

- **Batch size 128** (increase with `--batch-size 256` if VRAM allows)
- **Mixed precision (AMP)** on CUDA
- **GPU-resident tensors** — train/val data loaded directly to GPU (zero H2D copy per epoch)
- **Preload train + val** into RAM/GPU with disk cache under `{output_dir}/.daae_cache/`
- **32 parallel preload workers** for first-time cache build
- **`torch.compile`** enabled by default (disable with `--no-compile` if needed)

### Model training

```bash
python -u -m daae.train \
  --original-root ../DF/CW \
  --defended-root ../defense/palette/CW/DF \
  --output-dir ./results/palette-daae \
  --num-classes 95 --num-instances 1000 \
  --trace-format paired_dir \
  --train-ratio 0.3 \
  --max-len 10000 \
  --epochs 50 --batch-size 3
```

### Defended dataset recovery

```
python -u -m daae.recover \
  --checkpoint ./results/palette-daae/daae_best.pt \
  --defended-root /data/dataset/newDataset/defenses/palette/CW/DF \
  --output-dir ./results/palette-daae \
  --num-classes 100 --num-instances 2000
```
