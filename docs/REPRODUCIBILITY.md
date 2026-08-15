# Reproducibility Guide

## 1. Release status

The current public repository contains one TensorFlow/Keras notebook, `Code/hyscan.ipynb`. It is useful as a research prototype and exposes the principal HySCAN ideas, but it does not yet provide a modular, command-line implementation of every experiment in the ICML paper.

This distinction matters because the paper's reported results use dataset-specific ResNet-110 and ResNet-50 pipelines, five seeds, adaptive attacks, and joint-randomness Monte Carlo certification. The current notebook demonstrates a smaller ResNet-18-style workflow with local NumPy arrays.

## 2. What the notebook currently contains

- TensorFlow/Keras custom layers for global, feature, spatial, and channel attention noise.
- A normalized SoftPlus fusion layer corresponding to the SANI concept.
- An attention-gated stochastic convolution layer corresponding to the RWAN concept.
- ResNet-18-style residual blocks in which stochastic convolutions replace standard convolutions.
- Gaussian-noise training through a `tf.data` pipeline.
- Model checkpointing, early stopping, and learning-rate reduction.
- Illustrative smoothing/certification utilities.
- A CleverHans PGD evaluation example.

Notebook metadata records Python 3.11.13 and a Kaggle GPU runtime.

## 3. Current notebook defaults

| Item | Notebook value |
|---|---|
| Input files | `X.npy`, `Y.npy` |
| Split | 20% test, then 20% of remaining data for validation (`random_state=42`) |
| Demonstration backbone | ResNet-18-style `[2,2,2,2]` |
| Default input shape | `128×128×3` |
| Default classes | 5 |
| Gaussian training noise | `σ=0.5` |
| Weight-noise standard deviation | `0.05` |
| Spatial/channel noise factors | `0.6 / 0.6` |
| Batch size | 8 |
| Epochs | 100 |
| Active optimizer | Adam, LR `0.001` |
| Notebook smoothing samples | commonly `N₀=1,000`, `N=100,000`, `α=1e-3` |
| PGD demo | CleverHans, 10 iterations, random start |

Some notebook cells are exploratory or duplicated. Run the notebook selectively and inspect cell dependencies rather than assuming it is a clean top-to-bottom script.

## 4. Paper configuration

### Backbones

- ResNet-110: CIFAR-10 and CIFAR-100.
- ResNet-50: ImageNet-1k, CelebA, NCT-CRC-HE-100K, NIH-CXR, EyePACS, and HAM10000.

### Optimization

| Family | Epochs | Batch | Optimizer | Schedule |
|---|---:|---:|---|---|
| CIFAR-10/100 | 200 | 256 | AdamW, LR `1e-2`, WD `1e-4` | Step 30, multiplier `0.1` |
| CelebA + medical | 200 | 64 | SGD, LR `5e-2` | Step 3, multiplier `0.8` |
| ImageNet | 200 | 300 | SGD, LR `1e-1`, momentum `0.9`, WD `1e-4` | Warm-up; step 30, multiplier `0.1` |

All paper results are averaged over five seeds and were run on an NVIDIA A100 80GB.

### Certification

- Fresh Gaussian, RWAN, and SANI randomness on every query.
- Default `n₀=100`, `n=100,000`, `α=0.001`.
- `σ ∈ {0.25, 0.50, 1.0, 2.0}`.
- Dataset-specific fixed certification subsets.

### Empirical attacks

- APGD-20: APGD-CE and APGD-T-DLR, 20 iterations, five restarts.
- AutoAttack.
- PGD sweeps over perturbation size and 10–100 steps.
- EOT-PGD, BPDA, and EOT-BPDA.
- Main budgets: `8/255` and `16/255`.

## 5. Important differences

| Component | Paper | Current notebook |
|---|---|---|
| Primary backbones | ResNet-110 / ResNet-50 | ResNet-18-style prototype |
| Datasets | Eight named benchmarks | Generic `X.npy` / `Y.npy` arrays |
| Training length | 200 epochs | 100 epochs |
| Batch sizes | 256 / 64 / 300 | 8 |
| Optimizers | AdamW or SGD | Adam |
| Certified pilot size | `n₀=100` | illustrative code commonly uses `N₀=1,000` |
| Attack suite | APGD, AA, PGD, EOT/BPDA | PGD example only |
| Experiment orchestration | Dataset-specific, five seeds | Notebook cells |
| Released checkpoints | Required for exact tables | Not included |

Do not describe a notebook run with its default settings as an exact reproduction of a paper table.

## 6. Certification caution

The notebook contains an illustrative `HybridCertifier` that combines smoothing with a deterministic margin divided by a user-provided Lipschitz constant. The included ResNet-style architecture is not accompanied by a proof that its global Lipschitz constant is `1.0`. Therefore:

- treat the deterministic-radius branch as demonstration code only;
- do not report it as a certificate unless a valid end-to-end Lipschitz bound is established;
- use the paper's joint-randomness randomized-smoothing procedure for paper-faithful certification.

The class docstring in one notebook utility also refers to a deterministic evaluation model. HySCAN's paper-level predictor is stochastic at evaluation and certification time, so verify that every implemented stochastic source remains active and is resampled.

## 7. Recommended reproduction layout

A future full release should separate reusable components from experiment drivers:

```text
hyscan/
├── models/
│   ├── rwan.py
│   ├── sani.py
│   ├── resnet110.py
│   └── resnet50.py
├── attacks/
│   ├── apgd.py
│   ├── autoattack.py
│   └── adaptive_eot.py
├── certification/
│   ├── smooth.py
│   └── certify.py
├── data/
│   └── <dataset loaders>
├── configs/
│   ├── cifar10.yaml
│   ├── imagenet.yaml
│   └── medical/*.yaml
└── scripts/
    ├── train_certified.py
    ├── train_empirical.py
    ├── evaluate_attacks.py
    └── certify.py
```

## 8. Checklist for a table-level reproduction

- [ ] Record the exact source/version of every dataset.
- [ ] Publish immutable train/validation/test and certification-subset manifests.
- [ ] Match the paper backbone and input resolution.
- [ ] Match every RWAN/SANI noise and recursion parameter.
- [ ] Match Gaussian training noise to certification `σ`.
- [ ] Use all five seeds and publish their values.
- [ ] Keep internal randomness active during attacks and certification.
- [ ] Match APGD/AA steps, restarts, loss variants, and budgets.
- [ ] Include EOT/BPDA evaluations for the stochastic predictor.
- [ ] Save checkpoints, optimizer state, logs, and software versions.
- [ ] Report clean, certified, empirical, cost, and abstention statistics together.
- [ ] Hash model checkpoints and result manifests.

## 9. Suggested command interface for a future release

The current repository does not yet provide these commands; they illustrate a publication-quality target interface:

```bash
python scripts/train_certified.py --config configs/cifar10.yaml --sigma 0.25 --seed 0
python scripts/certify.py --checkpoint runs/cifar10/seed0/best.pt --n0 100 --n 100000 --alpha 0.001
python scripts/train_empirical.py --config configs/nih_cxr.yaml --epsilon 8/255 --seed 0
python scripts/evaluate_attacks.py --checkpoint runs/nih_cxr/seed0/best.pt --suite apgd,aa,eot-pgd,bpda,eot-bpda
```

## 10. Environment note

`requirements.txt` and `environment.yml` are compatibility-oriented scaffolding derived from notebook imports. The original repository does not include an author-pinned lockfile, CUDA version, or exact package export. Record the actual resolved environment before reporting results.
