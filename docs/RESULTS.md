# Paper-Reported Results

All values in this document are percentages transcribed from the HySCAN paper. Use the paper as the authoritative source and preserve its attack, smoothing, subset, and seed protocols when comparing methods.

## 1. Certified accuracy on ImageNet and CIFAR-10

Radii: `r ∈ {0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00}`.

### ImageNet-1k

| σ | r=0 | r=.25 | r=.50 | r=.75 | r=1.0 | r=1.25 | r=1.5 | r=2.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 71.9 | 65.2 | 56.7 | 47.9 | 42.3 | 37.1 | 30.7 | 7.15 |
| 0.50 | 68.9 | 61.5 | 55.1 | 46.8 | 42.3 | 37.4 | 33.9 | 26.2 |

### CIFAR-10

| σ | r=0 | r=.25 | r=.50 | r=.75 | r=1.0 | r=1.25 | r=1.5 | r=2.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 85.2 | 70.4 | 57.3 | 45.5 | 37.9 | 30.7 | 24.1 | 9.94 |
| 0.50 | 80.3 | 66.4 | 56.1 | 45.8 | 38.3 | 31.7 | 25.1 | 13.9 |

## 2. Certified accuracy on NIH-CXR, HAM10000, and CelebA

Columns report certified accuracy at `r ∈ {0, 0.5, 1.0}`.

| Dataset | σ | r=0 | r=.5 | r=1.0 |
|---|---:|---:|---:|---:|
| NIH-CXR | 0.25 | 81.2 | 63.1 | 39.4 |
| NIH-CXR | 0.50 | 76.1 | 59.6 | 42.3 |
| NIH-CXR | 1.00 | 71.4 | 61.1 | 42.9 |
| HAM10000 | 0.25 | 96.9 | 62.4 | 37.2 |
| HAM10000 | 0.50 | 92.8 | 61.2 | 37.9 |
| HAM10000 | 1.00 | 87.7 | 63.3 | 39.5 |
| CelebA | 0.25 | 96.5 | 59.7 | 34.9 |
| CelebA | 0.50 | 92.5 | 60.7 | 36.1 |
| CelebA | 1.00 | 87.4 | 63.9 | 38.4 |

## 3. Certified accuracy on EyePACS and NCT-CRC-HE-100K

| Dataset | σ | r=0 | r=.25 | r=.50 | r=.75 | r=1.0 |
|---|---:|---:|---:|---:|---:|---:|
| EyePACS | 0.25 | 87.1±0.82 | 67.3±0.59 | 52.2±1.93 | 46.4±0.88 | 40.5±1.87 |
| EyePACS | 0.50 | 83.3±0.58 | 64.7±0.51 | 53.5±1.92 | 46.8±0.97 | 41.7±1.31 |
| NCT-CRC-HE-100K | 0.25 | 95.9±1.62 | 73.4±0.56 | 64.2±1.52 | 52.5±1.45 | 32.6±0.69 |
| NCT-CRC-HE-100K | 0.50 | 91.9±1.03 | 71.4±0.95 | 64.2±0.83 | 52.7±0.91 | 37.2±1.69 |

## 4. Empirical robustness on medical benchmarks

APGD-20 and AA-20 at `ε ∈ {8/255, 16/255}`.

| Dataset | Clean | APGD 8/255 | APGD 16/255 | AA 8/255 | AA 16/255 |
|---|---:|---:|---:|---:|---:|
| NCT-CRC-HE-100K | 91.5±2.25 | 90.7±1.69 | 80.2±2.61 | 89.5±1.24 | 77.3±2.98 |
| NIH-CXR | 90.1±0.87 | 88.9±1.75 | 78.1±2.82 | 87.3±1.98 | 75.1±2.59 |
| EyePACS | 78.3±1.92 | 73.9±1.57 | 62.2±2.91 | 73.1±2.25 | 60.5±2.88 |
| HAM10000 | 75.1±1.84 | 68.3±2.75 | 57.2±2.86 | 66.9±3.17 | 55.2±2.62 |

## 5. Extended empirical robustness on natural images

PGD-20 and AA-20 at `ε=8/255`.

| Dataset | Clean | PGD-20 | AA-20 |
|---|---:|---:|---:|
| CIFAR-10 | 90.93 | 74.63 | 78.47 |
| ImageNet-1k | 74.24 | 60.69 | 68.32 |

## 6. RWAN and SANI ablation

CIFAR-10, `σ=0.25`, certified at `r=0.75`; PGD-20 at `ε=8/255`.

| RWAN | SANI | Certified | PGD-20 |
|:---:|:---:|---:|---:|
| ✗ | ✗ | 26.1 | 59.8 |
| ✗ | ✓ | 37.5 | 67.9 |
| ✓ | ✗ | 40.1 | 69.7 |
| ✓ | ✓ | **45.5** | **74.3** |

## 7. RecRAN and NoRAN ablation

The starting point without either SANI branch corresponds to the RWAN-only configuration.

| RecRAN | NoRAN | Certified | PGD-20 |
|:---:|:---:|---:|---:|
| ✗ | ✗ | 40.1 | 69.7 |
| ✗ | ✓ | 42.4 | 72.5 |
| ✓ | ✗ | 43.9 | 73.2 |
| ✓ | ✓ | **45.5** | **74.3** |

## 8. Computational cost

| Backbone | Variant | Input | Parameters | FLOPs | Activation memory | Inference time |
|---|---|---|---:|---:|---:|---:|
| ResNet-50 | Vanilla | 224×224×3 | 25.6M | 3.8G | 98.89 MB | 2.29 ms |
| ResNet-50 | HySCAN | 224×224×3 | 149.8M | 179.03G | 571.4 MB | 15.7 ms |
| ResNet-110 | Vanilla | 32×32×3 | 47.6M | 8.10G | 105.33 MB | 0.33 ms |
| ResNet-110 | HySCAN | 32×32×3 | 101.6M | 8.75G | 387.71 MB | 4.29 ms |

The ImageNet-scale cost increase is driven mainly by repeated SANI evaluation for every RWAN gate and every block-level feature injection. Certification further multiplies full-forward cost by approximately `n₀+n` per example.

## 9. Interpretation boundaries

- Certified and empirical results use different norms and protocols; they should not be merged into one undifferentiated robustness score.
- Certified accuracy depends on `σ`, target radius, confidence level, test subset, and full joint-randomness sampling.
- Empirical accuracy depends on the attack implementation, number of steps/restarts, EOT/BPDA treatment, and whether internal stochasticity remains active.
- Compare methods only under matched backbones, data splits, preprocessing, budgets, and evaluation procedures.
