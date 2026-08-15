# Dataset and Evaluation Inventory

HySCAN is evaluated on eight image-classification benchmarks spanning natural images and medical imaging. The repository does not redistribute any dataset; obtain each dataset from its official source and comply with its license and access conditions.

## 1. Benchmark matrix

| Dataset | Domain | Paper backbone | Certified evaluation | Empirical evaluation |
|---|---|---|---|---|
| CIFAR-10 | Natural image | ResNet-110 | Yes | Yes |
| CIFAR-100 | Natural image | ResNet-110 | Yes / extended analysis | Stronger-PGD stress tests |
| ImageNet-1k | Natural image | ResNet-50 | Yes | Yes |
| CelebA | Face image | ResNet-50 | Yes | Not a main empirical table |
| NCT-CRC-HE-100K | Medical image | ResNet-50 | Yes | APGD-20 and AA-20 |
| NIH-CXR | Medical image | ResNet-50 | Yes | APGD-20, AA-20, and adaptive attacks |
| EyePACS | Medical image | ResNet-50 | Yes | APGD-20 and AA-20 |
| HAM10000 | Medical image | ResNet-50 | Yes | APGD-20 and AA-20 |

## 2. Certified evaluation

The paper uses Gaussian smoothing levels

```text
σ ∈ {0.25, 0.50, 1.0, 2.0}
```

and reports certified accuracy at predetermined ℓ₂ radii. The main natural-image table reports ImageNet and CIFAR-10 at radii from `0` to `2.0`; the main medical table reports NIH-CXR, HAM10000, and CelebA at radii `0`, `0.5`, and `1.0`. The appendix additionally reports EyePACS and NCT-CRC-HE-100K at radii `0`, `0.25`, `0.5`, `0.75`, and `1.0`.

See [CERTIFICATION.md](CERTIFICATION.md) for the dataset-specific subset rules.

## 3. Empirical evaluation

The main medical-image tables evaluate:

```text
APGD-20 and AA-20
ε ∈ {8/255, 16/255}
```

The extended natural-image table evaluates CIFAR-10 and ImageNet at `ε=8/255` with PGD-20 and AA-20. Further stress tests vary PGD perturbation strength from `0.01` to `0.08`, the number of PGD steps from `10` to `100`, and include EOT-PGD, BPDA, and EOT-BPDA.

## 4. Paper optimization recipes

| Dataset family | Epochs | Batch size | Optimizer | Schedule |
|---|---:|---:|---|---|
| CIFAR-10/100 | 200 | 256 | AdamW, LR `1e-2`, weight decay `1e-4` | Step size 30, multiplier `0.1` |
| CelebA and four medical datasets | 200 | 64 | SGD, LR `5e-2` | Step size 3, multiplier `0.8` |
| ImageNet-1k | 200 | 300 | SGD, LR `1e-1`, momentum `0.9`, weight decay `1e-4` | Short warm-up; step size 30, multiplier `0.1` |

HySCAN's RWAN and SANI stochasticity remains active throughout training. Certified models are trained with Gaussian augmentation at the same `σ` used during certification.

## 5. Reproducibility metadata to preserve

For every dataset, store:

- source and version;
- download checksum or archive identifier;
- class mapping and target definition;
- preprocessing and normalization;
- train/validation/test indices;
- certification subset indices;
- augmentation settings;
- model backbone and all HySCAN noise parameters;
- seed list;
- attack and certification configurations.

The paper reports averages over five random seeds. Exact seed values and complete split manifests are not included in the current notebook release and should be published for table-level reproduction.

## 6. Current notebook data interface

`Code/hyscan.ipynb` expects two locally prepared NumPy files:

```text
X.npy
Y.npy
```

The notebook performs a 20% test split and then uses 20% of the remaining data for validation with `random_state=42`, corresponding to approximately 64% training, 16% validation, and 20% testing. This generic array workflow is not a substitute for the paper's dataset-specific loaders and released split manifests.
