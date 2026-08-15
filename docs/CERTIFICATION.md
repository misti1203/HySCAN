# HySCAN Certification and Adaptive Evaluation Guide

## 1. Certification distribution

HySCAN is stochastic even when deployed. A correct certificate must therefore match the deployed predictor rather than certify only the Gaussian input noise while ignoring internal randomness.

For every Monte Carlo query, draw independently:

- Gaussian input noise `ε ~ N(0, σ²I)`.
- RWAN randomness `ω`, including kernel noise and gating randomness.
- SANI randomness `ψ`, including RecRAN and NoRAN feature noise.

The smoothed classifier is the majority prediction under the joint product distribution `(ε, ω, ψ)`.

## 2. Two-stage Monte Carlo procedure

For a test input `x`:

1. Draw `n₀` joint-randomness samples and identify the most frequent class `ĉ_A`.
2. Draw `n` new joint-randomness samples and count predictions equal to `ĉ_A`.
3. Compute a one-sided Clopper–Pearson lower confidence bound `p_A^LB` at failure probability `α`.
4. Use `p_B^UB = 1 - p_A^LB` as a valid upper bound on the strongest competing class.
5. Abstain when `p_A^LB ≤ 1/2`.
6. Otherwise return

   $$
   \widehat r_2(x)=\frac{\sigma}{2}
   \left[\Phi^{-1}(p_A^{LB})-\Phi^{-1}(p_B^{UB})\right]
   =\sigma\Phi^{-1}(p_A^{LB}).
   $$

The resulting certificate has overall failure probability at most `α` under the stated procedure.

## 3. Paper settings

### Default

```text
n₀    = 100
n     = 100,000
α     = 0.001
σ     ∈ {0.25, 0.50, 1.0, 2.0}
```

The paper reports certified accuracy at predetermined ℓ₂ radii rather than only mean certified radius.

### Dataset-specific sampling

| Dataset | Certification selection |
|---|---|
| ImageNet-1k | Every 100th test example |
| CIFAR-10/100 | Every 5th test example |
| CelebA | Uniform label-stratified subset of 200; `n=50,000`, failure probability `0.05` |
| NCT-CRC-HE-100K | Uniform label-stratified subset targeting approximately 2,000 examples |
| NIH-CXR | Uniform label-stratified subset targeting approximately 2,000 examples |
| EyePACS | Uniform label-stratified subset targeting approximately 2,000 examples |
| HAM10000 | Full test split when feasible; otherwise a uniform label-stratified subset |

Persist the exact selected indices with the released results. Recreating a fresh subset can change the reported certified accuracy.

## 4. RS-compatible training

For certification-only experiments, use the same smoothing level during training and certification and set the empirical adversarial term coefficient to `κ = 0`.

Each training forward pass must independently resample:

- one Gaussian perturbation per example;
- RWAN kernel/gating noise;
- SANI feature-injection noise.

Do not freeze one internal stochastic realization across an epoch or across certification samples.

## 5. Empirical ℓ∞ evaluation

The paper evaluates white-box attacks with HySCAN randomness active:

- APGD-20, defined as the union of APGD-CE and APGD-T-DLR, each with 20 iterations and five random restarts;
- AutoAttack;
- PGD stress tests across larger perturbation sizes and 10–100 steps;
- EOT-PGD, BPDA, and EOT-BPDA adaptive evaluations.

Default empirical budgets are `ε ∈ {8/255, 16/255}`. Every stochastic attack forward pass draws fresh internal randomness, and gradients are computed through that realization. EOT evaluations average across repeated stochastic forwards rather than attacking a deterministic or noise-disabled surrogate.

## 6. Common implementation failures

### Certifying a different predictor

**Incorrect:** train and deploy with internal stochasticity but compute certificates after disabling RWAN or SANI.

**Correct:** estimate class probabilities with exactly the same stochastic mechanisms active.

### Reusing noise

Reusing one `ω` or `ψ` realization across all Monte Carlo samples does not estimate the paper's joint class probabilities. Resample all sources independently.

### Ignoring adaptive attacks

A stochastic defense must not be evaluated only with a single-sample gradient. Include EOT/BPDA-style evaluations and report their sample counts.

### Claiming a deterministic Lipschitz radius without a bound

The released notebook contains an illustrative deterministic-margin utility with a user-supplied Lipschitz constant. Do not report this as a valid certificate unless every network component has a proven global Lipschitz bound and the bound is propagated correctly. The paper's main certificate is the joint-randomness randomized-smoothing certificate.

### Mixing confidence conventions

Document whether `α` is one-sided or split across two bounds, and keep the implementation consistent with the reported theorem and table.

### Reporting only non-abstained samples

Certified accuracy must use the full certified evaluation subset as the denominator, counting abstentions and incorrect predictions appropriately.

## 7. Computational cost

Per-example certification requires approximately `(n₀+n)` full HySCAN forward passes. With `n=100,000`, certification is expensive by construction, and every pass includes both RWAN and SANI.

Practical implementation should:

- batch Monte Carlo samples;
- use deterministic index manifests for resumability;
- checkpoint class counts periodically;
- store `σ`, `n₀`, `n`, `α`, model checksum, seed, subset indices, and software versions beside each certificate;
- verify that batching does not accidentally share one random draw across examples or samples.

## 8. Minimal result record

A reproducible certificate file should contain at least:

```yaml
model_checkpoint_sha256: ...
dataset: ...
test_subset_indices: ...
sigma: 0.5
n0: 100
n: 100000
alpha: 0.001
internal_randomness_active: true
class_counts: ...
predicted_class: ...
pA_lower: ...
radius_l2: ...
abstained: false
seed: ...
```
