# HySCAN Method Guide

## 1. Objective

HySCAN is designed to narrow the practical gap between two robustness regimes:

- **Certified ℓ₂ robustness**, obtained through randomized smoothing.
- **Empirical ℓ∞ robustness**, measured with strong white-box and adaptive attacks.

Instead of placing randomness at one isolated locus, HySCAN distributes freshly resampled, attention-conditioned stochasticity across both convolutional operators and intermediate representations.

## 2. HySCAN block

For block input $x_i$, RWAN forms a stochastic residual transformation $G_i^{J_i}(x_i;\omega_i)$ and SANI then perturbs the resulting block representation:

$$
z_i(x_i)=S_i\!\left(x_i+G_i^{J_i}(x_i;\omega_i);\psi_i\right).
$$

Here:

- $\omega_i$ collects RWAN's sampled weight noise and gating randomness.
- $\psi_i$ collects SANI's feature-injection randomness.
- Both are resampled on each forward pass.

In a ResNet-style backbone, every standard convolution inside a residual block is replaced by RWAN, and SANI is applied after the residual addition.

## 3. RWAN — Random Weights with Attention Noise

### 3.1 Motivation

Uniform Gaussian perturbation of every kernel coefficient is stationary and feature-agnostic. It may harm discriminative filters and can be estimated by attackers through repeated queries. RWAN makes the perturbation **input-conditioned, channel-selective, and heteroscedastic**.

### 3.2 Construction

For layer $j$ in block $i$:

1. Sample kernel noise:

   $$
   \xi_j^i\sim\mathcal{N}(0,\sigma_w^2I).
   $$

2. Obtain a SANI attention-noise response from the current feature map and average it over batch and spatial dimensions.

3. Apply max-absolute normalization to create a per-input-channel gate $g_j^i(x_j^i)$.

4. Learn a nonnegative noise strength:

   $$
   \rho_j^i=\operatorname{SoftPlus}(\rho_{i,j}^{\mathrm{logit}}).
   $$

5. Form the effective stochastic kernel:

   $$
   \widetilde{W}_j^i(x_j^i;\omega_j^i)
   =W_j^i+\rho_j^i\bigl(\xi_j^i\odot g_j^i(x_j^i)\bigr).
   $$

6. Perform convolution with $\widetilde{W}_j^i$, followed by the backbone's normalization and activation.

### 3.3 Structural properties

The paper establishes that:

- The normalized gate satisfies $\|g_j^i(x_j^i)\|_\infty\le 1$.
- The Frobenius magnitude of the injected kernel perturbation is bounded by the learnable scale times the sampled noise magnitude.
- Conditioned on the input, the RWAN perturbation is Gaussian with channel-dependent variance.

RWAN therefore behaves as an input-conditioned stochastic ensemble of convolutional operators rather than a single deterministic kernel.

## 4. SANI — Stochastic Attention Noise Injection

### 4.1 Dual-branch learner

Given a feature map $u$, SANI creates two residual perturbations:

- $\Delta^{\mathrm{rec}}=u^{\mathrm{rec}}-u$ from **RecRAN**.
- $\Delta^{\mathrm{nor}}=u^{\mathrm{nor}}-u$ from **NoRAN**.

It fuses them as

$$
\mathcal{A}(u;\psi)=
\frac{\lambda_{\mathrm{rec}}\Delta^{\mathrm{rec}}+\lambda_{\mathrm{nor}}\Delta^{\mathrm{nor}}}
{\lambda_{\mathrm{rec}}+\lambda_{\mathrm{nor}}+\nu},
$$

where $\lambda_{\mathrm{rec}}$ and $\lambda_{\mathrm{nor}}$ are SoftPlus-parameterized nonnegative scalars and $\nu>0$ provides numerical stability.

### 4.2 RecRAN

RecRAN performs progressive stochastic attention modulation. The paper uses five recursive refinement rounds in its complexity accounting. Its purpose is to encourage stability under repeated stochastic transformations, which supports both smoothing-based certification and empirical robustness.

### 4.3 NoRAN

NoRAN is a non-recursive, single-pass channel-aware branch. It uses multiple global descriptors—minimum, sum, maximum, and average pooling in the paper's formulation—to generate complementary heteroscedastic perturbations.

### 4.4 Two coordinated uses

The same SANI learner is invoked in two ways:

1. **Per RWAN layer:** its response is pooled and normalized to gate kernel noise. This response is not added to the feature stream at that point.
2. **Per HySCAN block:** it is added to the post-residual feature map to explicitly randomize the intermediate representation.

This sharing is the core cross-space coupling: weight-space and feature-space stochasticity are coordinated rather than independently appended.

### 4.5 Scale control

The normalized fusion forms a nonnegative sub-convex mixture. For any convex norm,

$$
\|\mathcal{A}(u;\psi)\|
\le w_{\mathrm{rec}}\|\Delta^{\mathrm{rec}}\|
+w_{\mathrm{nor}}\|\Delta^{\mathrm{nor}}\|
\le \max\left\{\|\Delta^{\mathrm{rec}}\|,\|\Delta^{\mathrm{nor}}\|}\right\}.
$$

This prevents the fusion itself from amplifying noise beyond the larger branch increment.

## 5. Joint randomized classifier

Let input smoothing noise be $\varepsilon\sim\mathcal{N}(0,\sigma^2I)$. HySCAN defines

$$
F_\Theta(x)=\arg\max_{c\in\mathcal{Y}}
\Pr_{\varepsilon,\omega,\psi}
\left[f_\Theta(x+\varepsilon;\omega,\psi)=c\right].
$$

At the population level, the smoothed predictor is deterministic because it marginalizes the complete joint distribution.

## 6. Certified guarantee

For class probabilities

$$
p_c(x)=\Pr_{\varepsilon,\omega,\psi}
[f_\Theta(x+\varepsilon;\omega,\psi)=c],
$$

let $p_A$ be the top-class probability and $p_B$ the largest competing probability. The adopted randomized-smoothing radius is

$$
r_2(x)=\frac{\sigma}{2}
\left(\Phi^{-1}(p_A)-\Phi^{-1}(p_B)\right).
$$

The standard certificate remains valid because $p_A$ and $p_B$ are defined under the same complete randomness used by deployment. See [CERTIFICATION.md](CERTIFICATION.md).

## 7. Empirical robustness mechanism

For empirical attacks, a new internal realization is sampled on every stochastic forward pass. The paper analyzes directional gradient dispersion under $U=(\omega,\psi)$ and decomposes its variance into contributions from SANI and RWAN through the law of total variance.

This does not replace adaptive evaluation. It requires attacks to keep randomness active and, when appropriate, average or differentiate through repeated stochastic realizations using EOT/BPDA-style procedures.

## 8. Training objectives

### 8.1 RS-compatible certified training

Set $\kappa=0$ and optimize the jointly smoothed risk:

$$
\min_\Theta\;\mathbb{E}_{(x,y)}
\mathbb{E}_{\varepsilon,\omega,\psi}
\left[\ell(f_\Theta(x+\varepsilon;\omega,\psi),y)\right].
$$

The smoothing level used for Gaussian augmentation must match the certification setting.

### 8.2 Hybrid certified–empirical training

For empirical ℓ∞ robustness, add a worst-case term:

$$
\min_\Theta \mathbb{E}_{(x,y)}\left[
\mathbb{E}_{\varepsilon,U}\ell(f_\Theta(x+\varepsilon;U),y)
+\kappa\max_{\|\delta\|_\infty\le\epsilon_\infty}
\mathbb{E}_{\varepsilon,U}\ell(f_\Theta(x+\delta+\varepsilon;U),y)
\right].
$$

A fresh internal draw is used at every attack step. The coefficient $\kappa$ controls the certified–empirical trade-off.

## 9. What the ablations establish

On CIFAR-10 at $\sigma=0.25$, certified radius $r=0.75$, and PGD-20 $\epsilon=8/255$:

- SANI alone improves both certified and empirical robustness.
- RWAN alone improves both and is stronger than SANI alone at the selected point.
- RWAN + SANI is strictly strongest.
- Within SANI, RecRAN and NoRAN each contribute; their normalized fusion is strongest.
- Removing RWAN's internal weight noise lowers the certified frontier.
- Normalized SANI fusion outperforms a naive unnormalized sum.

## 10. Computational implications

RWAN's kernel-noise sampling is relatively inexpensive; repeated SANI evaluation is the dominant extra cost. SANI is invoked once per RWAN layer for gating and once per block for feature injection. This effect is particularly pronounced on high-resolution ImageNet feature maps.

The paper identifies possible efficiency directions that preserve certification semantics: reduce RecRAN depth, compute gates more coarsely, or enable HySCAN selectively in later low-resolution stages.
