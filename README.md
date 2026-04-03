<div align="center">

# Thermodynamic Intelligence — Computation Code

**Compute useful internal structure for a physics-grounded definition of intelligence**

<br />

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/fti?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/fti/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
&nbsp;
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
&nbsp;
[![Paper: Entropy](https://img.shields.io/badge/Paper-Entropy_(MDPI)-4B8BBE?style=for-the-badge)](https://www.mdpi.com/journal/entropy)

---

Companion code for **"Intelligence as Useful Structure: A Thermodynamic Perspective"** (Djordjevic, 2026). Exact computations, neural network experiments, plus a 121-test validation suite.

[The Problem](#the-problem) | [Quick Start](#quick-start) | [What's Inside](#whats-inside) | [Results](#results) | [Citation](#citation)

</div>

## The Problem

Most thermodynamic intelligence metrics count all stored information equally. A system that perfectly memorises noise scores the same as one that extracts the signal. That makes no physical sense. A lookup table is not smarter than a compressed model just because it stores more bits.

The paper proposes a usefulness filter: only count internal structure that actually matters for staying alive and acting well. The central quantity is:

$$C_u = I(M_t;\; Z^V_{t+\tau} \mid H_t)$$

This repo lets you compute $C_u$ exactly for a toy environment and estimate it for neural networks. Then check whether it predicts generalisation better than raw parameter count. It does.

## Quick Start

```bash
git clone https://github.com/199-biotechnologies/fti.git
cd fti
pip install numpy torch scipy

# Exact toy computation (reproduces Table I)
python compute_cu.py

# Neural network experiment
python neural_experiment.py

# Full 40-hypothesis stress test
python hypothesis_battery.py
```

## What's Inside

```
fti/
├── compute_cu.py             # Exact C_u for 3 agents in binary environment
├── neural_experiment.py      # Bottleneck vs wide vs noise-only networks
├── multi_seed_experiment.py  # 10-seed robustness (p < 3e-29)
├── validation_suite.py       # 7 agents, 3 MI methods, permutation tests
├── hypothesis_battery.py     # 40 tests across 6 categories
├── ib_validation.py          # Information Bottleneck optimality proof
├── edge_cases.py             # 23 boundary conditions
└── LICENSE
```

### Core Scripts

| Script | What it does | Runtime |
|--------|-------------|---------|
| `compute_cu.py` | Exact $C_u$, $W_\text{diss}$, $I^*_\text{eff}$, $A$ for three toy agents at $T{=}300\text{K}$ | < 1s |
| `neural_experiment.py` | Train bottleneck / wide / noise-only nets, estimate $C_u$ via probes | ~30s |
| `multi_seed_experiment.py` | Statistical robustness across 10 random seeds | ~5 min |

### Validation Scripts

| Script | What it tests | Tests |
|--------|--------------|-------|
| `validation_suite.py` | Cross-validates MI with linear probe, MLP probe, and KSG estimator | 7 agents |
| `hypothesis_battery.py` | Formula bounds, correlated noise, multi-class, adversarial, scaling | 40 |
| `ib_validation.py` | Blahut-Arimoto IB curve, rate-distortion correspondence | 18 |
| `edge_cases.py` | Reversibility paradox, negative learning, continuous Z, temporal MI | 23 |

## Results

Three agents learn from the same binary environment. The rule learner compresses the observation down to just the viability-relevant bit. The lookup table stores everything. The noise memoriser stores the wrong thing.

| Agent | Total stored | Useful ($C_u$) | Dissipation | Efficiency ($I^*_\text{eff}$) | Generalisation ($A$) |
|-------|-------------|----------------|-------------|-------------------------------|----------------------|
| Rule learner | 1 bit | **1 bit** | 16 $k_BT\ln2$ | **0.0625** | **1.00** |
| Lookup table | 2 bits | 1 bit | 24 $k_BT\ln2$ | 0.0417 | 0.54 |
| Noise memoriser | 1 bit | **0 bits** | 16 $k_BT\ln2$ | 0.0000 | 0.00 |

The rule learner is 50% more efficient and has perfect adaptive reach. The noise memoriser stores the same amount of information as the rule learner — but none of it is useful. $C_u$ catches this; raw information metrics don't.

Across 7 neural network variants and 10 seeds:

- **$r(C_u, A) = 0.998$** ($p = 3 \times 10^{-7}$) — useful structure predicts generalisation
- **$r(\text{params}, A) = 0.278$** ($p = 0.55$) — parameter count does not
- **IB validation**: the rule learner sits at the knee of the Information Bottleneck curve

## The Intelligence Profile

The paper separates intelligence into three independent axes:

| Axis | Quantity | Measures |
|------|----------|---------|
| Capability | $A(\pi; \mu)$ | Viability across environments |
| Structure | $C_u = I(M_t; Z^V_{t+\tau} \mid H_t)$ | Relevance-filtered internal model quality |
| Efficiency | $I^*_\text{eff} = \frac{k_B \bar{T} \ln 2 \;\Delta C_u}{W_\text{diss}}$ | Useful bits gained per joule |

Builds on the Information Bottleneck (Tishby et al., 1999) and semantic information (Kolchinsky & Wolpert, 2018). What's new is putting these together: a usefulness-filtered structure term, a thermodynamic efficiency term, and a behavioural reach axis — evaluated as a profile, not collapsed into a single number.

## Citation

```bibtex
@article{djordjevic2026intelligence,
  title={Intelligence as Useful Structure: A Thermodynamic Perspective},
  author={Djordjevic, Boris},
  journal={Entropy},
  year={2026},
  note={Submitted}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [Paperfoot AI](https://paperfoot.com)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/fti?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/fti/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
