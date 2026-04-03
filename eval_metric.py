#!/usr/bin/env python3
"""
Evaluation script for autoresearch.
Metric: I_eff efficiency ratio (Bottleneck / Wide).
Higher = better demonstration that compression helps.

Also gates on:
- C_u(Bottleneck) > 0.8 (must still learn useful structure)
- C_u(Noise) < 0.1 (noise agent must not cheat)
- A(Bottleneck) > 0.9 (must generalise)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import json

# ─── Config (autoresearch modifies these) ───
BOTTLENECK_DIM = 2
HIDDEN_DIM_BN = 8
HIDDEN_DIM_WIDE = 64
N_TRAIN = 2000
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 64
NOISE_SCALE = 0.3
NOISE_DIM_SCALE = 1.0
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
N_RELEVANT = 4
N_NOISE = 4
INPUT_DIM = N_RELEVANT + N_NOISE
N_CLASSES = 2
N_SEEDS = 5  # fewer seeds for speed during autoresearch

# ─── Environment ───
def generate_data(n, ood=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    z = rng.integers(0, 2, size=n)
    relevant = np.stack([z + rng.standard_normal(n) * NOISE_SCALE for _ in range(N_RELEVANT)], axis=1)
    if ood:
        noise = rng.standard_normal((n, N_NOISE)) * 2.0 * NOISE_DIM_SCALE + 1.5
    else:
        noise = rng.standard_normal((n, N_NOISE)) * NOISE_DIM_SCALE
    return np.hstack([relevant, noise]).astype(np.float32), z.astype(np.int64)


# ─── Models ───
class BottleneckNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM_BN), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_BN, BOTTLENECK_DIM),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM_BN), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_BN, N_CLASSES),
        )
    def forward(self, x): return self.decoder(self.encoder(x))
    def get_rep(self, x):
        with torch.no_grad(): return self.encoder(x)

class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM_WIDE), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM_WIDE, HIDDEN_DIM_WIDE), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_WIDE, N_CLASSES),
        )
    def forward(self, x): return self.net(x)
    def get_rep(self, x):
        with torch.no_grad():
            return self.net[3](self.net[2](self.net[1](self.net[0](x))))

class NoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_NOISE, HIDDEN_DIM_BN), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_BN, BOTTLENECK_DIM),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(BOTTLENECK_DIM, HIDDEN_DIM_BN), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_BN, N_CLASSES),
        )
    def forward(self, x): return self.decoder(self.encoder(x[:, N_RELEVANT:]))
    def get_rep(self, x):
        with torch.no_grad(): return self.encoder(x[:, N_RELEVANT:])


def count_flops(model):
    return sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))

def train(model, loader):
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    f = count_flops(model)
    total = 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            total += 3 * f * xb.shape[0]
    return total

def estimate_cu(model, X, y, dim):
    model.eval()
    rep = model.get_rep(torch.from_numpy(X))
    probe = nn.Linear(dim, 2)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    yt = torch.from_numpy(y)
    s = int(0.8 * len(X))
    for _ in range(200):
        opt.zero_grad(); nn.CrossEntropyLoss()(probe(rep[:s]), yt[:s]).backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        ce = nn.CrossEntropyLoss()(probe(rep[s:]), yt[s:]).item()
    return max(0, 1.0 - ce / np.log(2))

def adaptive_reach(model, n_envs=100, theta=0.7):
    model.eval()
    rng = np.random.default_rng()
    ok = 0
    for _ in range(n_envs):
        ns, nv, on = rng.uniform(-3,3), rng.uniform(0.5,5), rng.uniform(0.1,1)
        z = rng.integers(0,2,size=100)
        rel = np.stack([z + rng.standard_normal(100)*on for _ in range(N_RELEVANT)], axis=1)
        noi = rng.standard_normal((100, N_NOISE))*nv + ns
        X = np.hstack([rel, noi]).astype(np.float32)
        with torch.no_grad():
            p = model(torch.from_numpy(X)).argmax(1).numpy()
        ok += ((p == z).mean() >= theta)
    return ok / n_envs


# ─── Main eval ───
results = {n: {'cu': [], 'ieff': [], 'A': []} for n in ['bn', 'wide', 'noise']}

for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    X, y = generate_data(N_TRAIN, rng=rng)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=BATCH_SIZE, shuffle=True)

    for tag, Cls, dim in [('bn', BottleneckNet, BOTTLENECK_DIM),
                          ('wide', WideNet, HIDDEN_DIM_WIDE),
                          ('noise', NoiseNet, BOTTLENECK_DIM)]:
        m = Cls()
        flops = train(m, loader)
        cu = estimate_cu(m, X, y, dim)
        np.random.seed(seed+1000)
        A = adaptive_reach(m)
        results[tag]['cu'].append(cu)
        results[tag]['ieff'].append(cu / flops if flops > 0 else 0)
        results[tag]['A'].append(A)

# Compute metric
bn_ieff = np.mean(results['bn']['ieff'])
wide_ieff = np.mean(results['wide']['ieff'])
bn_cu = np.mean(results['bn']['cu'])
noise_cu = np.mean(results['noise']['cu'])
bn_A = np.mean(results['bn']['A'])

# Quality gates
if bn_cu < 0.8:
    print(f"GATE FAIL: C_u(BN) = {bn_cu:.4f} < 0.8", file=sys.stderr)
    print("0.0")
    sys.exit(0)
if noise_cu > 0.1:
    print(f"GATE FAIL: C_u(Noise) = {noise_cu:.4f} > 0.1", file=sys.stderr)
    print("0.0")
    sys.exit(0)
if bn_A < 0.9:
    print(f"GATE FAIL: A(BN) = {bn_A:.3f} < 0.9", file=sys.stderr)
    print("0.0")
    sys.exit(0)

ratio = bn_ieff / wide_ieff if wide_ieff > 0 else 0
print(f"{ratio:.4f}")

# Debug info to stderr
print(f"C_u: bn={bn_cu:.4f} wide={np.mean(results['wide']['cu']):.4f} noise={noise_cu:.4f}", file=sys.stderr)
print(f"I_eff: bn={bn_ieff:.2e} wide={wide_ieff:.2e}", file=sys.stderr)
print(f"A: bn={bn_A:.3f} wide={np.mean(results['wide']['A']):.3f} noise={np.mean(results['noise']['A']):.3f}", file=sys.stderr)
print(f"Ratio: {ratio:.4f}", file=sys.stderr)
