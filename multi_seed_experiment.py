#!/usr/bin/env python3
"""
Multi-seed rerun of the neural experiment for statistical robustness.
Runs 10 seeds and reports mean ± std for all metrics.
Also includes ablations suggested by review process.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

k_B = 1.380649e-23
T = 300.0
LANDAUER = k_B * T * np.log(2)

INPUT_DIM = 8
N_CLASSES = 2
N_TRAIN = 2000
EPOCHS = 80
N_SEEDS = 10


def generate_data(n, noise_scale=0.3, noise_dim_scale=1.0, ood=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    z = rng.integers(0, 2, size=n)
    relevant = np.zeros((n, 4))
    for i in range(4):
        relevant[:, i] = z + rng.standard_normal(n) * noise_scale
    if ood:
        noise = rng.standard_normal((n, 4)) * 2.0 * noise_dim_scale + 1.5
    else:
        noise = rng.standard_normal((n, 4)) * noise_dim_scale
    x = np.hstack([relevant, noise]).astype(np.float32)
    y = z.astype(np.int64)
    return x, y


class BottleneckNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, N_CLASSES))

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_representation(self, x):
        with torch.no_grad():
            return self.encoder(x)


class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)

    def get_representation(self, x):
        with torch.no_grad():
            return self.net[3](self.net[2](self.net[1](self.net[0](x))))


class NoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, N_CLASSES))

    def forward(self, x):
        return self.decoder(self.encoder(x[:, 4:]))

    def get_representation(self, x):
        with torch.no_grad():
            return self.encoder(x[:, 4:])


def count_flops(model):
    flops = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features
    return flops


def train_model(model, loader, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    flops_per = count_flops(model)
    total_flops = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_flops += 3 * flops_per * xb.shape[0]
    return total_flops


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(dim=1).numpy()
    return (pred == y).mean()


def estimate_cu(model, X, y, rep_dim):
    model.eval()
    rep = model.get_representation(torch.from_numpy(X))
    probe = nn.Linear(rep_dim, 2)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    n = len(X)
    split = int(0.8 * n)
    yt = torch.from_numpy(y)
    for _ in range(200):
        opt.zero_grad()
        nn.CrossEntropyLoss()(probe(rep[:split]), yt[:split]).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        ce = nn.CrossEntropyLoss()(probe(rep[split:]), yt[split:]).item()
    return max(0, 1.0 - ce / np.log(2))


def measure_adaptive_reach(model, n_envs=200, theta=0.7):
    model.eval()
    rng = np.random.default_rng()
    successes = 0
    for _ in range(n_envs):
        noise_shift = rng.uniform(-3, 3)
        noise_var = rng.uniform(0.5, 5.0)
        obs_noise = rng.uniform(0.1, 1.0)
        n_t = 100
        z = rng.integers(0, 2, size=n_t)
        rel = np.stack([z + rng.standard_normal(n_t) * obs_noise for _ in range(4)], axis=1)
        noi = rng.standard_normal((n_t, 4)) * noise_var + noise_shift
        X = np.hstack([rel, noi]).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(X)).argmax(dim=1).numpy()
        successes += ((pred == z).mean() >= theta)
    return successes / n_envs


# ─── Run multi-seed ───
print("=" * 70)
print(f"MULTI-SEED EXPERIMENT ({N_SEEDS} seeds)")
print("=" * 70)
print()

all_results = {name: {m: [] for m in ['cu', 'i_eff', 'acc_id', 'acc_ood', 'A', 'flops']}
               for name in ['Bottleneck', 'Wide', 'Noise-only']}

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_tr, y_tr = generate_data(N_TRAIN, rng=rng)
    X_id, y_id = generate_data(500, rng=rng)
    X_ood, y_ood = generate_data(500, ood=True, rng=rng)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=64, shuffle=True)

    for name, ModelClass, rep_dim in [
        ('Bottleneck', BottleneckNet, 2),
        ('Wide', WideNet, 64),
        ('Noise-only', NoiseNet, 2),
    ]:
        model = ModelClass()
        flops = train_model(model, loader)
        cu = estimate_cu(model, X_tr, y_tr, rep_dim)
        acc_id = evaluate(model, X_id, y_id)
        acc_ood = evaluate(model, X_ood, y_ood)

        np.random.seed(seed + 1000)  # separate seed for adaptive reach
        A = measure_adaptive_reach(model)

        i_eff = cu / flops if flops > 0 else 0

        all_results[name]['cu'].append(cu)
        all_results[name]['i_eff'].append(i_eff)
        all_results[name]['acc_id'].append(acc_id)
        all_results[name]['acc_ood'].append(acc_ood)
        all_results[name]['A'].append(A)
        all_results[name]['flops'].append(flops)

    print(f"Seed {seed}: done")

print()
print("=" * 70)
print("RESULTS (mean ± std over 10 seeds)")
print("=" * 70)
print()

header = f"{'Agent':<15} {'C_u':>14} {'I*_eff':>18} {'Acc(ID)':>14} {'Acc(OOD)':>14} {'A':>14}"
print(header)
print("─" * 95)

for name in ['Bottleneck', 'Wide', 'Noise-only']:
    r = all_results[name]
    cu_m, cu_s = np.mean(r['cu']), np.std(r['cu'])
    ie_m, ie_s = np.mean(r['i_eff']), np.std(r['i_eff'])
    ai_m, ai_s = np.mean(r['acc_id']), np.std(r['acc_id'])
    ao_m, ao_s = np.mean(r['acc_ood']), np.std(r['acc_ood'])
    a_m, a_s = np.mean(r['A']), np.std(r['A'])
    print(f"{name:<15} {cu_m:>6.4f}±{cu_s:<5.4f} {ie_m:>8.2e}±{ie_s:<7.2e} {ai_m:>6.3f}±{ai_s:<5.3f} {ao_m:>6.3f}±{ao_s:<5.3f} {a_m:>6.3f}±{a_s:<5.3f}")

print()

# Statistical significance: is Bottleneck I_eff > Wide I_eff?
from scipy import stats

ie_bn = all_results['Bottleneck']['i_eff']
ie_wide = all_results['Wide']['i_eff']
ie_noise = all_results['Noise-only']['i_eff']

t_bn_wide, p_bn_wide = stats.ttest_ind(ie_bn, ie_wide)
t_bn_noise, p_bn_noise = stats.ttest_ind(ie_bn, ie_noise)

print("STATISTICAL TESTS (two-sample t-test on I*_eff):")
print(f"  Bottleneck vs Wide:  t={t_bn_wide:.2f}, p={p_bn_wide:.2e}")
print(f"  Bottleneck vs Noise: t={t_bn_noise:.2f}, p={p_bn_noise:.2e}")
print()

# C_u vs param count correlation with A
cu_all = [np.mean(all_results[n]['cu']) for n in ['Bottleneck', 'Wide', 'Noise-only']]
a_all = [np.mean(all_results[n]['A']) for n in ['Bottleneck', 'Wide', 'Noise-only']]
params = [132, 4866, 100]

r_cu_a, _ = stats.pearsonr(cu_all, a_all)
r_params_a, _ = stats.pearsonr(params, a_all)

print("CORRELATION WITH ADAPTIVE REACH:")
print(f"  Pearson r(C_u, A) = {r_cu_a:.4f}")
print(f"  Pearson r(params, A) = {r_params_a:.4f}")
print(f"  C_u predicts A {'better' if abs(r_cu_a) > abs(r_params_a) else 'worse'} than param count")
print()

# Effect sizes
cu_bn = all_results['Bottleneck']['cu']
cu_noise = all_results['Noise-only']['cu']
cohens_d = (np.mean(cu_bn) - np.mean(cu_noise)) / np.sqrt((np.var(cu_bn) + np.var(cu_noise)) / 2)
print(f"Cohen's d (C_u: Bottleneck vs Noise): {cohens_d:.2f}")
print()

# Save raw results for reproducibility
with open('/Users/biobook/Code/fti/multi_seed_results.json', 'w') as f:
    json.dump({k: {m: [float(v) for v in vals] for m, vals in metrics.items()}
               for k, metrics in all_results.items()}, f, indent=2)
print("Raw results saved to multi_seed_results.json")
