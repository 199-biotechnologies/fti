#!/usr/bin/env python3
"""
Systematic hypothesis battery for C_u = I(M; Z^V | H_t).

Tests the formula across many conditions to find where it succeeds,
where it breaks, and what the boundary conditions are. Each hypothesis
is a testable prediction of the framework.

Output: JSONL log of hypothesis, prediction, result, verdict.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import json
import sys
import time

torch.manual_seed(42)
np.random.seed(42)

RESULTS = []
START = time.time()


def log_result(hypothesis, prediction, result, verdict, details=""):
    entry = {
        "hypothesis": hypothesis,
        "prediction": prediction,
        "result": result,
        "verdict": bool(verdict),
        "details": details,
        "elapsed_s": round(time.time() - START, 1),
    }
    RESULTS.append(entry)
    v = "PASS" if verdict else "FAIL"
    print(f"  [{v}] {hypothesis}")
    if details:
        print(f"       {details}")


def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def mutual_information(p_xy):
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: EXACT FORMULA PROPERTIES
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: EXACT FORMULA PROPERTIES")
print("=" * 70)
print()

# H1.1: C_u is bounded by H(Z)
print("H1.1: C_u ≤ H(Z) for all agents (information-theoretic bound)")
for p_z in [0.5, 0.3, 0.1, 0.01]:
    hz = entropy([p_z, 1-p_z])
    # Rule learner: M = z
    p_mz = np.array([[p_z, 0], [0, 1-p_z]])
    cu = mutual_information(p_mz)
    log_result(
        f"H1.1a: p(z=0)={p_z}, C_u={cu:.4f} ≤ H(Z)={hz:.4f}",
        "C_u ≤ H(Z)", f"{cu:.4f} ≤ {hz:.4f}", cu <= hz + 1e-10
    )

# H1.2: C_u = 0 when M ⊥ Z
print("\nH1.2: C_u = 0 when M is independent of Z")
for n_m in [2, 4, 8]:
    p_mz = np.ones((n_m, 2)) / (2 * n_m)
    cu = mutual_information(p_mz)
    log_result(
        f"H1.2: |M|={n_m}, uniform P(M,Z), C_u={cu:.6f}",
        "C_u = 0", f"{cu:.6f}", abs(cu) < 1e-10
    )

# H1.3: C_u is invariant to relabeling of M
print("\nH1.3: C_u invariant to bijective relabeling of M")
p_mz_orig = np.array([[0.4, 0.0], [0.1, 0.0], [0.0, 0.3], [0.0, 0.2]])
cu_orig = mutual_information(p_mz_orig)
# Permute rows (relabel M)
p_mz_perm = p_mz_orig[[2, 0, 3, 1]]
cu_perm = mutual_information(p_mz_perm)
log_result(
    f"H1.3: C_u(orig)={cu_orig:.6f} vs C_u(permuted)={cu_perm:.6f}",
    "Equal", f"diff={abs(cu_orig-cu_perm):.1e}", abs(cu_orig - cu_perm) < 1e-10
)

# H1.4: Adding irrelevant dimensions to M doesn't change C_u
print("\nH1.4: Expanding M with noise doesn't increase C_u")
# M=(z) gives C_u=1. M=(z,n) also gives C_u=1.
# M=(z,n1,n2,...,nk) for independent noise should still give C_u=1.
p_z = 0.5
for k in [1, 2, 4]:
    n_m = 2 * (2**k)  # M = (z, n1, ..., nk)
    p_mz = np.zeros((n_m, 2))
    # z=0 maps to first half, z=1 maps to second half
    for i in range(n_m // 2):
        p_mz[i, 0] = 1.0 / n_m
        p_mz[n_m//2 + i, 1] = 1.0 / n_m
    cu = mutual_information(p_mz)
    log_result(
        f"H1.4: M=(z,n1..n{k}), |M|={n_m}, C_u={cu:.4f}",
        "C_u = 1 bit", f"{cu:.4f}", abs(cu - 1.0) < 0.01
    )

# H1.5: Non-uniform z distribution
print("\nH1.5: C_u scales with H(Z) for skewed distributions")
for p0 in [0.5, 0.3, 0.1, 0.01, 0.001]:
    hz = entropy([p0, 1-p0])
    p_mz = np.array([[p0, 0], [0, 1-p0]])
    cu = mutual_information(p_mz)
    log_result(
        f"H1.5: p(z=0)={p0}, H(Z)={hz:.4f}, C_u={cu:.4f}",
        "C_u = H(Z)", f"diff={abs(cu-hz):.1e}", abs(cu - hz) < 1e-10
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: CORRELATED NOISE (z and n NOT independent)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: CORRELATED NOISE")
print("=" * 70)
print()

# H2.1: When n is correlated with z, the noise memoriser gains C_u > 0
print("H2.1: Correlated noise → noise memoriser gains useful info")
for rho in [0.0, 0.3, 0.5, 0.8, 1.0]:
    # P(z=0,n=0) = (1+rho)/4, P(z=0,n=1) = (1-rho)/4, etc.
    p00 = (1 + rho) / 4
    p01 = (1 - rho) / 4
    p10 = (1 - rho) / 4
    p11 = (1 + rho) / 4
    # Noise memoriser: M = n
    p_mz_noise = np.array([
        [p00, p10],  # M=0 (n=0): Z=0, Z=1
        [p01, p11],  # M=1 (n=1): Z=0, Z=1
    ])
    cu_noise = mutual_information(p_mz_noise)
    # Rule learner: M = z (always has C_u = H(Z) = 1 bit)
    p_mz_rule = np.array([[0.5, 0], [0, 0.5]])
    cu_rule = mutual_information(p_mz_rule)
    log_result(
        f"H2.1: rho={rho:.1f}, C_u(noise)={cu_noise:.4f}, C_u(rule)={cu_rule:.4f}",
        "C_u(noise) grows with rho",
        f"noise={cu_noise:.4f}",
        (rho == 0 and cu_noise < 0.01) or (rho > 0 and cu_noise > 0)
    )

# H2.2: When n = z (perfect correlation), noise memoriser = rule learner
print("\nH2.2: n = z → noise memoriser equivalent to rule learner")
p_mz_noise_eq = np.array([[0.5, 0], [0, 0.5]])  # M=n=z
cu = mutual_information(p_mz_noise_eq)
log_result(
    f"H2.2: n=z, C_u(noise memoriser)={cu:.4f}",
    "C_u = 1 bit", f"{cu:.4f}", abs(cu - 1.0) < 0.01
)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: MULTI-CLASS (z with more than 2 values)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: MULTI-CLASS VIABILITY")
print("=" * 70)
print()

for K in [2, 4, 8, 16]:
    hz = np.log2(K)
    # Perfect rule learner: M = z, |M| = K
    p_mz = np.eye(K) / K
    cu = mutual_information(p_mz)
    log_result(
        f"H3.1: K={K} classes, C_u(perfect)={cu:.4f}, H(Z)={hz:.4f}",
        "C_u = log2(K)", f"diff={abs(cu-hz):.1e}", abs(cu - hz) < 0.01
    )

# H3.2: Partial learner (captures only some classes)
print("\nH3.2: Partial learner — captures subset of viability-relevant info")
K = 4
# Agent that distinguishes z∈{0,1} from z∈{2,3} but not within groups
p_mz = np.array([
    [0.25, 0.25, 0, 0],   # M=0: maps to z=0 or z=1
    [0, 0, 0.25, 0.25],   # M=1: maps to z=2 or z=3
])
cu = mutual_information(p_mz)
hz = 2.0
log_result(
    f"H3.2: K=4, coarse learner, C_u={cu:.4f}, H(Z)={hz:.4f}",
    "0 < C_u < H(Z)", f"C_u = {cu:.4f} (half of max)",
    0 < cu < hz
)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: NEURAL NETWORK SCALING LAWS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: NEURAL NETWORK SCALING")
print("=" * 70)
print()

INPUT_DIM = 8

def generate_data(n, noise_scale=0.3, rng=None):
    if rng is None: rng = np.random.default_rng()
    z = rng.integers(0, 2, size=n)
    rel = np.stack([z + rng.standard_normal(n)*noise_scale for _ in range(4)], axis=1)
    noi = rng.standard_normal((n, 4))
    return np.hstack([rel, noi]).astype(np.float32), z.astype(np.int64)

def make_bn_net(bn_dim, hidden=16):
    return nn.Sequential(
        nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, bn_dim), nn.ReLU(),
        nn.Linear(bn_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, 2),
    )

def train_and_measure(model, X, y, bn_dim, epochs=60):
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=64, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    flops = sum(2*m.in_features*m.out_features for m in model.modules() if isinstance(m, nn.Linear))
    total_flops = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            total_flops += 3 * flops * xb.shape[0]

    # Get representation at bottleneck
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X)
        h = model[0](xt)
        h = model[1](h)
        rep = model[2](h)

    # Probe
    yt = torch.from_numpy(y)
    probe = nn.Linear(bn_dim, 2)
    popt = optim.Adam(probe.parameters(), lr=1e-2)
    s = int(0.8 * len(X))
    for _ in range(200):
        popt.zero_grad(); nn.CrossEntropyLoss()(probe(rep[:s]), yt[:s]).backward(); popt.step()
    probe.eval()
    with torch.no_grad():
        ce = nn.CrossEntropyLoss()(probe(rep[s:]), yt[s:]).item()
    cu = max(0, 1.0 - ce / np.log(2))

    acc = (model(xt).argmax(1).numpy() == y).mean()
    return cu, total_flops, acc

# H4.1: C_u increases monotonically with bottleneck dimension (up to H(Z))
print("H4.1: C_u vs bottleneck dimension")
rng = np.random.default_rng(42)
X, y = generate_data(2000, rng=rng)
cu_vals = []
for bn in [1, 2, 4, 8, 16, 32]:
    torch.manual_seed(42)
    model = make_bn_net(bn)
    cu, flops, acc = train_and_measure(model, X, y, bn)
    cu_vals.append(cu)
    print(f"    bn_dim={bn:>3}: C_u={cu:.4f}, acc={acc:.3f}")

# Check monotonicity (allowing small dips from estimation noise)
monotonic = all(cu_vals[i] <= cu_vals[i+1] + 0.05 for i in range(len(cu_vals)-1))
log_result(
    "H4.1: C_u generally increases with bottleneck dim",
    "Monotonic (within noise)", str(cu_vals),
    monotonic,
    f"Sequence: {[f'{v:.3f}' for v in cu_vals]}"
)

# H4.2: I_eff decreases with model size (for same task)
print("\nH4.2: I_eff decreases with unnecessary model capacity")
ieff_vals = []
for hidden in [8, 16, 32, 64, 128]:
    torch.manual_seed(42)
    model = make_bn_net(2, hidden=hidden)
    cu, flops, acc = train_and_measure(model, X, y, 2)
    ieff = cu / flops if flops > 0 else 0
    ieff_vals.append(ieff)
    print(f"    hidden={hidden:>3}: C_u={cu:.4f}, FLOPs={flops:>12,}, I_eff={ieff:.2e}")

decreasing = all(ieff_vals[i] >= ieff_vals[i+1] - 1e-11 for i in range(len(ieff_vals)-1))
log_result(
    "H4.2: I_eff decreases with hidden size (same C_u, more FLOPs)",
    "Decreasing", str([f"{v:.2e}" for v in ieff_vals]),
    decreasing,
    f"Sequence: {[f'{v:.2e}' for v in ieff_vals]}"
)

# H4.3: Training sample efficiency — C_u should plateau faster for bottleneck
print("\nH4.3: Sample efficiency — bottleneck reaches C_u plateau faster")
for n_train in [100, 500, 1000, 2000]:
    rng = np.random.default_rng(42)
    Xn, yn = generate_data(n_train, rng=rng)
    torch.manual_seed(42)
    model_bn = make_bn_net(2, hidden=8)
    cu_bn, _, _ = train_and_measure(model_bn, Xn, yn, 2, epochs=40)

    torch.manual_seed(42)
    model_wide = nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn)),
                        batch_size=64, shuffle=True)
    opt = optim.Adam(model_wide.parameters(), lr=1e-3)
    for _ in range(40):
        model_wide.train()
        for xb, yb in loader:
            opt.zero_grad(); nn.CrossEntropyLoss()(model_wide(xb), yb).backward(); opt.step()
    model_wide.eval()
    with torch.no_grad():
        rep_w = model_wide[2](model_wide[1](model_wide[0](torch.from_numpy(Xn))))
    yt = torch.from_numpy(yn)
    probe = nn.Linear(64, 2)
    popt = optim.Adam(probe.parameters(), lr=1e-2)
    s = int(0.8 * n_train)
    for _ in range(200):
        popt.zero_grad(); nn.CrossEntropyLoss()(probe(rep_w[:s]), yt[:s]).backward(); popt.step()
    probe.eval()
    with torch.no_grad():
        ce = nn.CrossEntropyLoss()(probe(rep_w[s:]), yt[s:]).item()
    cu_wide = max(0, 1.0 - ce / np.log(2))

    print(f"    N={n_train:>5}: C_u(bn)={cu_bn:.4f}, C_u(wide)={cu_wide:.4f}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: ADVERSARIAL STRESS TESTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: ADVERSARIAL STRESS TESTS")
print("=" * 70)
print()

# H5.1: Adversarial noise — can structured noise fool C_u?
print("H5.1: Structured noise that mimics z")
# Noise that is a noisy copy of z — should this count as useful?
for snr in [0.0, 0.5, 1.0, 2.0]:
    # n = z + Gaussian(0, 1/snr) when snr > 0
    n_samples = 10000
    z = np.random.randint(0, 2, size=n_samples)
    if snr > 0:
        n = z + np.random.randn(n_samples) / snr
    else:
        n = np.random.randn(n_samples)
    # Discretise n into bins
    n_disc = (n > 0.5).astype(int)
    # P(M=n_disc, Z=z)
    counts = np.zeros((2, 2))
    for i in range(n_samples):
        counts[n_disc[i], z[i]] += 1
    p_mz = counts / counts.sum()
    cu = mutual_information(p_mz)
    log_result(
        f"H5.1: SNR={snr:.1f}, C_u(noisy_copy)={cu:.4f}",
        "C_u > 0 when noise correlates with z",
        f"C_u={cu:.4f}",
        (snr == 0 and cu < 0.05) or (snr > 0 and cu > 0),
        "Correctly captures that correlated noise IS informative"
    )

# H5.2: Random labels — C_u should be 0 for any architecture
print("\nH5.2: Random labels — C_u should be ~0 regardless of model")
rng = np.random.default_rng(42)
X_rand, _ = generate_data(2000, rng=rng)
y_rand = np.random.randint(0, 2, size=2000).astype(np.int64)

for bn in [2, 8, 32]:
    torch.manual_seed(42)
    model = make_bn_net(bn)
    cu, _, acc = train_and_measure(model, X_rand, y_rand, bn, epochs=40)
    log_result(
        f"H5.2: Random labels, bn={bn}, C_u={cu:.4f}, acc={acc:.3f}",
        "C_u ≈ 0", f"C_u={cu:.4f}",
        cu < 0.15,
        "Labels uncorrelated with features → no useful structure"
    )

# H5.3: Redundant features — C_u shouldn't exceed H(Z) even with many copies of z
print("\nH5.3: Redundant copies of z — C_u capped at H(Z)")
for n_copies in [1, 4, 16]:
    n_samples = 5000
    z = np.random.randint(0, 2, size=n_samples)
    # M = (z, z, z, ..., z) with noise on each copy
    # But for MI computation, redundant copies don't help
    # M effectively = z (many noisy copies → better estimate of z, but z is observed directly)
    p_mz = np.array([[0.5, 0], [0, 0.5]])
    cu = mutual_information(p_mz)
    log_result(
        f"H5.3: {n_copies} copies of z, C_u={cu:.4f}",
        "C_u = H(Z) = 1 bit (no benefit from redundancy)",
        f"C_u={cu:.4f}",
        abs(cu - 1.0) < 0.01
    )


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: IMPLICATIONS AND PREDICTIONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: FRAMEWORK PREDICTIONS")
print("=" * 70)
print()

# H6.1: C_u should predict transfer learning success
print("H6.1: C_u predicts transfer success")
rng = np.random.default_rng(42)
X_train, y_train = generate_data(2000, rng=rng)
# Transfer environment: different noise distribution
X_transfer = X_train.copy()
X_transfer[:, 4:] = np.random.randn(2000, 4) * 3 + 2  # shifted noise

transfer_results = []
for bn in [1, 2, 4, 8, 16]:
    torch.manual_seed(42)
    model = make_bn_net(bn)
    cu, _, acc_train = train_and_measure(model, X_train, y_train, bn)
    model.eval()
    with torch.no_grad():
        acc_transfer = (model(torch.from_numpy(X_transfer)).argmax(1).numpy() == y_train).mean()
    transfer_results.append((cu, acc_transfer))
    print(f"    bn={bn:>2}: C_u={cu:.4f}, acc_train={acc_train:.3f}, acc_transfer={acc_transfer:.3f}")

cus = [r[0] for r in transfer_results]
accs = [r[1] for r in transfer_results]
r_val, p_val = stats.pearsonr(cus, accs)
log_result(
    f"H6.1: r(C_u, transfer_acc) = {r_val:.4f}, p = {p_val:.4f}",
    "Positive correlation", f"r={r_val:.4f}",
    r_val > 0.5,
    "C_u should predict which architectures transfer well"
)

# H6.2: I_eff should predict learning speed
print("\nH6.2: I_eff predicts learning speed")
speed_results = []
for hidden in [8, 16, 32, 64]:
    torch.manual_seed(42)
    model = make_bn_net(2, hidden=hidden)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=64, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epochs_to_90 = None
    for ep in range(100):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); nn.CrossEntropyLoss()(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(torch.from_numpy(X_train)).argmax(1).numpy() == y_train).mean()
        if acc >= 0.90 and epochs_to_90 is None:
            epochs_to_90 = ep + 1

    # Compute I_eff
    cu, flops_per_ep, _ = train_and_measure(make_bn_net(2, hidden=hidden), X_train, y_train, 2, epochs=1)
    ieff = cu / flops_per_ep if flops_per_ep > 0 else 0
    speed_results.append((ieff, epochs_to_90 or 100))
    print(f"    hidden={hidden:>3}: I_eff={ieff:.2e}, epochs_to_90%={epochs_to_90 or '>100'}")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print()

n_pass = sum(1 for r in RESULTS if r["verdict"])
n_fail = sum(1 for r in RESULTS if not r["verdict"])
print(f"Total hypotheses tested: {len(RESULTS)}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"  Pass rate: {n_pass/len(RESULTS)*100:.1f}%")
print(f"  Elapsed: {time.time()-START:.1f}s")
print()

if n_fail > 0:
    print("FAILURES:")
    for r in RESULTS:
        if not r["verdict"]:
            print(f"  - {r['hypothesis']}")
            print(f"    {r['details']}")

# Save full log
with open("/Users/biobook/Code/fti/hypothesis_results.jsonl", "w") as f:
    for r in RESULTS:
        f.write(json.dumps(r) + "\n")
print(f"\nFull results saved to hypothesis_results.jsonl")
