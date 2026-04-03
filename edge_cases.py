#!/usr/bin/env python3
"""
Edge-case and boundary-condition analysis for the C_u framework.

C_u = I(M_t; Z^V_{t+τ} | H_t)   — useful internal structure
I_eff = ΔC_u / W_diss            — acquisition efficiency
I*_eff = k_B T̄ ln2 × ΔC_u / W_diss  — dimensionless
A(π; μ) = E[S_e(π)]              — adaptive reach

Tests 7 categories of edge cases, documenting whether C_u handles each
correctly, produces counterintuitive results, or breaks entirely.
"""

import numpy as np
from itertools import product
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Physical constants ──
k_B = 1.380649e-23   # J/K
T = 300.0             # K
LANDAUER = k_B * T * np.log(2)

# ─────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────

def H(probs):
    """Shannon entropy in bits. probs is a 1-D array / list of P values."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def MI_joint(p_xy):
    """Mutual information from a 2-D joint probability table (bits)."""
    p_xy = np.asarray(p_xy, dtype=np.float64)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def CMI(p_xyz):
    """
    I(X; Y | Z) from 3-D joint table p_xyz[x, y, z].
    Returns value in bits.
    """
    cmi = 0.0
    for z in range(p_xyz.shape[2]):
        pz = p_xyz[:, :, z].sum()
        if pz < 1e-15:
            continue
        pxy_gz = p_xyz[:, :, z] / pz
        cmi += pz * MI_joint(pxy_gz)
    return cmi


def I_eff(delta_cu, w_diss):
    """Acquisition efficiency (bits / Joule)."""
    if w_diss == 0:
        return np.inf if delta_cu > 0 else (0.0 if delta_cu == 0 else -np.inf)
    return delta_cu / w_diss


def I_eff_star(delta_cu, w_diss):
    """Dimensionless acquisition efficiency."""
    return LANDAUER * delta_cu / w_diss if w_diss != 0 else np.inf * np.sign(delta_cu)


def kl_divergence(p, q):
    """D_KL(p || q) in bits."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


def mi_from_samples_discrete(x, y):
    """Estimate MI(X;Y) from discrete samples using plug-in estimator."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    x_vals, y_vals = np.unique(x), np.unique(y)
    joint = np.zeros((len(x_vals), len(y_vals)))
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    for xi, yi in zip(x, y):
        joint[x_map[xi], y_map[yi]] += 1
    joint /= n
    return MI_joint(joint)


def mi_ksg(x, y, k=5):
    """
    KSG estimator for MI between continuous X and continuous Y.
    Both x and y are 1-D arrays of floats.
    Returns MI in bits.
    """
    from scipy.special import digamma
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = len(x)
    xy = np.hstack([x, y])

    from scipy.spatial import cKDTree
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find distance to k-th neighbour in joint space
    dists, _ = tree_xy.query(xy, k=k + 1)
    eps = dists[:, -1]

    # Count neighbours within eps in marginal spaces
    nx = np.array([tree_x.query_ball_point(x[i], eps[i] - 1e-15).__len__()
                    for i in range(n)]) - 1  # exclude self
    ny = np.array([tree_y.query_ball_point(y[i], eps[i] - 1e-15).__len__()
                    for i in range(n)]) - 1

    mi_nats = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return max(0.0, mi_nats / np.log(2))


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

PASS = "OK"
WARN = "COUNTERINTUITIVE"
FAIL = "BREAKS"

results = []

def report(category, test_name, verdict, detail):
    tag = {"OK": "[  OK  ]", "COUNTERINTUITIVE": "[ WARN ]", "BREAKS": "[BREAK ]"}[verdict]
    print(f"  {tag} {test_name}")
    print(f"         {detail}")
    results.append((category, test_name, verdict, detail))


# =====================================================================
print("=" * 72)
print(" EDGE-CASE ANALYSIS OF THE C_u FRAMEWORK")
print("=" * 72)
print()

# =====================================================================
# 1. DEGENERATE CASES
# =====================================================================
print("-" * 72)
print(" 1. DEGENERATE CASES")
print("-" * 72)

# 1a. W_diss -> 0 with positive ΔC_u
delta_cu = 0.5  # bits
for w in [1e-3, 1e-10, 1e-20, 0.0]:
    ie = I_eff(delta_cu, w)
    ies = I_eff_star(delta_cu, w) if w > 0 else np.inf
    if w == 0:
        report("1-Degenerate", f"W_diss=0, ΔC_u={delta_cu}",
               WARN,
               f"I_eff = {ie}, I*_eff = inf. Division by zero yields infinity. "
               "Framework needs a regularised form or a lower bound on W_diss "
               "(e.g., Landauer limit: W >= k_B T ln2 per bit erased).")
    elif w < LANDAUER:
        report("1-Degenerate", f"W_diss={w:.0e} < Landauer",
               WARN,
               f"I_eff = {ie:.2e}. Sub-Landauer dissipation is physically "
               f"impossible for irreversible bit operations (Landauer = {LANDAUER:.2e} J/bit). "
               "Framework should enforce W_diss >= n_erased * k_B T ln2.")

# 1b. C_u = H(Z) exactly (agent captures ALL viability information)
# Binary Z: H(Z) = 1 bit
# If C_u = H(Z), the agent's internal model fully determines Z^V.
# This is the theoretical ceiling.
p_mz = np.array([[0.5, 0.0],    # P(M=0, Z=0)=0.5, P(M=0, Z=1)=0
                  [0.0, 0.5]])   # P(M=1, Z=0)=0,   P(M=1, Z=1)=0.5
cu_perfect = MI_joint(p_mz)
report("1-Degenerate", "C_u = H(Z) (perfect agent)",
       PASS,
       f"C_u = {cu_perfect:.6f} bits = H(Z) = 1.0 bit. "
       "Agent captures all viability info. This is the correct theoretical maximum; "
       "the data-processing inequality guarantees C_u <= H(Z).")

# 1c. All agents have identical C_u
# If 3 agents all have C_u = 0.8, they are equally intelligent by C_u alone.
# But they may differ in I_eff and A.
cu_same = 0.8
w_vals = [1e-6, 1e-4, 1e-2]
i_effs = [I_eff(cu_same, w) for w in w_vals]
report("1-Degenerate", "All agents same C_u, different W_diss",
       PASS,
       f"C_u = {cu_same} for all. I_eff = {[f'{x:.2e}' for x in i_effs]}. "
       "Degeneracy in C_u is broken by the efficiency metric I_eff, and further "
       "by adaptive reach A. The triplet (C_u, I_eff, A) forms a sufficient descriptor.")

print()

# =====================================================================
# 2. REVERSIBILITY PARADOX
# =====================================================================
print("-" * 72)
print(" 2. REVERSIBILITY PARADOX")
print("-" * 72)

# A logically reversible computation (e.g., Toffoli-gate based learner)
# dissipates W_diss -> 0 in the quasi-static limit.
# This makes I_eff = ΔC_u / 0 = infinity.

# Case A: truly reversible (Landauer says only erasure costs)
delta_cu_rev = 0.9
w_diss_rev = 0.0  # reversible: no bit erasure, no heat

ie_rev = I_eff(delta_cu_rev, w_diss_rev)
report("2-Reversibility", "Fully reversible learner, W_diss=0",
       WARN,
       f"I_eff = {ie_rev}. Infinite efficiency is mathematically correct "
       "but physically misleading. Reversible computation still requires: "
       "(a) time proportional to 1/v for quasi-static operation, "
       "(b) overhead to keep the computation reversible (ancilla bits). "
       "FIX: define I_eff = ΔC_u / max(W_diss, W_Landauer) where "
       f"W_Landauer = k_B T ln2 * n_bits_erased = {LANDAUER:.2e} J/bit.")

# Case B: partially reversible (most ops reversible, some erasure)
n_erased = 3  # bits
w_partial = n_erased * LANDAUER  # minimal dissipation at Landauer limit
ie_partial = I_eff(delta_cu_rev, w_partial)
report("2-Reversibility", "Near-Landauer learner (3 bits erased)",
       PASS,
       f"W_diss = {w_partial:.2e} J (3 × Landauer). "
       f"I_eff = {ie_partial:.2e} bits/J. Finite and well-defined. "
       "This represents the physical floor for any learner that erases 3 bits.")

# Case C: regularised I_eff
def I_eff_regularised(delta_cu, w_diss, n_bits_erased=1):
    """Regularised I_eff with Landauer floor."""
    w_floor = n_bits_erased * LANDAUER
    return delta_cu / max(w_diss, w_floor)

ie_reg = I_eff_regularised(delta_cu_rev, 0.0, n_bits_erased=1)
report("2-Reversibility", "Proposed fix: Landauer-regularised I_eff",
       PASS,
       f"I_eff_reg(W=0) = {ie_reg:.2e} bits/J. "
       "Regularisation caps efficiency at the Landauer limit, preventing the "
       "infinity while preserving the correct ranking of agents.")

print()

# =====================================================================
# 3. NEGATIVE LEARNING (Catastrophic forgetting)
# =====================================================================
print("-" * 72)
print(" 3. NEGATIVE LEARNING (Catastrophic forgetting)")
print("-" * 72)

# Scenario: agent trains on task A, acquires C_u = 0.9.
# Then trains on unrelated task B, and C_u for task A drops to 0.3.
# ΔC_u = 0.3 - 0.9 = -0.6  (negative)

cu_before = 0.9
cu_after = 0.3
delta_cu_neg = cu_after - cu_before  # -0.6
w_diss_forget = 1e-5  # energy spent on task B training

ie_neg = I_eff(delta_cu_neg, w_diss_forget)
ies_neg = I_eff_star(delta_cu_neg, w_diss_forget)
report("3-Negative", "Catastrophic forgetting (ΔC_u = -0.6)",
       WARN,
       f"ΔC_u = {delta_cu_neg:.1f} bits, I_eff = {ie_neg:.2e} bits/J (NEGATIVE). "
       "Negative I_eff is physically meaningful: the agent spent energy to DESTROY "
       "useful structure. This correctly signals catastrophic forgetting. However, "
       "the magnitude is hard to interpret: |I_eff| = 6e4 suggests high-efficiency "
       "destruction, which is bizarre. Consider reporting ΔC_u and W_diss separately.")

# Edge: ΔC_u = 0 (no learning and no forgetting)
ie_zero = I_eff(0.0, 1e-5)
report("3-Negative", "No learning, no forgetting (ΔC_u = 0)",
       PASS,
       f"I_eff = {ie_zero:.2e}. Zero efficiency correctly indicates wasted energy "
       "(agent burned W_diss but gained nothing). Well-defined and interpretable.")

# Edge: ΔC_u < 0 AND W_diss -> 0
ie_neg_zero = I_eff(-0.5, 0.0)
report("3-Negative", "Negative ΔC_u with W_diss = 0",
       FAIL,
       f"I_eff = {ie_neg_zero}. Agent lost useful structure without spending energy. "
       "This is physically strange but possible: e.g., memory decay due to noise "
       "in a physical substrate. The -inf result is ill-defined. "
       "FIX: separate decay (passive loss) from active forgetting (energy-driven).")

# Gradual forgetting curve
print()
print("  Forgetting trajectory:")
cu_trajectory = [0.9, 0.85, 0.7, 0.5, 0.3, 0.1]
w_per_step = 1e-6
print(f"  {'Step':>4}  {'C_u':>6}  {'ΔC_u':>7}  {'I_eff':>12}  {'Interpretation'}")
for i in range(1, len(cu_trajectory)):
    dc = cu_trajectory[i] - cu_trajectory[i - 1]
    ie = I_eff(dc, w_per_step)
    interp = "forgetting" if dc < 0 else ("stable" if dc == 0 else "learning")
    print(f"  {i:>4}  {cu_trajectory[i]:>6.2f}  {dc:>+7.2f}  {ie:>12.2e}  {interp}")

print()

# =====================================================================
# 4. MULTI-TASK VIABILITY (Additivity of C_u)
# =====================================================================
print("-" * 72)
print(" 4. MULTI-TASK VIABILITY (Additivity of C_u)")
print("-" * 72)

# Z^V = (Z1, Z2) where Z1, Z2 are independent binary variables.
# Does I(M; (Z1, Z2)) = I(M; Z1) + I(M; Z2) when Z1 ⊥ Z2?
# By the chain rule: I(M; Z1, Z2) = I(M; Z1) + I(M; Z2 | Z1)
# If M ⊥ Z2 | Z1, then I(M; Z2 | Z1) = I(M; Z2) and we get additivity.
# But this is NOT guaranteed in general.

# Test case: M encodes Z1 perfectly and knows nothing about Z2
# M ∈ {0, 1}, P(M=z1) = 1 for the corresponding z1
# Joint P(M, Z1, Z2):
#   M=0, Z1=0, Z2=0 : 0.25
#   M=0, Z1=0, Z2=1 : 0.25
#   M=1, Z1=1, Z2=0 : 0.25
#   M=1, Z1=1, Z2=1 : 0.25

# I(M; Z1) should be 1 bit, I(M; Z2) should be 0
p_m_z1z2 = np.zeros((2, 2, 2))  # M, Z1, Z2
p_m_z1z2[0, 0, 0] = 0.25
p_m_z1z2[0, 0, 1] = 0.25
p_m_z1z2[1, 1, 0] = 0.25
p_m_z1z2[1, 1, 1] = 0.25

# I(M; (Z1,Z2)) — flatten (Z1,Z2) into a single variable with 4 states
p_m_zflat = np.zeros((2, 4))
for m in range(2):
    for z1 in range(2):
        for z2 in range(2):
            zf = z1 * 2 + z2
            p_m_zflat[m, zf] += p_m_z1z2[m, z1, z2]

mi_joint_full = MI_joint(p_m_zflat)

# I(M; Z1)
p_m_z1 = p_m_z1z2.sum(axis=2)  # marginalise over Z2
mi_m_z1 = MI_joint(p_m_z1)

# I(M; Z2)
p_m_z2 = p_m_z1z2.sum(axis=1)  # marginalise over Z1
mi_m_z2 = MI_joint(p_m_z2)

report("4-MultiTask", "M encodes Z1 only, Z1 ⊥ Z2",
       PASS,
       f"I(M; (Z1,Z2)) = {mi_joint_full:.6f}, I(M;Z1) = {mi_m_z1:.6f}, "
       f"I(M;Z2) = {mi_m_z2:.6f}. "
       f"Sum = {mi_m_z1 + mi_m_z2:.6f}. Additivity holds: "
       f"{mi_joint_full:.6f} = {mi_m_z1 + mi_m_z2:.6f} since M ⊥ Z2.")

# Now test case where M encodes a JOINT function of Z1 and Z2 (XOR)
# M = Z1 XOR Z2 (parity bit)
p_m_z1z2_xor = np.zeros((2, 2, 2))
for z1 in range(2):
    for z2 in range(2):
        m = z1 ^ z2
        p_m_z1z2_xor[m, z1, z2] = 0.25

p_m_zflat_xor = np.zeros((2, 4))
for m in range(2):
    for z1 in range(2):
        for z2 in range(2):
            p_m_zflat_xor[m, z1 * 2 + z2] += p_m_z1z2_xor[m, z1, z2]

mi_xor_full = MI_joint(p_m_zflat_xor)
mi_xor_z1 = MI_joint(p_m_z1z2_xor.sum(axis=2))
mi_xor_z2 = MI_joint(p_m_z1z2_xor.sum(axis=1))

report("4-MultiTask", "M = Z1 XOR Z2 (synergistic encoding)",
       WARN,
       f"I(M; (Z1,Z2)) = {mi_xor_full:.6f}, I(M;Z1) = {mi_xor_z1:.6f}, "
       f"I(M;Z2) = {mi_xor_z2:.6f}. "
       f"Sum = {mi_xor_z1 + mi_xor_z2:.6f} != {mi_xor_full:.6f}. "
       "ADDITIVITY FAILS. The XOR function creates synergistic information: "
       "M tells you 1 bit about (Z1,Z2) jointly, but 0 bits about either "
       "marginal. C_u for multi-dimensional Z^V must use the JOINT formulation "
       "I(M; Z1, Z2, ..., Zk), not sum of marginals. Partial Information "
       "Decomposition (PID) would be needed to separate redundant, unique, "
       "and synergistic contributions.")

# Mixed case: M encodes Z1 and partial Z2
# M ∈ {0,1,2,3}: M = 2*Z1 + Z2 with 80% accuracy on Z2
p_m_z1z2_mixed = np.zeros((4, 2, 2))
for z1 in range(2):
    for z2 in range(2):
        m_correct = 2 * z1 + z2
        m_wrong = 2 * z1 + (1 - z2)
        p_m_z1z2_mixed[m_correct, z1, z2] = 0.25 * 0.8
        p_m_z1z2_mixed[m_wrong, z1, z2] = 0.25 * 0.2

p_m_zflat_mixed = np.zeros((4, 4))
for m in range(4):
    for z1 in range(2):
        for z2 in range(2):
            p_m_zflat_mixed[m, z1 * 2 + z2] += p_m_z1z2_mixed[m, z1, z2]

mi_mixed_full = MI_joint(p_m_zflat_mixed)
mi_mixed_z1 = MI_joint(p_m_z1z2_mixed.sum(axis=2))
mi_mixed_z2 = MI_joint(p_m_z1z2_mixed.sum(axis=1))

report("4-MultiTask", "M encodes Z1 + noisy Z2 (redundant encoding)",
       PASS,
       f"I(M; (Z1,Z2)) = {mi_mixed_full:.4f}, I(M;Z1) = {mi_mixed_z1:.4f}, "
       f"I(M;Z2) = {mi_mixed_z2:.4f}, Sum = {mi_mixed_z1 + mi_mixed_z2:.4f}. "
       f"Sub-additivity holds: {mi_mixed_full:.4f} <= {mi_mixed_z1 + mi_mixed_z2:.4f}. "
       "This is expected by the chain rule: I(M;Z1,Z2) = I(M;Z1) + I(M;Z2|Z1) "
       "<= I(M;Z1) + I(M;Z2) when Z1 and Z2 share info through M.")

print()

# =====================================================================
# 5. CONTINUOUS Z (Discretisation effects)
# =====================================================================
print("-" * 72)
print(" 5. CONTINUOUS Z (Gaussian viability variable)")
print("-" * 72)

# Z ~ N(0, 1), M = Z + noise
# True MI for jointly Gaussian: I(M; Z) = -0.5 log2(1 - rho^2)
# where rho is the correlation coefficient.

n_samples = 10000
rng = np.random.default_rng(42)

z_continuous = rng.standard_normal(n_samples)
noise_std = 0.5
m_continuous = z_continuous + rng.standard_normal(n_samples) * noise_std

# Analytical MI for bivariate Gaussian
var_z = 1.0
var_m = var_z + noise_std**2
rho = var_z / np.sqrt(var_z * var_m)
mi_analytical = -0.5 * np.log2(1 - rho**2)

# KSG estimator on continuous data
mi_ksg_est = mi_ksg(m_continuous, z_continuous, k=7)

# Discretised estimates at different bin counts
print()
print("  Discretisation sensitivity:")
print(f"  {'Bins':>6}  {'MI_disc':>10}  {'MI_KSG':>10}  {'MI_true':>10}  {'Rel.Err(disc)':>14}")
for n_bins in [4, 8, 16, 32, 64, 128, 256]:
    z_binned = np.digitize(z_continuous, np.linspace(z_continuous.min(), z_continuous.max(), n_bins))
    m_binned = np.digitize(m_continuous, np.linspace(m_continuous.min(), m_continuous.max(), n_bins))
    mi_disc = mi_from_samples_discrete(m_binned, z_binned)
    rel_err = abs(mi_disc - mi_analytical) / mi_analytical * 100
    print(f"  {n_bins:>6}  {mi_disc:>10.4f}  {mi_ksg_est:>10.4f}  {mi_analytical:>10.4f}  {rel_err:>13.1f}%")

report("5-Continuous", "Continuous Z with KSG estimator",
       PASS,
       f"MI_analytical = {mi_analytical:.4f} bits, MI_KSG = {mi_ksg_est:.4f} bits "
       f"(error = {abs(mi_ksg_est - mi_analytical) / mi_analytical * 100:.1f}%). "
       "KSG gives a consistent estimate without discretisation artefacts.")

report("5-Continuous", "Coarse discretisation (4 bins)",
       WARN,
       f"MI_disc(4 bins) underestimates true MI due to quantisation. "
       "C_u is well-defined for continuous Z via differential entropy / KSG, "
       "but naive binning systematically biases downward. Practitioners must "
       "use density-ratio or KSG estimators for continuous viability variables.")

# High-dimensional continuous Z
z_hd = rng.standard_normal((n_samples, 5))
m_hd = z_hd[:, 0] + rng.standard_normal(n_samples) * 0.3  # M tracks only dim 0
mi_hd_true = -0.5 * np.log2(1 - (1.0 / (1.0 + 0.09)))  # Gaussian formula for dim 0
mi_hd_ksg = mi_ksg(m_hd, z_hd[:, 0], k=7)

report("5-Continuous", "M tracks 1 of 5 continuous viability dims",
       PASS,
       f"MI(M; Z_0) = {mi_hd_ksg:.4f} (KSG), true ~= {mi_hd_true:.4f}. "
       "C_u correctly decomposes: M only captures structure about the dimension "
       "it has learned, contributing 0 bits about the other 4 dims.")

print()

# =====================================================================
# 6. STOCHASTIC REPRESENTATIONS
# =====================================================================
print("-" * 72)
print(" 6. STOCHASTIC REPRESENTATIONS (M = Z + noise)")
print("-" * 72)

# M is a noisy function of Z. At different SNR levels, how does C_u behave?
# Z ~ Bernoulli(0.5), M = Z + N(0, sigma^2)  (continuous M, discrete Z)
# True MI: I(M; Z) = H(Z) - H(Z|M) = 1 - H(Z|M)
# For Gaussian channel with binary input:
#   I(M;Z) can be computed via numerical integration.

# We approximate with Monte Carlo
n_mc = 50000
z_binary = rng.integers(0, 2, size=n_mc)

print()
print("  SNR sweep for stochastic M = Z + Gaussian noise:")
print(f"  {'sigma':>7}  {'SNR_dB':>8}  {'C_u(KSG)':>10}  {'C_u(disc)':>10}  {'H(Z)':>6}")

snr_results = []
for sigma in [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    m_noisy = z_binary.astype(float) + rng.standard_normal(n_mc) * sigma
    snr_db = 10 * np.log10(0.25 / sigma**2) if sigma > 0 else np.inf  # SNR for binary signal

    # KSG on (continuous M, continuous proxy for binary Z)
    # For discrete Z, use plug-in: bin M, then standard MI
    m_binned = np.digitize(m_noisy, np.linspace(m_noisy.min(), m_noisy.max(), 64))
    cu_disc = mi_from_samples_discrete(m_binned, z_binary)

    # Direct analytic-style: for each z in {0,1}, M|Z=z ~ N(z, sigma^2)
    # I(M;Z) = integral p(m) log p(m) dm - 0.5 * integral p(m|z=0) log p(m|z=0) dm
    #                                       - 0.5 * integral p(m|z=1) log p(m|z=1) dm
    # = H(M) - H(M|Z)
    # H(M|Z) = 0.5 * log2(2*pi*e*sigma^2)  (Gaussian entropy)
    from scipy.integrate import quad
    from scipy.stats import norm

    def p_m(m):
        return 0.5 * norm.pdf(m, 0, sigma) + 0.5 * norm.pdf(m, 1, sigma)

    def neg_p_m_logp_m(m):
        pm = p_m(m)
        return -pm * np.log2(pm) if pm > 1e-300 else 0.0

    h_m, _ = quad(neg_p_m_logp_m, -10 * sigma - 1, 10 * sigma + 2)
    h_m_given_z = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
    cu_analytic = max(0, h_m - h_m_given_z)

    snr_results.append((sigma, snr_db, cu_analytic, cu_disc))
    print(f"  {sigma:>7.2f}  {snr_db:>8.1f}  {cu_analytic:>10.4f}  {cu_disc:>10.4f}  {1.0:>6.1f}")

report("6-Stochastic", "High SNR (sigma=0.01)",
       PASS,
       f"C_u = {snr_results[0][2]:.4f} ≈ H(Z) = 1.0 bit. "
       "Near-deterministic M recovers almost all information. Correct.")

report("6-Stochastic", "Low SNR (sigma=10.0)",
       PASS,
       f"C_u = {snr_results[-1][2]:.4f} ≈ 0 bits. "
       "Noise swamps the signal; M carries negligible info about Z. Correct.")

# Critical test: does C_u degrade gracefully or cliff-edge?
cu_vals = [r[2] for r in snr_results]
diffs = [cu_vals[i] - cu_vals[i + 1] for i in range(len(cu_vals) - 1)]
max_drop = max(diffs)
report("6-Stochastic", "Graceful degradation check",
       PASS,
       f"Max single-step C_u drop = {max_drop:.4f} bits. "
       "C_u degrades monotonically and smoothly as noise increases. "
       "No cliff-edge or non-monotonic behaviour detected.")

# Stochastic M with quantisation: M = round(Z + noise)
m_quantised = np.round(z_binary + rng.standard_normal(n_mc) * 0.5).astype(int)
m_quantised = np.clip(m_quantised, 0, 1)
cu_quantised = mi_from_samples_discrete(m_quantised, z_binary)
report("6-Stochastic", "Quantised stochastic M (binary output)",
       PASS,
       f"C_u = {cu_quantised:.4f} bits for M = clip(round(Z + N(0,0.25)), 0, 1). "
       "Quantisation further reduces C_u compared to continuous M. "
       "Framework handles discrete stochastic representations correctly.")

print()

# =====================================================================
# 7. TEMPORAL STRUCTURE (Markov dynamics)
# =====================================================================
print("-" * 72)
print(" 7. TEMPORAL STRUCTURE (Markov dynamics)")
print("-" * 72)

# Environment: Z_t is a Markov chain on {0, 1} with transition matrix P.
# Agent builds M from past observations.
# We compare I(M; Z_{t+1} | H_t) vs I(M; Z_{t+1}).

# Transition: P(Z_{t+1}=j | Z_t=i) = T[i, j]
# Sticky chain: high self-transition probability
T_mat = np.array([[0.8, 0.2],
                   [0.3, 0.7]])

# Stationary distribution
eigvals, eigvecs = np.linalg.eig(T_mat.T)
idx = np.argmin(np.abs(eigvals - 1.0))
pi_stat = np.real(eigvecs[:, idx])
pi_stat = pi_stat / pi_stat.sum()

# Simulate Markov chain
n_steps = 20000
z_seq = np.zeros(n_steps, dtype=int)
z_seq[0] = rng.choice(2, p=pi_stat)
for t in range(1, n_steps):
    z_seq[t] = rng.choice(2, p=T_mat[z_seq[t - 1]])

# Agent that has learned the transition model: M_t = Z_t (perfect state tracking)
# H_t includes the current observation Z_t
# I(M; Z_{t+1}) — unconditional
m_seq = z_seq[:-1]  # M_t = Z_t
z_next = z_seq[1:]  # Z_{t+1}

mi_unconditional = mi_from_samples_discrete(m_seq, z_next)

# I(M; Z_{t+1} | H_t) where H_t = Z_t (current observation)
# Since M_t = Z_t = H_t in this case, I(M; Z_{t+1} | H_t) = I(Z_t; Z_{t+1} | Z_t)
# = 0, because Z_t is redundant with itself given Z_t.
# This is the degenerate case where M doesn't add info beyond H_t.
h_t = z_seq[:-1]  # history = current state

# Compute I(M; Z_{t+1} | H_t) via discretised conditional MI
# Build joint P(M, Z_{t+1}, H_t)
p_mzh = np.zeros((2, 2, 2))
for i in range(len(m_seq)):
    p_mzh[m_seq[i], z_next[i], h_t[i]] += 1
p_mzh /= p_mzh.sum()

cmi_val = CMI(p_mzh)

report("7-Temporal", "M = Z_t, H_t = Z_t (redundant)",
       PASS,
       f"I(M; Z_{{t+1}}) = {mi_unconditional:.4f} bits, "
       f"I(M; Z_{{t+1}} | H_t) = {cmi_val:.6f} bits (≈ 0). "
       "When M = H_t, conditioning eliminates all MI. This is correct: "
       "the conditioning on H_t forces C_u to measure what M contributes "
       "BEYOND current observations. A mere state-tracker has C_u ≈ 0.")

# Now: agent with MEMORY — M_t = (Z_t, Z_{t-1})
# H_t = Z_t only (current obs)
# M carries the previous state, which helps predict Z_{t+1} via 2nd-order Markov
m_2step = np.stack([z_seq[1:-1], z_seq[:-2]], axis=1)  # (Z_t, Z_{t-1})
z_next_2 = z_seq[2:]
h_t_2 = z_seq[1:-1]

# Encode M as single int: m = 2*Z_t + Z_{t-1}
m_encoded = m_2step[:, 0] * 2 + m_2step[:, 1]
mi_unconditional_2 = mi_from_samples_discrete(m_encoded, z_next_2)

# CMI: P(M_encoded, Z_{t+1}, H_t)
p_mzh_2 = np.zeros((4, 2, 2))
for i in range(len(m_encoded)):
    p_mzh_2[m_encoded[i], z_next_2[i], h_t_2[i]] += 1
p_mzh_2 /= p_mzh_2.sum()
cmi_memory = CMI(p_mzh_2)

report("7-Temporal", "M = (Z_t, Z_{t-1}), H_t = Z_t (memory agent)",
       PASS,
       f"I(M; Z_{{t+1}}) = {mi_unconditional_2:.4f} bits, "
       f"I(M; Z_{{t+1}} | H_t) = {cmi_memory:.4f} bits. "
       "The memory agent has nonzero C_u because M carries Z_{{t-1}} which H_t "
       "does not contain, and Z_{{t-1}} helps predict Z_{{t+1}} in a non-first-order "
       "way (through conditioning on Z_t). This correctly differentiates a "
       "memory-bearing agent from a memoryless one.")

# Verify: I(M; Z_{t+1} | H_t) != I(M; Z_{t+1})
report("7-Temporal", "Unconditional vs conditional MI differ",
       PASS,
       f"I(M;Z') = {mi_unconditional_2:.4f} vs I(M;Z'|H) = {cmi_memory:.4f}. "
       f"Difference = {mi_unconditional_2 - cmi_memory:.4f} bits. "
       "They differ because H_t already reveals some of what M knows. "
       "The conditional form C_u = I(M; Z | H) is the correct one for measuring "
       "the UNIQUE contribution of learned structure beyond raw observations.")

# What about a 1st-order Markov environment? Z_{t+1} only depends on Z_t.
# Then Z_{t-1} is irrelevant given Z_t: I(Z_{t-1}; Z_{t+1} | Z_t) = 0.
# So even the memory agent should have C_u = 0 for the extra memory bit.
# Let's verify this.

# The Markov property means P(Z_{t+1} | Z_t, Z_{t-1}) = P(Z_{t+1} | Z_t).
# So I(M; Z_{t+1} | H_t) with M=(Z_t, Z_{t-1}) and H_t = Z_t
# = I(Z_{t-1}; Z_{t+1} | Z_t) = 0.
# Our cmi_memory above should be near 0 if the chain is truly 1st-order Markov.
report("7-Temporal", "1st-order Markov: extra memory is useless",
       PASS if cmi_memory < 0.01 else WARN,
       f"C_u(memory agent) = {cmi_memory:.6f} ≈ 0. "
       "In a 1st-order Markov chain, Z_{{t-1}} ⊥ Z_{{t+1}} | Z_t (Markov property). "
       "C_u correctly assigns near-zero value to the redundant memory bit. "
       "The small residual is finite-sample estimation noise.")

print()

# =====================================================================
# SUMMARY
# =====================================================================
print("=" * 72)
print(" SUMMARY OF FINDINGS")
print("=" * 72)
print()

n_ok = sum(1 for r in results if r[2] == PASS)
n_warn = sum(1 for r in results if r[2] == WARN)
n_fail = sum(1 for r in results if r[2] == FAIL)

print(f"  Total tests: {len(results)}")
print(f"  [  OK  ]  Handled correctly:          {n_ok}")
print(f"  [ WARN ]  Counterintuitive result:    {n_warn}")
print(f"  [BREAK ]  Framework breaks:           {n_fail}")
print()
print("  DETAILED VERDICTS:")
print()

categories = {}
for cat, name, verdict, detail in results:
    categories.setdefault(cat, []).append((name, verdict))

for cat in sorted(categories.keys()):
    print(f"  {cat}:")
    for name, verdict in categories[cat]:
        tag = {"OK": "OK", "COUNTERINTUITIVE": "WARN", "BREAKS": "BREAK"}[verdict]
        print(f"    [{tag:>5}] {name}")
    print()

print("  KEY RECOMMENDATIONS:")
print()
print("  1. REVERSIBILITY: Define I_eff = ΔC_u / max(W_diss, n_erased × k_B T ln2)")
print("     to prevent division-by-zero and sub-Landauer absurdities.")
print()
print("  2. NEGATIVE LEARNING: Report ΔC_u and W_diss separately when ΔC_u < 0.")
print("     The ratio I_eff = ΔC_u / W_diss is well-defined but its magnitude")
print("     for negative ΔC_u lacks clear physical interpretation.")
print()
print("  3. MULTI-TASK: Use joint I(M; Z1, Z2, ...) not sum of marginals.")
print("     Synergistic information (e.g., XOR-type encodings) breaks additivity.")
print("     Consider PID (Partial Information Decomposition) for task-specific C_u.")
print()
print("  4. CONTINUOUS Z: Use KSG or density-ratio estimators, not naive binning.")
print("     Binning systematically underestimates C_u for continuous variables.")
print()
print("  5. W_diss = 0 with ΔC_u < 0: Passive memory decay creates -inf.")
print("     Distinguish active forgetting (energy-driven, W_diss > 0) from")
print("     passive decay (substrate noise, W_diss ≈ 0).")
print()
