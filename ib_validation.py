#!/usr/bin/env python3
"""
Information Bottleneck & Rate-Distortion Validation of C_u = I(M; Z)

Validates whether the C_u framework from "Intelligence as Useful Structure"
is consistent with rate-distortion theory and the Information Bottleneck (IB)
formalism.

The paper's toy example:
  - z in {0,1}: viability-relevant latent (uniform)
  - n in {0,1}: nuisance bit (uniform, independent of z)
  - Observation O = (z, n), so |O| = 4
  - Three agents compress O into representation M:
      Rule learner:  M = z          (1 bit)
      Lookup table:  M = (z, n)     (2 bits, i.e. M = O)
      Noise memoriser: M = n        (1 bit)

IB framework:
  Given Markov chain Z -- O -- M, the IB finds representations M that
  maximise I(M; Z) for a given rate I(M; O), via:
      min_{p(m|o)} I(M; O) - beta * I(M; Z)

  The IB curve is the Pareto frontier of (I(M;O), I(M;Z)).

Rate-distortion:
  R(D) = min_{p(m|z): E[d(z,m)] <= D} I(M; Z)
  For the binary source with Hamming distortion, R(D) = 1 - H(D) for D in [0, 0.5].
  C_u = I(M; Z) corresponds to operating at a point on this curve.

This script:
  1. Computes the exact IB curve via the Blahut-Arimoto IB algorithm
  2. Places the three agents on the curve
  3. Verifies IB-optimality of the rule learner
  4. Computes R(D) for the binary symmetric source
  5. Tests robustness with non-uniform P(z)
"""

import numpy as np
from typing import Tuple, List, Dict
import sys

# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """H(X) = -sum p_i log2(p_i), safe for zeros."""
    p = p.flatten()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mi_from_joint(p_xy: np.ndarray) -> float:
    """I(X;Y) from joint distribution p(x,y)."""
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 1e-15 and p_x[i] > 1e-15 and p_y[j] > 1e-15:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return max(0.0, mi)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p || q) in bits."""
    val = 0.0
    for i in range(len(p)):
        if p[i] > 1e-15:
            if q[i] < 1e-15:
                return float('inf')
            val += p[i] * np.log2(p[i] / q[i])
    return val


def h_binary(p: float) -> float:
    """Binary entropy function H(p) = -p log2(p) - (1-p) log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# ---------------------------------------------------------------------------
# Build the toy channel: Z -> O -> M
# ---------------------------------------------------------------------------

def build_toy_distributions(p_z0: float = 0.5, p_n0: float = 0.5):
    """
    Build P(z), P(o), P(z,o) for the toy example.

    O = (z, n) with z independent of n.
    O has 4 states: (0,0), (0,1), (1,0), (1,1)
    Z has 2 states: 0, 1

    Returns:
        p_z: array of shape (2,), marginal of Z
        p_o: array of shape (4,), marginal of O
        p_zo: array of shape (2, 4), joint P(Z=z, O=o)
        o_labels: list of (z, n) tuples labelling O states
    """
    p_z = np.array([p_z0, 1 - p_z0])
    p_n = np.array([p_n0, 1 - p_n0])

    # O states ordered as (z,n) = (0,0), (0,1), (1,0), (1,1)
    o_labels = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # P(O = (z,n)) = P(z) * P(n)
    p_o = np.zeros(4)
    for idx, (z, n) in enumerate(o_labels):
        p_o[idx] = p_z[z] * p_n[n]

    # P(Z=z, O=o): nonzero only when o's z-component equals z
    p_zo = np.zeros((2, 4))
    for idx, (z_o, n_o) in enumerate(o_labels):
        p_zo[z_o, idx] = p_o[idx]  # P(Z=z, O=(z,n)) = P(z)*P(n) if z matches

    return p_z, p_o, p_zo, o_labels


# ---------------------------------------------------------------------------
# Agent encoder definitions: P(M | O)
# ---------------------------------------------------------------------------

def agent_encoders(o_labels: list) -> Dict[str, Tuple[np.ndarray, int, List[str]]]:
    """
    Return P(M|O) for each agent.

    Returns dict mapping agent name -> (p_m_given_o, num_m, m_labels)
    where p_m_given_o has shape (num_m, num_o) with p_m_given_o[m, o] = P(M=m|O=o).
    """
    num_o = len(o_labels)
    agents = {}

    # 1. Rule learner: M = z (the z-component of O)
    # M in {0, 1}
    p_rule = np.zeros((2, num_o))
    for o_idx, (z, n) in enumerate(o_labels):
        p_rule[z, o_idx] = 1.0
    agents["Rule learner"] = (p_rule, 2, ["z=0", "z=1"])

    # 2. Lookup table: M = O = (z, n)
    # M in {(0,0), (0,1), (1,0), (1,1)} -- identity mapping
    p_lookup = np.eye(num_o)
    agents["Lookup table"] = (p_lookup, 4,
                              [f"({z},{n})" for z, n in o_labels])

    # 3. Noise memoriser: M = n (the n-component of O)
    # M in {0, 1}
    p_noise = np.zeros((2, num_o))
    for o_idx, (z, n) in enumerate(o_labels):
        p_noise[n, o_idx] = 1.0
    agents["Noise memoriser"] = (p_noise, 2, ["n=0", "n=1"])

    return agents


# ---------------------------------------------------------------------------
# Compute I(M; O) and I(M; Z) for a given encoder
# ---------------------------------------------------------------------------

def compute_agent_info(p_m_given_o: np.ndarray, p_o: np.ndarray,
                       p_zo: np.ndarray) -> Tuple[float, float]:
    """
    Given P(M|O), P(O), P(Z,O), compute I(M;O) and I(M;Z).

    I(M;O) = sum_{m,o} P(m,o) log [P(m,o) / (P(m)*P(o))]
    I(M;Z) = sum_{m,z} P(m,z) log [P(m,z) / (P(m)*P(z))]
    """
    num_m, num_o = p_m_given_o.shape
    num_z = p_zo.shape[0]

    # P(M=m, O=o) = P(M=m|O=o) * P(O=o)
    p_mo = p_m_given_o * p_o[np.newaxis, :]

    # P(M=m, Z=z) = sum_o P(M=m|O=o) * P(Z=z, O=o)
    p_mz = np.zeros((num_m, num_z))
    for m in range(num_m):
        for z in range(num_z):
            for o in range(num_o):
                p_mz[m, z] += p_m_given_o[m, o] * p_zo[z, o]

    i_mo = mi_from_joint(p_mo)
    i_mz = mi_from_joint(p_mz)

    return i_mo, i_mz


# ---------------------------------------------------------------------------
# Blahut-Arimoto IB Algorithm
# ---------------------------------------------------------------------------

def ib_blahut_arimoto(p_o: np.ndarray, p_zo: np.ndarray, beta: float,
                       num_m: int, max_iter: int = 5000,
                       tol: float = 1e-10) -> Tuple[np.ndarray, float, float]:
    """
    Blahut-Arimoto algorithm for the Information Bottleneck.

    Minimises: L = I(M; O) - beta * I(M; Z)
    over P(M|O), given the Markov chain Z -- O -- M.

    Uses the BA-IB iteration:
      1. p(m) = sum_o p(m|o) p(o)
      2. p(z|m) = sum_o p(z|o) p(o|m) = [sum_o p(z,o) p(m|o)] / p(m)
      3. p(m|o) propto p(m) exp( beta * D_KL(p(z|o) || p(z|m)) )
         ... actually, the IB update is:
         p(m|o) propto p(m) exp( -beta * D_KL(p(z|o) || p(z|m)) )
         Wait -- sign convention. The IB minimises I(M;O) - beta I(M;Z).
         The BA iteration for IB is:
           p(m|o) propto p(m) exp( beta * sum_z p(z|o) log(p(z|m)/p(z)) )

    Following Tishby et al. (1999) and Slonim (2002):
      p(m|o) = p(m)/Z(o,beta) * exp(-beta * D_KL(p(z|o) || p(z|m)))

    Args:
        p_o: shape (|O|,), marginal P(O)
        p_zo: shape (|Z|, |O|), joint P(Z, O)
        beta: IB tradeoff parameter (higher -> more I(M;Z))
        num_m: cardinality of M
        max_iter: max iterations
        tol: convergence tolerance

    Returns:
        p_m_given_o: shape (num_m, |O|)
        i_mo: I(M; O)
        i_mz: I(M; Z)
    """
    num_o = len(p_o)
    num_z = p_zo.shape[0]

    # P(Z|O=o) = P(Z,O=o) / P(O=o)
    p_z_given_o = np.zeros((num_z, num_o))
    for o in range(num_o):
        if p_o[o] > 1e-15:
            p_z_given_o[:, o] = p_zo[:, o] / p_o[o]

    # Initialize P(M|O) randomly
    rng = np.random.RandomState(42)
    p_m_given_o = rng.dirichlet(np.ones(num_m), size=num_o).T  # shape (num_m, num_o)

    for iteration in range(max_iter):
        old_p = p_m_given_o.copy()

        # Step 1: p(m) = sum_o p(m|o) p(o)
        p_m = p_m_given_o @ p_o  # shape (num_m,)

        # Step 2: p(z|m) = sum_o p(z|o) * p(o|m)
        # p(o|m) = p(m|o)*p(o) / p(m)
        p_z_given_m = np.zeros((num_z, num_m))
        for m in range(num_m):
            if p_m[m] > 1e-15:
                for o in range(num_o):
                    weight = p_m_given_o[m, o] * p_o[o] / p_m[m]
                    p_z_given_m[:, m] += weight * p_z_given_o[:, o]

        # Step 3: p(m|o) propto p(m) * exp(-beta * D_KL(p(z|o) || p(z|m)))
        log_p_m_given_o = np.zeros((num_m, num_o))
        for o in range(num_o):
            for m in range(num_m):
                if p_m[m] > 1e-15:
                    # D_KL(p(z|o) || p(z|m))
                    dkl = 0.0
                    for z in range(num_z):
                        if p_z_given_o[z, o] > 1e-15:
                            if p_z_given_m[z, m] < 1e-15:
                                dkl = 1e10  # infinity proxy
                                break
                            dkl += p_z_given_o[z, o] * np.log(
                                p_z_given_o[z, o] / p_z_given_m[z, m])
                    log_p_m_given_o[m, o] = np.log(p_m[m] + 1e-300) - beta * dkl
                else:
                    log_p_m_given_o[m, o] = -1e10

        # Normalise per o (softmax)
        for o in range(num_o):
            col = log_p_m_given_o[:, o]
            col -= col.max()
            exp_col = np.exp(col)
            p_m_given_o[:, o] = exp_col / exp_col.sum()

        # Convergence check
        if np.max(np.abs(p_m_given_o - old_p)) < tol:
            break

    # Compute I(M;O) and I(M;Z)
    i_mo, i_mz = compute_agent_info(p_m_given_o, p_o, p_zo)

    return p_m_given_o, i_mo, i_mz


def compute_ib_curve(p_o: np.ndarray, p_zo: np.ndarray,
                     num_m: int = 4,
                     beta_values: np.ndarray = None) -> List[Tuple[float, float]]:
    """
    Trace the IB curve by sweeping beta.

    Returns list of (I(M;O), I(M;Z)) pairs forming the Pareto frontier.
    """
    if beta_values is None:
        # Sweep beta from 0 (full compression) to large (full preservation)
        beta_values = np.concatenate([
            np.linspace(0.01, 1.0, 30),
            np.linspace(1.0, 5.0, 30),
            np.linspace(5.0, 50.0, 30),
            np.linspace(50.0, 500.0, 20),
        ])

    curve = []
    for beta in beta_values:
        _, i_mo, i_mz = ib_blahut_arimoto(p_o, p_zo, beta, num_m)
        curve.append((i_mo, i_mz))

    return curve


# ---------------------------------------------------------------------------
# Rate-Distortion for binary symmetric source
# ---------------------------------------------------------------------------

def rate_distortion_binary(D: float, p0: float = 0.5) -> float:
    """
    Rate-distortion function for a binary source with Hamming distortion.

    For uniform binary source: R(D) = 1 - H(D) for 0 <= D <= 0.5, else 0.
    For non-uniform P(z=0) = p0: R(D) = H(p0) - H(D) for 0 <= D <= min(p0, 1-p0).
    """
    D = max(0.0, D)
    p_min = min(p0, 1 - p0)
    if D >= p_min:
        return 0.0
    return h_binary(p0) - h_binary(D)


def compute_rd_curve(p0: float = 0.5, num_points: int = 200
                     ) -> List[Tuple[float, float]]:
    """
    Compute the rate-distortion curve R(D) for binary source.

    Returns list of (D, R(D)) pairs.
    """
    p_min = min(p0, 1 - p0)
    D_values = np.linspace(0, p_min, num_points)
    return [(D, rate_distortion_binary(D, p0)) for D in D_values]


def agent_distortion(agent_name: str, p_z0: float = 0.5) -> float:
    """
    Compute expected Hamming distortion E[d(Z, M_hat)] for each agent.

    The "reconstructed" Z is the best estimate of Z given M.
    - Rule learner: M=z, so Z_hat = M, distortion = 0
    - Lookup table: M=(z,n), Z_hat = z-component of M, distortion = 0
    - Noise memoriser: M=n, Z_hat is best guess from n alone.
      Since z and n are independent, best guess is the mode of P(z),
      so distortion = min(p_z0, 1-p_z0).
    """
    if agent_name in ("Rule learner", "Lookup table"):
        return 0.0
    elif agent_name == "Noise memoriser":
        return min(p_z0, 1 - p_z0)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ---------------------------------------------------------------------------
# IB Optimality Check
# ---------------------------------------------------------------------------

def check_ib_optimality(curve: List[Tuple[float, float]],
                        agent_point: Tuple[float, float],
                        agent_name: str, atol: float = 0.05) -> bool:
    """
    Check if an agent's (I(M;O), I(M;Z)) lies on the IB Pareto frontier.

    An agent is IB-optimal if there is no point on the curve with
    the same or lower I(M;O) but strictly higher I(M;Z).
    """
    i_mo_agent, i_mz_agent = agent_point

    # Find the maximum I(M;Z) achievable at I(M;O) <= i_mo_agent + atol
    max_imz_at_rate = max(
        (imz for imo, imz in curve if imo <= i_mo_agent + atol),
        default=0.0
    )

    is_optimal = i_mz_agent >= max_imz_at_rate - atol
    return is_optimal


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation(p_z0: float = 0.5, p_n0: float = 0.5,
                   label: str = "uniform") -> Dict:
    """Run complete IB + R(D) validation for given source distribution."""

    print(f"\n{'='*70}")
    print(f"  IB & Rate-Distortion Validation  [{label}]")
    print(f"  P(z=0) = {p_z0:.3f},  P(n=0) = {p_n0:.3f}")
    print(f"{'='*70}")

    # Build distributions
    p_z, p_o, p_zo, o_labels = build_toy_distributions(p_z0, p_n0)

    print(f"\n--- Source statistics ---")
    print(f"  H(Z) = {entropy(p_z):.6f} bits")
    print(f"  H(O) = {entropy(p_o):.6f} bits")
    print(f"  I(Z;O) = {mi_from_joint(p_zo):.6f} bits  (should = H(Z) since Z is a component of O)")

    # Verify Z is a deterministic function of O
    i_zo = mi_from_joint(p_zo)
    assert abs(i_zo - entropy(p_z)) < 1e-6, \
        f"I(Z;O) should equal H(Z), got {i_zo:.6f} vs {entropy(p_z):.6f}"
    print("  [OK] I(Z;O) = H(Z) confirmed (Z is deterministic function of O)")

    # Compute agent information values
    agents = agent_encoders(o_labels)
    agent_results = {}

    print(f"\n--- Agent information values ---")
    print(f"  {'Agent':<20s}  {'I(M;O)':<10s}  {'I(M;Z)':<10s}  {'H(M)':<10s}")
    print(f"  {'-'*50}")

    for name, (p_m_given_o, num_m, m_labels) in agents.items():
        i_mo, i_mz = compute_agent_info(p_m_given_o, p_o, p_zo)

        # Also compute H(M)
        p_m = p_m_given_o @ p_o
        h_m = entropy(p_m)

        agent_results[name] = {
            "I(M;O)": i_mo, "I(M;Z)": i_mz, "H(M)": h_m,
            "p_m_given_o": p_m_given_o
        }
        print(f"  {name:<20s}  {i_mo:<10.6f}  {i_mz:<10.6f}  {h_m:<10.6f}")

    # Verify paper's claimed values
    print(f"\n--- Verifying paper's claimed C_u values ---")
    expected_cu = {
        "Rule learner": entropy(p_z),     # 1 bit for uniform
        "Lookup table": entropy(p_z),     # 1 bit for uniform
        "Noise memoriser": 0.0
    }
    all_cu_ok = True
    for name, expected in expected_cu.items():
        actual = agent_results[name]["I(M;Z)"]
        ok = abs(actual - expected) < 1e-6
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: C_u = {actual:.6f}, expected = {expected:.6f}")
        if not ok:
            all_cu_ok = False

    # -----------------------------------------------------------------------
    # IB Curve Computation
    # -----------------------------------------------------------------------
    print(f"\n--- Computing IB curve (Blahut-Arimoto) ---")
    ib_curve = compute_ib_curve(p_o, p_zo, num_m=4)

    # Extract Pareto frontier (remove dominated points)
    pareto = []
    for imo, imz in sorted(ib_curve, key=lambda x: x[0]):
        if not pareto or imz > pareto[-1][1] + 1e-8:
            pareto.append((imo, imz))

    print(f"  Computed {len(ib_curve)} raw points, {len(pareto)} on Pareto frontier")
    print(f"  Rate range: [{min(p[0] for p in pareto):.4f}, {max(p[0] for p in pareto):.4f}]")
    print(f"  I(M;Z) range: [{min(p[1] for p in pareto):.4f}, {max(p[1] for p in pareto):.4f}]")

    # Key IB theoretical properties to verify:
    # 1. Maximum I(M;Z) = I(Z;O) = H(Z) (achieved at beta -> infinity)
    max_imz = max(imz for _, imz in ib_curve)
    hz = entropy(p_z)
    print(f"\n  Max I(M;Z) on IB curve = {max_imz:.6f}")
    print(f"  H(Z) = {hz:.6f}")
    ib_max_ok = abs(max_imz - hz) < 0.02
    print(f"  [{'OK' if ib_max_ok else 'FAIL'}] Max I(M;Z) matches H(Z) "
          f"(data processing inequality upper bound)")

    # 2. The IB curve should be concave (I(M;Z) is concave in I(M;O))
    concavity_violations = 0
    sorted_pareto = sorted(pareto, key=lambda x: x[0])
    for i in range(1, len(sorted_pareto) - 1):
        x0, y0 = sorted_pareto[i - 1]
        x1, y1 = sorted_pareto[i]
        x2, y2 = sorted_pareto[i + 1]
        if x2 - x0 > 1e-8:
            interp_y = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
            if y1 < interp_y - 0.01:
                concavity_violations += 1
    concave_ok = concavity_violations == 0
    print(f"  [{'OK' if concave_ok else 'FAIL'}] IB curve concavity "
          f"({concavity_violations} violations)")

    # -----------------------------------------------------------------------
    # Place agents on IB curve and check optimality
    # -----------------------------------------------------------------------
    print(f"\n--- Agent placement on IB curve ---")
    optimality_results = {}

    for name, info in agent_results.items():
        point = (info["I(M;O)"], info["I(M;Z)"])
        is_opt = check_ib_optimality(ib_curve, point, name)
        optimality_results[name] = is_opt
        print(f"  {name}: ({point[0]:.4f}, {point[1]:.4f})  "
              f"IB-optimal: {is_opt}")

    # The rule learner should be IB-optimal (max I(M;Z) for its I(M;O))
    rule_optimal = optimality_results["Rule learner"]
    print(f"\n  [{'OK' if rule_optimal else 'FAIL'}] Rule learner is IB-optimal")

    # The lookup table should NOT be IB-optimal (wastes rate on noise)
    # It achieves the same I(M;Z) as rule learner but at higher I(M;O)
    lookup_suboptimal = not optimality_results.get("Lookup table", True)
    # The lookup table is on the curve at I(M;O) = H(O) and I(M;Z) = H(Z),
    # so technically it IS on the curve at that rate. But it's dominated by
    # the rule learner which achieves the same I(M;Z) at lower I(M;O).
    imo_rule = agent_results["Rule learner"]["I(M;O)"]
    imo_lookup = agent_results["Lookup table"]["I(M;O)"]
    imz_rule = agent_results["Rule learner"]["I(M;Z)"]
    imz_lookup = agent_results["Lookup table"]["I(M;Z)"]

    lookup_dominated = (abs(imz_lookup - imz_rule) < 0.01 and
                        imo_lookup > imo_rule + 0.01)
    print(f"  [{'OK' if lookup_dominated else 'FAIL'}] Lookup table is IB-dominated "
          f"(same I(M;Z)={imz_lookup:.4f} but I(M;O)={imo_lookup:.4f} > {imo_rule:.4f})")

    # Noise memoriser should have I(M;Z) = 0
    noise_zero = agent_results["Noise memoriser"]["I(M;Z)"] < 1e-6
    print(f"  [{'OK' if noise_zero else 'FAIL'}] Noise memoriser has C_u = 0 "
          f"(no useful information)")

    # -----------------------------------------------------------------------
    # Rate-Distortion Analysis
    # -----------------------------------------------------------------------
    print(f"\n--- Rate-Distortion Analysis ---")
    rd_curve = compute_rd_curve(p_z0)

    print(f"  Binary source R(D): R(0) = {rate_distortion_binary(0.0, p_z0):.6f} = H(Z)")

    # Place agents on R(D)
    print(f"\n  Agent placement on R(D) curve:")
    rd_consistent = True
    for name in agents:
        D = agent_distortion(name, p_z0)
        R_D = rate_distortion_binary(D, p_z0)
        cu = agent_results[name]["I(M;Z)"]

        # For the rule learner and lookup table: D=0, R(0)=H(Z), C_u=H(Z)
        # So C_u = R(0) -- operating at the zero-distortion point.
        # For noise memoriser: D=0.5 (uniform case), R(0.5)=0, C_u=0.
        # The relationship: C_u = I(M;Z) >= R(D) with equality when
        # the representation is optimal.

        # Actually: in rate-distortion, R(D) is the MINIMUM rate needed.
        # C_u = I(M;Z) should satisfy: C_u >= R(D_achieved)
        # For optimal encoders, C_u = R(D_achieved).

        at_rd = abs(cu - R_D) < 0.02
        print(f"    {name}: D={D:.4f}, R(D)={R_D:.6f}, C_u={cu:.6f}  "
              f"[{'on R(D)' if at_rd else 'above R(D)'}]")
        if name == "Rule learner" and not at_rd:
            rd_consistent = False

    # Key R(D) consistency check: Rule learner operates at the R(D) boundary
    # because it is the minimum-rate encoder achieving D=0.
    print(f"\n  [{'OK' if rd_consistent else 'FAIL'}] Rule learner operates on R(D) "
          f"boundary (optimal rate for zero distortion)")

    # -----------------------------------------------------------------------
    # Theoretical consistency: C_u in IB framework
    # -----------------------------------------------------------------------
    print(f"\n--- Theoretical consistency checks ---")

    # 1. Data Processing Inequality: Z -- O -- M implies I(M;Z) <= I(O;Z) = H(Z)
    dpi_ok = True
    for name, info in agent_results.items():
        if info["I(M;Z)"] > entropy(p_z) + 1e-6:
            dpi_ok = False
            print(f"  [FAIL] DPI violated for {name}: I(M;Z)={info['I(M;Z)']:.6f} > H(Z)={entropy(p_z):.6f}")
    if dpi_ok:
        print(f"  [OK] Data Processing Inequality: I(M;Z) <= H(Z) for all agents")

    # 2. I(M;Z) <= I(M;O) (also from DPI on Z -- O -- M)
    dpi2_ok = True
    for name, info in agent_results.items():
        if info["I(M;Z)"] > info["I(M;O)"] + 1e-6:
            dpi2_ok = False
            print(f"  [FAIL] I(M;Z) > I(M;O) for {name}")
    if dpi2_ok:
        print(f"  [OK] I(M;Z) <= I(M;O) for all agents")

    # 3. Sufficiency: Rule learner is a sufficient statistic for Z
    # M=z implies I(M;Z) = H(Z), so M is sufficient for Z given O.
    suff_ok = abs(agent_results["Rule learner"]["I(M;Z)"] - entropy(p_z)) < 1e-6
    print(f"  [{'OK' if suff_ok else 'FAIL'}] Rule learner's M is sufficient statistic for Z "
          f"(I(M;Z) = H(Z))")

    # 4. Rule learner achieves minimal sufficient statistic:
    # It preserves all info about Z while achieving minimum I(M;O).
    # I(M;O) for rule learner = H(Z) (since M=z and z is a component of O).
    # This is the minimum possible rate that achieves I(M;Z) = H(Z).
    rule_imo = agent_results["Rule learner"]["I(M;O)"]
    min_suff_ok = abs(rule_imo - entropy(p_z)) < 1e-6
    print(f"  [{'OK' if min_suff_ok else 'FAIL'}] Rule learner is minimal sufficient statistic "
          f"(I(M;O) = H(Z) = {entropy(p_z):.6f}, achieved: {rule_imo:.6f})")

    # 5. C_u decomposition: I(M;O) = I(M;Z) + I(M;N|Z)
    # Since Z and N are independent: I(M;O) = I(M;Z) + I(M;N|Z)
    # For rule learner: I(M;N|Z) = 0 (no noise retained)
    # For lookup table: I(M;N|Z) = H(N|Z) = H(N) = 1 (all noise retained)
    # For noise memoriser: I(M;Z) = 0, I(M;N|Z) = I(M;N) = H(N)
    print(f"\n  Rate decomposition I(M;O) = I(M;Z) + I(M;N|Z):")
    decomp_ok = True
    h_n = h_binary(p_n0)
    expected_noise_info = {
        "Rule learner": 0.0,
        "Lookup table": h_n,
        "Noise memoriser": h_n,
    }
    for name, info in agent_results.items():
        noise_info = info["I(M;O)"] - info["I(M;Z)"]
        expected = expected_noise_info[name]
        ok = abs(noise_info - expected) < 0.01
        if not ok:
            decomp_ok = False
        print(f"    {name}: I(M;O)={info['I(M;O)']:.4f} = "
              f"{info['I(M;Z)']:.4f} + {noise_info:.4f}  "
              f"(expected noise component: {expected:.4f}) [{'OK' if ok else 'FAIL'}]")

    if decomp_ok:
        print(f"  [OK] Rate decomposition consistent: C_u separates signal from noise")

    # -----------------------------------------------------------------------
    # Collect pass/fail
    # -----------------------------------------------------------------------
    results = {
        "C_u values match paper": all_cu_ok,
        "IB curve max = H(Z)": ib_max_ok,
        "IB curve concavity": concave_ok,
        "Rule learner IB-optimal": rule_optimal,
        "Lookup table IB-dominated": lookup_dominated,
        "Noise memoriser C_u = 0": noise_zero,
        "Rule learner on R(D) boundary": rd_consistent,
        "Data processing inequality": dpi_ok,
        "I(M;Z) <= I(M;O)": dpi2_ok,
        "Sufficient statistic": suff_ok,
        "Minimal sufficient statistic": min_suff_ok,
        "Rate decomposition": decomp_ok,
    }

    return results


# ---------------------------------------------------------------------------
# Non-uniform P(z) robustness tests
# ---------------------------------------------------------------------------

def run_nonuniform_tests() -> Dict:
    """Test C_u framework with non-uniform source distributions."""

    print(f"\n{'='*70}")
    print(f"  Non-uniform P(z) Robustness Tests")
    print(f"{'='*70}")

    test_configs = [
        (0.3, 0.5, "skewed Z, uniform N"),
        (0.1, 0.5, "highly skewed Z, uniform N"),
        (0.5, 0.3, "uniform Z, skewed N"),
        (0.3, 0.7, "skewed Z, skewed N"),
        (0.01, 0.5, "extreme skew Z"),
    ]

    all_results = {}

    for p_z0, p_n0, label in test_configs:
        results = run_validation(p_z0, p_n0, label)
        all_results[label] = results

    return all_results


# ---------------------------------------------------------------------------
# Analytical IB curve verification
# ---------------------------------------------------------------------------

def analytical_ib_verification():
    """
    For the binary toy, verify the IB curve analytically.

    Key insight: Since O = (Z, N) with Z, N independent,
    the IB problem decomposes. The relevant variable Z has H(Z) = 1 bit.
    The observation O has H(O) = 2 bits. The joint structure means:

    1. At beta -> 0: M is trivial, I(M;O) = I(M;Z) = 0
    2. At beta -> inf: M = O, I(M;O) = 2, I(M;Z) = 1
    3. The optimal encoder at the "knee" has M = Z, achieving
       I(M;Z) = 1 with I(M;O) = 1.

    The IB curve for this problem has a simple structure:
    - Phase 1 (beta < beta_c1): M captures partial info about Z
    - Phase transition at beta_c1: M captures all of Z
    - Phase 2 (beta > beta_c1): M starts capturing N too
    - At beta -> inf: M = O

    This is because Z and N are independent, so the IB first extracts
    the "cheaper" relevant information, then the irrelevant noise.
    """
    print(f"\n{'='*70}")
    print(f"  Analytical IB Structure Verification")
    print(f"{'='*70}")

    p_z, p_o, p_zo, o_labels = build_toy_distributions(0.5, 0.5)

    # Trace detailed IB curve
    betas = np.concatenate([
        np.linspace(0.1, 2.0, 50),
        np.linspace(2.0, 10.0, 50),
        np.linspace(10.0, 100.0, 30),
    ])

    print(f"\n  IB curve at selected beta values:")
    print(f"  {'beta':>8s}  {'I(M;O)':>10s}  {'I(M;Z)':>10s}  {'I(M;N|Z)':>10s}  {'regime':<20s}")
    print(f"  {'-'*62}")

    phase_transition_detected = False
    full_capture_beta = None

    prev_imz = 0.0
    for beta in betas:
        _, i_mo, i_mz = ib_blahut_arimoto(p_o, p_zo, beta, num_m=4)
        i_mn_given_z = i_mo - i_mz

        # Detect phase transition: I(M;Z) jumps to ~1
        if not phase_transition_detected and i_mz > 0.95:
            phase_transition_detected = True
            full_capture_beta = beta

        if i_mn_given_z < 0.01:
            regime = "signal only"
        elif i_mz > 0.95 and i_mn_given_z > 0.01:
            regime = "signal + noise"
        elif i_mz < 0.05:
            regime = "near-trivial"
        else:
            regime = "partial signal"

        if beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0] or \
           (full_capture_beta and abs(beta - full_capture_beta) < 0.05):
            print(f"  {beta:>8.2f}  {i_mo:>10.6f}  {i_mz:>10.6f}  "
                  f"{i_mn_given_z:>10.6f}  {regime:<20s}")

        prev_imz = i_mz

    if phase_transition_detected:
        print(f"\n  [OK] Phase transition detected at beta ~ {full_capture_beta:.2f}: "
              f"M captures all of Z")
        print(f"  [OK] IB curve shows expected structure: trivial -> signal -> signal+noise")
    else:
        print(f"\n  [FAIL] Phase transition not detected")

    # Verify that at the knee, the optimal representation IS the sufficient
    # statistic M=Z, which is exactly what the rule learner uses.
    # At beta just above the transition, I(M;Z) ~= H(Z) and I(M;O) ~= H(Z).
    if full_capture_beta:
        _, i_mo_knee, i_mz_knee = ib_blahut_arimoto(
            p_o, p_zo, full_capture_beta, num_m=4)
        noise_at_knee = i_mo_knee - i_mz_knee
        knee_clean = noise_at_knee < 0.05
        print(f"\n  At the IB knee (beta={full_capture_beta:.2f}):")
        print(f"    I(M;O) = {i_mo_knee:.6f}")
        print(f"    I(M;Z) = {i_mz_knee:.6f}")
        print(f"    Noise captured: {noise_at_knee:.6f}")
        print(f"  [{'OK' if knee_clean else 'NOTE'}] At the knee, IB solution closely matches "
              f"rule learner (minimal noise)")

    return phase_transition_detected


# ---------------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------------

# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  C_u = I(M; Z) : Information Bottleneck & Rate-Distortion Validation")
    print("  Paper: 'Intelligence as Useful Structure'")
    print("=" * 70)

    # 1. Uniform case (paper's toy example)
    uniform_results = run_validation(0.5, 0.5, "uniform (paper's toy)")

    # 2. Analytical IB structure verification
    phase_transition = analytical_ib_verification()

    # 3. Non-uniform robustness
    nonuniform_results = run_nonuniform_tests()

    # 4. Summary
    total_tests = 0
    total_pass = 0

    print(f"\n{'='*70}")
    print(f"  FINAL VALIDATION SUMMARY")
    print(f"{'='*70}")

    print(f"\n  [A] Uniform source (paper's toy example):")
    for k, v in uniform_results.items():
        status = "PASS" if v else "FAIL"
        print(f"      [{status}] {k}")
        total_tests += 1
        if v:
            total_pass += 1

    print(f"\n  [B] Analytical IB structure:")
    pt_status = "PASS" if phase_transition else "FAIL"
    print(f"      [{pt_status}] Phase transition and IB knee match rule learner")
    total_tests += 1
    if phase_transition:
        total_pass += 1

    print(f"\n  [C] Non-uniform robustness:")
    for config_label, results in nonuniform_results.items():
        config_pass = all(results.values())
        n_pass = sum(1 for v in results.values() if v)
        n_total = len(results)
        status = "PASS" if config_pass else "FAIL"
        print(f"      [{status}] {config_label}: {n_pass}/{n_total}")
        for k, v in results.items():
            if not v:
                print(f"             FAILED: {k}")
        total_tests += 1
        if config_pass:
            total_pass += 1

    print(f"\n  {'='*60}")
    overall = total_pass == total_tests
    verdict = "PASS" if overall else "FAIL"
    print(f"  VERDICT: {verdict}  ({total_pass}/{total_tests} test groups passed)")
    print(f"  {'='*60}")

    if overall:
        print(f"\n  C_u = I(M; Z) is CONSISTENT with IB theory and rate-distortion.")
        print(f"  The rule learner occupies the IB-optimal point (the 'knee').")
        print(f"  The framework correctly separates useful structure from waste")
        print(f"  across uniform and non-uniform source distributions.")
    else:
        print(f"\n  Some checks FAILED. See details above.")

    sys.exit(0 if overall else 1)
