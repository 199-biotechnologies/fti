# Research Programme: C_u Neural Experiment

## Goal
Maximise the I_eff ratio (Bottleneck / Wide) while maintaining quality gates.
This demonstrates the paper's claim that compressed, relevance-filtered representations
are more efficient than brute-force memorisation.

## Current Baseline
- Ratio ≈ 40× (Bottleneck I_eff / Wide I_eff)
- C_u(BN) ≈ 0.96, C_u(Wide) ≈ 1.0, C_u(Noise) ≈ 0.003
- A(BN) ≈ 1.0, A(Wide) ≈ 0.96, A(Noise) ≈ 0.0

## Ideas to Try (ordered by priority)

### Hyperparameters
1. Reduce bottleneck epochs (it converges fast — fewer epochs = fewer FLOPs = higher I_eff)
2. Increase wide net epochs (if it needs more, its FLOPs go up = lower I_eff)
3. Larger batch size for bottleneck (fewer gradient steps = fewer backward passes)
4. Higher learning rate for bottleneck (faster convergence)
5. Lower learning rate for wide net (needs more steps)

### Architecture
6. Narrower bottleneck (1-dim instead of 2? Risky — may lose information)
7. Wider wide net (128 instead of 64 — makes it even more wasteful)
8. Add weight decay to wide net only (regularise it differently)
9. Try different noise dimensionality (more noise dims = harder for wide net)

### Environment
10. Increase noise dimensions (N_NOISE = 8 or 16 instead of 4)
11. Reduce relevant features (N_RELEVANT = 2 instead of 4)
12. Higher observation noise (NOISE_SCALE = 0.5+)

## Constraints
- C_u(BN) must stay > 0.8 (bottleneck must still learn z)
- C_u(Noise) must stay < 0.1 (noise agent must remain useless)
- A(BN) must stay > 0.9 (must generalise)
- Keep N_SEEDS ≥ 3 for statistical reliability
- Don't change the probe-based MI estimation method (that's a separate concern)

## Anti-Gaming Rules
- The ratio must improve because the bottleneck is genuinely more efficient,
  NOT because the wide net is sabotaged (e.g., zero learning rate)
- Wide net must still achieve C_u > 0.8 on its own
