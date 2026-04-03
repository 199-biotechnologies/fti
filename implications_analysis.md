# Implications Analysis: What the C_u Framework Predicts About Real Systems

**Framework under analysis:** Djordjevic (2026), "Intelligence as Useful Structure: A Thermodynamic Perspective"

**Core quantities:**
- C_u = I(M_t; Z^V_{t+tau} | H_t) — useful internal structure (bits of viability-relevant information retained by the agent's representation)
- I_eff = Delta C_u / W_diss — acquisition efficiency (bits of useful structure gained per unit dissipated work)
- I*_eff = k_B T_bar ln2 * Delta C_u / W_diss — dimensionless, Landauer-normalised efficiency
- A(pi; mu) — adaptive reach (expected success rate across a distribution of environments)

---

## 1. Predictions for Machine Learning

### 1.1 Model Compression (Pruning, Distillation)

The framework makes a clear, testable prediction: model compression techniques that preserve task-relevant mutual information I(M; Z^V) while reducing total stored information H(M) should **improve I_eff without degrading C_u or A**. This follows directly from the toy example, where the rule learner and the lookup table share identical C_u = 1 bit, but the rule learner achieves higher I*_eff (0.0625 vs 0.0417) because it stores only the relevant bit.

**Specific predictions:**

- **Pruning** that removes parameters encoding nuisance correlations (analogous to the noise memoriser's n-bit) should leave C_u approximately unchanged while reducing W_diss. The framework predicts that structured pruning guided by task-relevance — not just magnitude — should yield better I_eff improvements. Lottery-ticket-style pruning, which identifies subnetworks that are sufficient for the task, is interpretable in C_u terms as finding the minimal representation that preserves I(M; Z^V).

- **Knowledge distillation** should be understood as transferring C_u from teacher to student while compressing H(M). If distillation succeeds, the student's C_u should approximate the teacher's, but with lower H(M) and lower W_diss during inference. The framework predicts that distillation will fail specifically when the student architecture lacks the representational capacity to encode the relevant I(M; Z^V) — not merely when parameter counts differ.

- **Quantisation** (reducing bit-width) is predicted to be benign as long as the quantised representation preserves the mutual information with Z^V. Below some critical bit-width, C_u should drop sharply, corresponding to the well-documented accuracy cliffs in aggressive quantisation.

**Limitation:** The framework does not by itself predict *which* parameters encode nuisance information. That requires a task-specific estimator of Z^V.

### 1.2 Smaller Fine-Tuned Models vs Large Pretrained Models

The framework predicts that fine-tuned models should have **higher I_eff for the target task** than large pretrained models, but the comparison requires care.

A large pretrained model has high total H(M) because it encodes broad predictive structure across many potential Z^V distributions (language, vision, reasoning). For any single task family, much of that structure is nuisance — analogous to the lookup table's extra bit. Fine-tuning discards or downweights irrelevant structure, concentrating the representation on the task-relevant Z^V.

**Prediction:** For a fixed target task, a smaller fine-tuned model should exhibit:
- Similar or modestly lower C_u relative to the pretrained model (it may lose some marginally relevant structure)
- Substantially lower W_diss (both in training and inference)
- Higher I_eff = Delta C_u / W_diss
- Potentially higher A on the target task distribution, lower A on unrelated distributions

This aligns with empirical findings that fine-tuned 7B models can outperform general-purpose 70B models on narrow domains. The C_u framework gives a thermodynamic explanation: the fine-tuned model's representation is better compressed relative to the task-specific Z^V.

**However:** The pretrained model's advantage is in A — adaptive reach across diverse task families. A single I_eff number does not capture this. The framework correctly insists on evaluating both axes.

### 1.3 Mixture-of-Experts (MoE) Architectures

This is where the framework generates a genuinely interesting prediction. In MoE architectures (e.g., Switch Transformers, Mixtral), only a subset of expert modules are activated for any given input. The framework's treatment of W_diss as *physically dissipated work during processing* implies the following:

**Prediction:** Inactive experts should **not** count toward W_diss for a given input, because they perform no irreversible computation on that input. W_diss should be computed over the activated pathway only. This means MoE architectures can, in principle, maintain high C_u (because the full model stores broad useful structure) while achieving lower per-input W_diss than dense models of equivalent total parameter count.

**Implication:** MoE should have higher I_eff than equivalently parameterised dense models, specifically because the routing mechanism acts as an implicit information bottleneck — selecting which expert's structure is relevant for the current input. The gating function in MoE is, in C_u terms, a learned selector for which subset of Z^V is relevant given the current observation.

**Testable nuance:** If MoE routing is poor (experts are activated randomly or redundantly), then the dissipated work is wasted on irrelevant computation, analogous to the noise memoriser. The framework predicts that MoE models with better-calibrated routing should have higher I_eff even at the same FLOPs budget.

**Open question:** Should the energy cost of training all experts count toward W_diss even if most are inactive at inference? The framework distinguishes learning-phase I_eff from deployed-state C_u. During training, all expert parameters are updated, so W_diss includes the full model. At inference (Delta C_u = 0), the relevant quantity is the stock C_u and behavioral A, not I_eff. This asymmetry is a feature, not a bug, of the framework.

### 1.4 LoRA vs Full Fine-Tuning

**Prediction:** LoRA (Low-Rank Adaptation) should have higher I_eff than full fine-tuning for task-specific adaptation.

**Mechanistic argument:**
- LoRA modifies a low-rank subspace of the weight matrices. If the task-relevant adaptation lies in a low-dimensional manifold of parameter space — which empirical results suggest it often does — then LoRA's Delta C_u should approximate full fine-tuning's Delta C_u.
- LoRA's W_diss is substantially lower because: (a) far fewer parameters are updated (fewer bit erasures during training), (b) gradient computation is cheaper (lower FLOPs), and (c) the frozen base model parameters contribute zero W_diss during adaptation.
- Therefore I_eff(LoRA) = Delta C_u / W_diss should be higher.

**When LoRA should fail in C_u terms:** When the task-relevant structure requires high-rank modifications to the representation — i.e., when the task's Z^V is not well-approximated by a low-rank perturbation of the pretrained model's representation. In such cases, Delta C_u(LoRA) << Delta C_u(full), and despite lower W_diss, the model may have lower or comparable I_eff.

**Empirical test:** Compare I_eff (estimated via probe-based MI and wall-clock energy) for LoRA vs full fine-tuning across tasks of varying "adaptation complexity." The framework predicts that LoRA's I_eff advantage should be largest for tasks where the pretrained model already encodes most of the relevant structure.

---

## 2. Predictions for Neuroscience

### 2.1 Sleep Consolidation

The C_u framework provides a natural account of sleep's function in terms of information filtering. The core claim — that intelligent systems should retain I(M; Z^V) while discarding nuisance structure — maps directly onto the complementary learning systems theory and the synaptic homeostasis hypothesis.

**Prediction:** During sleep consolidation, the brain should be performing something functionally equivalent to increasing I_eff by:
1. **Transferring viability-relevant structure** from hippocampal episodic traces (high H(M), uncompressed) to neocortical representations (lower H(M), compressed but preserving C_u). This corresponds to the transition from the lookup-table agent to the rule-learner agent.
2. **Discarding nuisance correlations** that were encoded during waking experience but do not contribute to I(M; Z^V). This is the synaptic downscaling component: reducing H(M) while preserving C_u.
3. **Reducing W_diss per unit computation** by consolidating representations into more energy-efficient cortical circuits. Sparse, compressed cortical representations require fewer synaptic operations (lower metabolic cost) than dense hippocampal engrams.

**Testable predictions:**
- Post-sleep representations should show **higher probe-estimated I(representation; task-relevant variable)** relative to total representational entropy, compared to pre-sleep representations. In other words, the ratio C_u / H(M) should increase.
- Sleep deprivation should impair this compression, leading to representations that retain more nuisance information (lower I_eff) even if raw task accuracy is temporarily preserved.
- The amount of slow-wave sleep (associated with hippocampal-cortical replay) should correlate with the magnitude of the C_u / H(M) compression ratio improvement.

**Limitation:** Measuring C_u in biological neural networks requires defining Z^V for the organism, which is non-trivial. Viability-relevant variables in freely behaving animals are not as cleanly specified as in the toy example.

### 2.2 Biological vs Artificial Efficiency

The framework offers a partial explanation for why biological neural networks are vastly more energy-efficient than artificial ones. A single synaptic event dissipates roughly 10^4 kT (approximately 10 fJ), while a single GPU floating-point operation dissipates approximately 10 pJ — about 10^6 times the Landauer limit. Biological synapses are roughly 100x closer to the Landauer floor per elementary operation.

**But the framework's explanation goes beyond raw energy per operation:**

The critical distinction is that biological networks appear to have much higher C_u / H(M) ratios — they store less total information but more of it is viability-relevant. The brain does not memorise all sensory input; it aggressively compresses toward representations that support prediction and action. This is exactly the information-bottleneck optimisation that C_u rewards.

**Specific predictions:**
- The brain's energy budget should be disproportionately allocated to regions that contribute most to C_u (prediction, control, decision-making) rather than raw sensory throughput. This is broadly consistent with the high metabolic cost of prefrontal and associative cortex relative to primary sensory cortex.
- Biological learning rules (e.g., spike-timing-dependent plasticity, neuromodulatory gating) should, on average, favour updates that increase I(M; Z^V) over those that increase H(M) without improving prediction or control. The framework interprets dopaminergic reward signals as providing the "Z^V filter" — marking which environmental variables are viability-relevant.

**Important caveat:** The framework does not derive biological efficiency from first principles. It provides a descriptive vocabulary (C_u, I_eff) that *characterises* the difference, but it does not explain *why* evolution arrived at these efficient solutions. That requires evolutionary theory (see Section 3).

### 2.3 Brain Regions and C_u

**Prediction:** Brain regions with higher C_u (more viability-relevant predictive and control information per unit metabolic cost) should correspond to areas identified by lesion studies and causal interventions as critical for adaptive behaviour.

**Specific expectations:**
- **Prefrontal cortex** (planning, flexible behaviour): high C_u because its representations encode abstract, transfer-relevant structure about task contingencies — exactly the cross-episode learned structure that H_t alone does not contain.
- **Hippocampus** (during encoding): high H(M) but moderate C_u — it stores episodic details that include nuisance information. Its C_u / H(M) ratio should be lower than neocortex.
- **Cerebellum** (motor prediction): high C_u for sensorimotor prediction, operating with extreme metabolic efficiency (high I_eff), consistent with its role in generating precise predictions with low metabolic overhead.
- **Primary sensory cortex**: moderate C_u relative to H(M) — these areas encode rich sensory detail, much of which is nuisance for any particular viability criterion.

**Testable approach:** Use representational similarity analysis (RSA) or probe classifiers applied to neural recordings to estimate I(neural activity; task-relevant variable) across brain regions, normalised by local glucose metabolism or BOLD signal. Regions with higher ratios should predict behavioural flexibility.

---

## 3. Predictions for Evolution

### 3.1 Compressed Representations and Fixed Action Patterns

The framework predicts that evolution should favour organisms that achieve high C_u with minimal H(M) — i.e., compressed representations of viability-relevant environmental structure. Fixed action patterns (FAPs) are an extreme case: a small, rigid behavioural program that achieves high viability in a specific niche.

**Analysis:**
- A FAP is like the rule learner in the toy: it encodes precisely the viability-relevant mapping (e.g., "if shadow overhead, freeze") with minimal representational overhead.
- The framework predicts that FAPs should have very high I_eff — they were "learned" (evolved) over many generations (high W_diss summed over evolutionary time) but the per-organism W_diss to deploy the FAP is minimal.
- FAPs should have high C_u within their niche-specific Z^V but low A across diverse environments. This is exactly what we observe: FAPs are highly effective in the relevant niche but brittle under environmental change.

**Prediction:** The evolutionary trajectory from simple reflexes to flexible cognition can be characterised as a shift from high-I_eff, low-A systems (FAPs: efficient but narrow) to lower-I_eff, high-A systems (general intelligence: less efficient per-task but broadly adaptive). This is a Pareto tradeoff that the framework makes explicit.

### 3.2 Intelligence vs Mere Adaptation

**Can C_u distinguish intelligent organisms from merely well-adapted ones?**

The framework's answer is nuanced. A well-adapted but non-intelligent organism (e.g., a bacterium with chemotaxis) has:
- Moderate C_u for its niche (it encodes gradient information relevant to nutrient location)
- High I_eff (chemotaxis is metabolically cheap)
- Low A (it cannot adapt to tasks outside its chemotactic niche)

An intelligent organism has:
- High C_u across diverse Z^V distributions (it encodes general-purpose predictive and control structure)
- Moderate I_eff (general-purpose representations are costlier to acquire and maintain)
- High A (it can deploy its structure across varied environments)

**The distinguishing feature is A, not C_u alone.** A bacterium may have locally high C_u (for its narrow Z^V), but its adaptive reach A is restricted. Intelligence, on this framework, requires that C_u is high across a broad reference distribution mu of environments. This is consistent with the paper's two-axis evaluation: intelligence is not a single scalar but a profile of (C_u, I_eff, A).

**Prediction:** Organisms at the boundary — those with moderate A and high niche-specific C_u (e.g., corvids, cephalopods) — should be distinguishable from mere adaptation by testing whether their C_u transfers to novel Z^V distributions. If their internal representations carry structure that generalises beyond the training niche, they have genuine intelligence. If their representations are niche-locked, they are merely well-adapted.

---

## 4. Predictions for Artificial General Intelligence

### 4.1 Scaling Laws

The framework makes a strong prediction about scaling laws: **adding parameters should help if and only if the additional capacity is used to encode more I(M; Z^V), not merely more H(M).**

**Specific predictions:**

- **Scaling parameter count without improving C_u should yield diminishing returns on A.** This is the lookup-table trap: a model that scales by storing more training examples verbatim (increasing H(M)) without compressing toward Z^V will plateau on out-of-distribution tasks.

- **The neural scaling law** (loss ~ N^{-alpha}) observed in language models reflects a regime where additional parameters *do* increase C_u, because language modelling requires compressing broad distributional structure that transfers to downstream tasks. The framework predicts this scaling should flatten when the model's representation saturates the available I(M; Z^V) for the training distribution — additional parameters would then encode only nuisance correlations.

- **Chinchilla-style compute-optimal scaling** (Hoffmann et al., 2022) is interpretable as optimising I_eff: for a fixed compute budget (W_diss), choose the model size and data quantity that maximise Delta C_u. Undertraining a large model wastes parameters (high H(M) capacity, low Delta C_u realised). Overtraining a small model wastes compute (high W_diss per unit Delta C_u at saturation).

- **Prediction:** Beyond some model size for a given task family, the marginal I_eff of each additional parameter should decrease. This should be measurable by tracking probe-estimated I(representation; task-relevant variable) as model size increases, normalised by training compute.

### 4.2 Understanding vs Pattern Matching

This is the framework's most philosophically charged prediction, and one must be careful not to overclaim.

The framework defines useful structure as I(M_t; Z^V_{t+tau} | H_t) — the mutual information between the agent's internal state and future viability-relevant variables, beyond what current observations provide. This conditional is critical: it measures what the agent's *learned model* contributes beyond raw input.

**Prediction:** A system that has "genuine understanding" (in the framework's terms) should have:
1. High C_u that persists across diverse Z^V distributions (not just the training distribution)
2. Representations where C_u is concentrated in abstract, compositional features rather than surface statistics
3. C_u that improves adaptive reach A in novel environments, not just in-distribution accuracy

A system that is merely pattern-matching should have:
1. High C_u on the training distribution but low C_u on shifted Z^V
2. Representations where H(M) is dominated by surface-level correlations
3. In-distribution accuracy that does not predict out-of-distribution A

**What the framework cannot do:** It cannot, from C_u alone, determine whether a system "truly understands." The framework measures whether internal structure is useful for prediction and control across environments. Whether this constitutes understanding in any deeper philosophical sense is outside its scope. The framework is deliberately agnostic on consciousness and phenomenal experience.

**Operational test:** Compare two models with equal in-distribution accuracy. Estimate C_u using probe classifiers on viability-relevant latent variables. The model with higher C_u should show better transfer, robustness, and compositional generalisation. If it does not, the C_u estimator is flawed or the framework's prediction is wrong.

---

## 5. Falsifiable Tests

### Test 1: C_u Predicts Transfer Better Than Parameter Count

**Measurement:** Train N architectures (varying size, depth, width) on the same task family. Estimate C_u using linear probes on task-relevant latent variables. Measure transfer performance on related but distinct task distributions.

**Expected result:** Rank correlation between C_u and transfer performance should exceed rank correlation between parameter count and transfer performance. The neural experiment (neural_experiment.py) demonstrates this in miniature: the bottleneck network (fewer params, high C_u) outperforms the wide network (more params, comparable C_u) on OOD environments.

**Falsification:** If raw parameter count predicts transfer performance as well as or better than C_u estimates across diverse architecture families, the framework's core claim — that useful structure matters more than raw capacity — is undermined.

**Feasibility:** Achievable within 6 months using existing model zoos (e.g., comparing BERT variants, ResNet sizes) and standard probe-based MI estimation techniques.

### Test 2: Compression Improves I_eff Without Degrading A

**Measurement:** Take a large pretrained model. Apply progressive pruning (structured, by magnitude; or by probe-estimated relevance to Z^V). At each pruning level, measure: (a) C_u via probe MI, (b) W_diss via wall-plug energy per inference, (c) A via performance across a distribution of evaluation tasks.

**Expected result:** Relevance-guided pruning should maintain C_u and A to higher compression ratios than random or magnitude-based pruning. I_eff should increase monotonically with pruning until C_u begins to drop.

**Falsification:** If magnitude-based pruning (which is agnostic to Z^V) matches or exceeds relevance-guided pruning in maintaining A, then the usefulness filter adds no predictive value beyond standard sparsity methods.

**Feasibility:** Achievable in under 1 year. Requires implementing a Z^V-aware pruning criterion, which could be approximated by probe-estimated MI on a held-out viability-relevant variable set.

### Test 3: LoRA Has Higher I_eff Than Full Fine-Tuning

**Measurement:** Fine-tune the same base model on the same task using (a) full fine-tuning, (b) LoRA at varying ranks. Measure Delta C_u (change in probe MI on Z^V pre- vs post-training) and W_diss (total training energy, measured at the wall plug or estimated from FLOPs).

**Expected result:** I_eff(LoRA) > I_eff(full fine-tuning) for tasks where the required adaptation is low-rank. For tasks requiring high-rank adaptation, the difference should shrink or reverse.

**Falsification:** If LoRA consistently achieves lower I_eff than full fine-tuning (lower Delta C_u per unit energy), the framework's prediction about efficient adaptation is wrong, or LoRA's efficiency gains do not translate to the thermodynamic level.

**Feasibility:** Directly testable today. Multiple groups have the infrastructure to measure both probe MI and training energy for LoRA vs full fine-tuning.

### Test 4: Sleep Improves C_u / H(M) Ratio in Neural Representations

**Measurement:** Record neural activity (or use fMRI representational patterns) before and after sleep in subjects performing a learning task. Estimate C_u (MI between neural representation and task-relevant variable) and H(M) (total representational entropy) using RSA or probe methods.

**Expected result:** Post-sleep representations should show higher C_u / H(M) ratio than pre-sleep representations. Sleep-deprived subjects should show smaller improvements or degradation of this ratio.

**Falsification:** If sleep does not systematically improve the C_u / H(M) ratio — or if sleep-deprived subjects show equal or better compression — the framework's account of sleep consolidation is empirically inadequate.

**Feasibility:** Feasible within 2 years. Requires careful experimental design (controlled sleep manipulation, appropriate representational analysis). Related work on post-sleep generalisation improvements exists but has not been framed in MI terms.

### Test 5: MoE Routing Quality Predicts I_eff

**Measurement:** Compare MoE models with varying routing quality (random routing, hash-based routing, learned routing, oracle routing that uses Z^V directly). Measure per-input W_diss (FLOPs of activated pathway) and C_u (probe MI on task-relevant variables).

**Expected result:** Better routing should yield higher I_eff because it activates experts whose parameters contribute to I(M; Z^V) for the given input, reducing wasted computation. Oracle routing should achieve the highest I_eff. Random routing should approach the I_eff of a dense model (no savings from sparsity).

**Falsification:** If routing quality has negligible effect on I_eff — i.e., if all routing strategies produce similar C_u / W_diss ratios — then the framework's prediction about MoE efficiency is incorrect, or the routing mechanism does not function as an information bottleneck.

**Feasibility:** Testable within 1 year using existing MoE implementations with swappable routing modules.

---

## 6. Comparison with Competing Frameworks

### 6.1 C_u vs Integrated Information Theory (Phi)

**Integrated Information Theory (IIT)** defines consciousness (and, by extension, a form of intelligence) via Phi — the amount of integrated information in a system, measured as the information generated by the whole above and beyond its parts.

| Dimension | C_u framework | IIT (Phi) |
|-----------|--------------|-----------|
| **Target phenomenon** | Adaptive intelligence | Consciousness / experience |
| **Core quantity** | I(M; Z^V \| H_t) — task-relevant MI | Phi — integrated information beyond partitions |
| **Requires task/environment?** | Yes — Z^V must be specified relative to viability | No — Phi is intrinsic to the system |
| **Computational tractability** | Estimable via probes and MI bounds; scales to real networks | Exact computation is intractable for systems beyond ~15 nodes; approximations (Phi*) exist but are debated |
| **Handles usefulness?** | Explicitly — only viability-relevant structure counts | No — Phi does not distinguish useful from useless integration |
| **Physically grounded?** | Yes — W_diss provides a physical cost term | Partially — information is physical, but Phi has no explicit energy/dissipation term |

**Where C_u succeeds and Phi does not:**
- C_u can distinguish a system that integrates enormous amounts of irrelevant information (high Phi, low C_u) from one that integrates modest amounts of viability-relevant information (lower Phi, high C_u). The paper's noise memoriser illustrates this: it processes and stores information but C_u = 0.
- C_u is computationally tractable for real neural networks and ML systems via probe-based MI estimation. Phi computation remains impractical for systems of biologically or engineering-relevant scale.
- C_u explicitly includes an efficiency term (I_eff), connecting intelligence to physical cost. IIT has no cost dimension.

**Where Phi succeeds and C_u does not:**
- Phi addresses the *intrinsic* character of a system's information processing. C_u is explicitly observer-relative in the sense that Z^V must be specified. Two evaluators with different Z^V definitions will assign different C_u values to the same system. Phi, whatever its other problems, aims for an observer-independent measure.
- IIT provides candidate explanations for why certain systems have subjective experience. C_u makes no claims about consciousness, phenomenal experience, or subjectivity.

**Relationship:** The two frameworks are not strictly competing — they target different phenomena. A system could have high Phi and low C_u (rich internal integration, but about nothing viability-relevant), or high C_u and low Phi (useful but modular representations). An interesting research direction would be to test whether Phi and C_u correlate empirically, which would suggest that viability-relevant computation tends to require integration, or dissociate, which would clarify their respective scopes.

### 6.2 C_u vs Free Energy Principle (FEP)

The Free Energy Principle (Friston, 2010) posits that adaptive systems minimise variational free energy — a bound on surprisal — by updating their generative models of the world. Active inference extends FEP to include action as a means of confirming predictions.

| Dimension | C_u framework | Free Energy Principle |
|-----------|--------------|----------------------|
| **Core mechanism** | Acquire and compress useful structure | Minimise variational free energy (prediction error + complexity) |
| **Cost term** | W_diss (physically dissipated work) | Complexity cost in the variational bound (information-theoretic, not directly physical) |
| **Action** | Captured via adaptive reach A and empowerment-related Z^V | Central — active inference is inherently about perception-action loops |
| **Scope** | Definition + measurement of intelligence | General principle of adaptive self-organising systems |
| **Normativity** | External — Z^V specified by viability criterion | Internal — surprisal defined relative to the system's own generative model |

**Where C_u succeeds and FEP does not:**
- C_u provides an explicit **efficiency metric** (I_eff) that FEP lacks. FEP describes *what* adaptive systems do (minimise free energy) but does not directly yield a cost-efficiency measure of *how well* they do it relative to physical expenditure.
- C_u's usefulness filter is more explicit. FEP's complexity cost penalises model complexity in the variational sense, but this is not identical to filtering for viability-relevance. A system minimising variational free energy might develop complex models of irrelevant regularities if they reduce prediction error on the training distribution.
- C_u distinguishes two systems that achieve equal free-energy minimisation but differ in the proportion of their learned structure that transfers to novel environments.

**Where FEP succeeds and C_u does not:**
- FEP provides a **generative mechanism** — active inference — that explains *how* systems acquire useful structure. C_u defines what useful structure is and how to measure it, but does not specify the learning dynamics that produce it.
- FEP's treatment of action is far more developed. The paper acknowledges this: "active inference already provides a rich account of viability and attracting states; the present contribution is narrower."
- FEP naturally handles the perception-action loop. C_u, as currently formulated, measures representational quality but does not model the closed-loop dynamics of sensing, acting, and updating.

**Relationship:** The frameworks are substantially complementary. C_u can be understood as providing a thermodynamic efficiency metric for the structure that FEP-based systems acquire. The paper positions itself this way: it is not replacing active inference but adding "an explicit efficiency term for the acquisition of relevance-filtered structure." A productive synthesis would use FEP to model the dynamics of learning and C_u to evaluate the efficiency and quality of what was learned.

### 6.3 Summary of Comparative Strengths

| Criterion | C_u | Phi (IIT) | FEP |
|-----------|-----|-----------|-----|
| Distinguishes useful from useless structure | Yes | No | Partially (complexity penalty) |
| Physically grounded cost metric | Yes (W_diss) | No | No (variational cost is info-theoretic) |
| Computationally tractable | Yes (probe MI) | No (intractable for large systems) | Partially (model-dependent) |
| Handles action/control | Partially (via A) | No | Yes (active inference) |
| Observer-independent | No (Z^V must be specified) | Aims to be | Partially (model-relative) |
| Explains learning dynamics | No | No | Yes |
| Addresses consciousness | No | Yes | Debated |
| Applicable across substrates | Yes | Yes (in principle) | Yes |

---

## Summary of Key Findings

**What the framework does well:**
1. It provides a clear, operationalisable distinction between useful and useless internal structure — a distinction that neither IIT nor raw thermodynamic efficiency metrics make.
2. Its predictions about compression, fine-tuning, and MoE architectures are specific and testable with existing infrastructure.
3. The two-axis evaluation (C_u/I_eff for efficiency, A for capability) avoids the trap of collapsing intelligence into a single scalar.

**Where the framework is incomplete:**
1. Estimating Z^V in practice requires domain-specific modelling decisions that the framework does not automate. The framework says "measure C_u" but does not say "here is how to identify Z^V for this system."
2. The framework is a measurement and evaluation scheme, not a generative theory. It does not predict *how* systems should learn, only how to evaluate *what* they learned.
3. The connection to the paper's formal C_u = I(M_t; Z^V_{t+tau} | H_t) and the toy computation's operational C_u = I(M; Z) involves a simplification (H_t = empty set) that is acknowledged but may limit applicability to sequential, multi-episode learning scenarios.

**What would strengthen the framework most:**
1. A concrete, validated estimator for C_u in a real ML system (not just the toy), with comparison to competing metrics.
2. A demonstration that C_u-based model selection outperforms parameter-count or loss-based model selection on transfer tasks.
3. An explicit bridge between C_u and FEP-style active inference dynamics, showing that free-energy minimising agents naturally maximise C_u under appropriate conditions.
