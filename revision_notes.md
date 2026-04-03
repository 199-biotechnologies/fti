# Revision notes for `paper_revised.typ`

This revision is designed for a perspective/proposal submission rather than a regular theory paper.

## Main conceptual changes

1. **Sharpens the distinct claim**
   - States explicitly that the paper's novelty is the usefulness filter on the numerator.
   - Positions Takahashi, Fagan, and Perrier as valid physical-efficiency metrics rather than as defective theories.
   - Avoids the overclaim that other formalisms are "not definitions" in any absolute sense.

2. **Separates stock from flow**
   - Defines useful internal structure as a stock variable, `C_u`.
   - Interprets `I_eff = ΔC_u / W_diss` as **acquisition efficiency** rather than total intelligence.
   - Addresses the steady-state objection: deployed systems can have `ΔC_u = 0` while still having high adaptive reach.

3. **Gives `A` a formal anchor**
   - Defines adaptive reach over a reference distribution of environments or task families.
   - Makes the viability/goal criterion explicit as a parameter of the framework.

4. **Gives `C_u` a concrete candidate definition**
   - Proposes `C_u = I(M_t ; Z_(t+τ)^V | H_t)`.
   - Connects this to semantic information and the information bottleneck.
   - Makes clear that this is a candidate operational route, not a solved universal estimator.

5. **Clarifies the role of Still's bound**
   - Uses Still only for the predictive-memory side.
   - Adds empowerment and active inference for control and viability.
   - Avoids implying that Still alone grounds active control.

6. **Adds a worked toy example**
   - Minimal latent-rule vs lookup-table vs nuisance-memory comparison.
   - Includes actual numbers to show how the framework discriminates systems that can tie on benchmark score.

7. **Adds temperature normalisation**
   - Introduces a dimensionless `I_eff*` for cross-substrate comparison.

## Main stylistic changes

- Removes most meta-narration.
- Replaces the old "First-principles argument" section with a more direct argument about useful structure.
- Tightens the conclusion around one core claim.
- Rewrites the positioning table to be fairer to adjacent work.

## What still needs manual checking

1. Compile the Typst file and check equation/table layout.
2. Decide whether to keep the expanded title or revert to the original shorter title.
3. Check journal word limit.
4. Decide whether to keep the toy example as a table in the main text or move it to a boxed example.
5. Consider adding one sentence on model-free reinforcement learning if you want to pre-empt that reviewer objection explicitly.
