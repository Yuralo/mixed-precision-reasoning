# Continue-or-pivot decision memo

## Decision

**Recommendation: Continue, but narrow scope.**

**Strongest defensible framing today: C. Empirical Study of Quantization-Induced Reasoning Trajectory Shifts.**

The paired precision effect is real in these artifacts, but the stronger causal/control thesis is not yet earned. The current evidence supports a short, explicitly gated continuation: run the temperature control and one independent replication. Do not invest in token-level precision kernels unless those gates pass.

## 1. What replicated?

| Result | Train (400) | Held-out test (400) | Interpretation |
|---|---:|---:|---|
| FP16 accuracy | 81.25% | 69.00% | FP16 is stronger in both splits |
| BNB4 accuracy | 72.50% | 63.00% | Quantization costs aggregate accuracy |
| FP16-only wins | 57 | 57 | Harm from BNB4 is repeatable |
| BNB4-only wins | 22 | 33 | Success sets are not nested |
| Oracle selector | 86.75% | 77.25% | Counterfactual complementarity is large |

On test, the FP16–BNB4 difference is 6.00% with paired bootstrap 95% CI [1.50%, 10.50%] and exact McNemar p=0.0149. The answer-flip rate is 38.75%, far larger than the net accuracy gap. This supports H1 for one model/dataset/quantizer.

The clean subset retains 25 BNB4-only wins and 47 FP16-only wins across 325 examples; therefore the phenomenon is not explained away by truncation or missing explicit answers.

## 2. What failed or remains unsupported?

- **Utility routing is weak.** At a 10% rerun budget, the best learned model reaches 65.50%; entropy reaches 66.00%, always-FP16 reaches 69.00%, and the oracle reaches 73.00%. At 20%, the oracle reaches 77.25%. The opportunity exists, but the current features do not identify it reliably.
- **Early-token control is not ready.** The best beneficial-switch ROC-AUC is 0.502 at 16 tokens, 0.572 at 32, and 0.639 at 64. Even the best 10% prefix router reaches only 64.75%.
- **Longer does not yet mean more reasoning.** BNB4 emits more tokens overall, but heuristic arithmetic-expression and self-correction counts do not rise systematically in rescue cases.
- **The clean trajectory contrast is suggestive, not conclusive.** Clean BNB4 rescues are +21.72 tokens longer on average, while clean BNB4-induced failures are -14.30; the groups are small and their individual confidence intervals include zero.
- **Naive first-divergence position is not mechanistic evidence.** Independent greedy outputs often differ immediately because tokenization/wording paths split; a same-prefix logit comparison is needed.
- **Efficiency is currently negative on this hardware.** BNB4 is memory-saving but slower at batch-one decoding in the recorded RTX 3090 run. No speed claim is justified.
- **H3 is unanswered.** No FP16 temperature artifact is present, so we cannot yet claim precision is distinct from ordinary decoding noise.

## 3. Strongest thesis framing

Choose **C. Empirical Study of Quantization-Induced Reasoning Trajectory Shifts** for now.

The phrase “precision as a control variable” should remain a hypothesis in the title/abstract until two things happen: (1) FP16 temperature sampling fails to reproduce BNB4’s rescue/failure sets, and (2) the signed trajectory effect replicates on another dataset, model, or quantizer. If those pass, promote to framing A. If complementarity replicates but mechanism/routing remains weak, use framing B or C.

## 4. Continue or not?

Continue for one focused evidence sprint, not as an open-ended systems build.

1. Run 100 test prompts at FP16 temperatures 0.3 and 0.7 with three samples each.
2. Compare BNB4 rescue-set overlap, answer diversity, length, and majority-vote outcomes.
3. Replicate greedy FP16/BNB4 on one genuinely independent setting: preferably a small MATH subset or Qwen2.5-3B, not another GSM8K slice alone.
4. Stop token-level implementation work unless a prefix router beats entropy and shows useful net gain before token 64.

### Kill criteria

Pivot away if BNB4-only wins largely coincide with ordinary FP16 sampling rescues, the paired complementarity collapses in the independent replication, or clean BNB4-only wins become negligible. If the effects replicate but routing stays weak, keep the project as a careful empirical trajectory study rather than an adaptive-inference thesis.

## Bottom line

There is a credible phenomenon: numerical precision changes which GSM8K problems this model solves. There is not yet evidence that precision is a practically controllable reasoning knob. The project is worth the next two discriminating experiments; it is not yet worth custom kernels or a broad PhD-level claim.
