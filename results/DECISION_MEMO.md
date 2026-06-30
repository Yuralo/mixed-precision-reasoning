# Continue-or-pivot decision memo

## Decision

**Recommendation: Only continue as empirical analysis.**

**Strongest defensible framing today: C. Empirical Study of Quantization-Induced Reasoning Trajectory Shifts.**

The paired precision effect is real, but the temperature control weakens the claim that precision is a distinct, exploitable control variable. Continue only through one independent replication; do not invest in token-level kernels or present adaptive precision as a demonstrated system.

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
- **H3 is weakened.** At temperature 0.7, any of three FP16 samples solves 100.00% of the BNB4-rescue prompts; majority vote reproduces 55.56%. Adding precision mode to the grouped controlled model changes ROC-AUC by only -0.0009 and log loss by +0.0004. Precision does not add predictive value in this diagnostic.
- **Temperature is not fully equivalent either.** The temperature-0.7 rescue-set Jaccard is only 37.50%, and its mean output is -19.09 tokens relative to BNB4 greedy. Sampling reproduces correctness opportunities more than it reproduces the BNB4 trajectory-length shift.

## Temperature-versus-quantization control

| Temperature | Per-completion accuracy | Empirical pass@k | Majority accuracy | Any-sample BNB4 rescue coverage | Majority BNB4 rescue coverage | Rescue Jaccard | Mean tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 66.00% | 83.00% | 67.00% | 66.67% | 11.11% | 33.33% | 156.33 |
| 0.7 | 69.67% | 89.00% | 70.00% | 100.00% | 55.56% | 37.50% | 149.90 |

Interpret overlap jointly with the grouped controlled-correctness diagnostic. Any-sample pass@k is expected to rise with repeated sampling and is not itself evidence of equivalence.

## 3. Strongest thesis framing

Choose **C. Empirical Study of Quantization-Induced Reasoning Trajectory Shifts** for now.

The phrase “precision as a control variable” should remain a hypothesis in the title/abstract until two things happen: (1) FP16 temperature sampling fails to reproduce BNB4’s rescue/failure sets, and (2) the signed trajectory effect replicates on another dataset, model, or quantizer. If those pass, promote to framing A. If complementarity replicates but mechanism/routing remains weak, use framing B or C.

## 4. Continue or not?

Continue only as a bounded empirical replication, not as an adaptive-precision systems build.

1. Replicate greedy FP16/BNB4 on one genuinely independent setting: preferably a small MATH subset or Qwen2.5-3B, not another GSM8K slice alone.
2. Pre-register the same paired outcomes, clean filter, token-length contrast, and oracle selector before looking at the replication.
3. Stop token-level implementation work unless a prefix router beats entropy and shows useful net gain before token 64.

### Kill criteria

Pivot away if BNB4-only wins largely coincide with ordinary FP16 sampling rescues, the paired complementarity collapses in the independent replication, or clean BNB4-only wins become negligible. If the effects replicate but routing stays weak, keep the project as a careful empirical trajectory study rather than an adaptive-inference thesis.

## Bottom line

Numerical precision changes the observed greedy trajectory, but ordinary FP16 sampling recovers much of the same correctness complementarity. The strong control-variable thesis is currently unsupported. One independent replication is justified; custom kernels and a broad adaptive-precision claim are not.
