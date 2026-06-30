# Stage 1 results and novelty audit

The completed cross-experiment synthesis, including the temperature control and final
thesis recommendation, is in [`FINAL_ASSESSMENT.md`](FINAL_ASSESSMENT.md).

Audit date: 30 June 2026

## Executive verdict

The experiment supports two claims:

1. Four-bit weight quantization causes a statistically detectable loss on held-out
   GSM8K, and failures are concentrated rather than uniformly random.
2. Cheap aggregate features from the quantized trace predict FP-correct/BNB4-wrong
   cases substantially better than random and better than single entropy or margin
   features.

It does **not** yet support three stronger claims:

1. The learned controller does not beat entropy at the key 10% end-to-end budget.
2. The current implementation is slower than FP16 and has no logged memory result.
3. The broad idea of dynamic uncertainty-aware precision is not novel as of 2026.

The utility-aware controller is now implemented. It confirms that the oracle
opportunity is much easier to state than to realize: the learned four-outcome model
does not beat entropy at the main 10% budget, and early-prefix prediction is weak.
The next decisive question is whether BNB4 behavior is distinct from FP16 temperature
sampling.

## 1. Artifact integrity

Both official splits contain exactly 400 FP rows, 400 BNB4 rows, and identical ID
sets. There are no duplicate IDs, reference mismatches, or differences between stored
answers and the current extractor. Model metadata confirms Qwen2.5-1.5B-Instruct,
FP16 CUDA reference, and BNB4 CUDA quantization.

The sample is deterministic but not randomly selected: it uses indices 0–399 from
each official split. Future replication should use a recorded random sample or the
full split.

## 2. Paired accuracy

| Split | n | FP16 | BNB4 | Difference | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K train | 400 | 81.25% | 72.50% | −8.75 pp | [−13.0, −4.5] pp | 0.000103 |
| GSM8K test | 400 | 69.00% | 63.00% | −6.00 pp | [−10.5, −1.5] pp | 0.0149 |
| Clean test subset | 325 | 71.69% | 64.92% | −6.77 pp | — | 0.0128 |

On held-out test, FP16 has a Wilson 95% interval of 64.30–73.33%; BNB4 has
58.17–67.59%. The paired bootstrap interval is the relevant result because both models
answer the same prompts.

### Held-out outcome counts

| Outcome | Count | Fraction |
|---|---:|---:|
| Both correct | 219 | 54.75% |
| FP correct, BNB4 wrong | 57 | 14.25% |
| FP wrong, BNB4 correct | 33 | 8.25% |
| Both wrong | 91 | 22.75% |

Among the 276 FP-correct examples, 57 fail after quantization: 20.65%, Wilson 95% CI
16.30–25.82%. Discordant examples favor FP by 57:33, an odds ratio of 1.73.

The answer flip rate is 38.75%. This is larger than the six-point net accuracy loss
because quantization sometimes rescues an FP mistake. Aggregate accuracy therefore
hides substantial reasoning-path churn.

## 3. Quality-controlled subset

The held-out test run has:

- FP token-cap rate: 1.75%; BNB4 token-cap rate: 4.0%;
- FP explicit-answer rate: 85.5%; BNB4 explicit-answer rate: 92.5%;
- 325 examples where both outputs are explicit and non-truncated.

On those 325 examples, the labels are 186 both-correct, 47 FP-only, 25 BNB4-only,
and 67 both-wrong. The gap remains significant. Truncation and answer parsing do not
explain the principal effect.

There is nevertheless a stopping limitation: the four-token grace period can stop on
an unfinished expression. One held-out FP output ends with `Final answer: 56 - 6` and
is parsed as 56. Future runs must stop only after a syntactically complete terminal
answer or EOS. Existing Stage 1 results should retain this caveat.

## 4. Oracle opportunity

Always BNB4 scores 63%; always FP16 scores 69%. A perfect per-example selector scores
77.25% by using FP for the 57 FP-only cases and retaining BNB4 for the 33 BNB4-only
cases.

| FP budget | Oracle accuracy | Useful interventions |
|---:|---:|---:|
| 0% | 63.00% | 0 |
| 5% | 68.00% | 20 |
| 10% | 73.00% | 40 |
| 20% | 77.25% | 57 |

The oracle fully saturates at a 14.25% actual intervention rate. At a nominal 10%
budget it recovers 10 of the available 14.25 points above BNB4—about 70% of the
selectable gain. This is a strong **existence result**, not evidence that the current
controller can realize it.

## 5. Failure concentration

Within FP-correct examples, univariate ranking AUCs on held-out test are:

| Quantized-only feature | AUC |
|---|---:|
| Generated length | 0.703 |
| Maximum entropy | 0.693 |
| Negative minimum logit margin | 0.669 |
| Negative mean logit margin | 0.650 |
| Mean entropy | 0.648 |
| Negative mean token probability | 0.644 |

FP-correct/BNB4-wrong examples average 203 generated tokens versus 139 for examples
where both are correct. Their mean maximum entropy is 2.57 versus 2.29; their mean
token probability is 0.889 versus 0.906.

Generated length is a strong feature but is known only after generation finishes. An
ablation excluding length is required before claiming an online uncertainty signal.

## 6. Held-out predictor

Training uses 325 FP-correct examples from GSM8K train. Evaluation uses 276 FP-correct
examples from official test, including 57 positives.

| Method | ROC-AUC (95% CI) | PR-AUC (95% CI) | Recall @10% | Precision @10% |
|---|---:|---:|---:|---:|
| Logistic regression | 0.804 [0.730, 0.867] | 0.608 [0.486, 0.731] | 0.351 | 0.714 |
| Random forest | 0.762 [0.691, 0.829] | 0.519 [0.399, 0.645] | 0.298 | 0.607 |
| Entropy | 0.693 [0.625, 0.767] | 0.427 [0.310, 0.561] | 0.281 | 0.571 |
| Margin | 0.669 [0.602, 0.738] | 0.319 [0.236, 0.421] | 0.228 | 0.464 |
| Random | 0.513 | 0.234 | 0.140 | 0.286 |

Positive prevalence is 0.207, so logistic PR-AUC is almost three times prevalence.
The clean-only evaluation remains predictive: logistic ROC-AUC 0.787 and PR-AUC
0.545.

This answers the diagnostic research question positively: failure states are
predictable. It does not automatically yield a good intervention policy because the
classifier is trained only on FP-correct examples and never learns which interventions
will destroy BNB4-correct answers.

## 7. End-to-end controller utility

| Policy | 5% FP | 10% FP | 20% FP | 50% FP |
|---|---:|---:|---:|---:|
| Entropy | 64.25% | **66.00%** | 66.25% | 68.25% |
| Logistic regression | **64.75%** | 65.25% | **66.75%** | 69.50% |
| Random forest | 64.00% | 65.00% | 66.50% | **70.50%** |
| Oracle | 68.00% | 73.00% | 77.25% | 77.25% |

At 10%, entropy is the best implemented policy. This fails the original go/no-go
criterion that the learned controller should beat entropy at the main budget. Learned
models have a small advantage at some larger budgets, and random forest exceeds
always-FP accuracy at 50%, but that uses FP on half of all examples.

The clean subset shows the same pattern: entropy is best at 10%; learned models become
better at 20–50%.

### Signed four-outcome controller

The revised controller is trained on all four outcomes and scores each example by
`P(FP-only) - P(BNB4-only)`. This directly penalizes harmful FP reruns.

| Quantized-only feature set | Beneficial ROC-AUC | PR-AUC | Accuracy @10% | Accuracy @20% |
|---|---:|---:|---:|---:|
| Length-only logistic | 0.568 | 0.200 | 65.00% | 66.00% |
| Uncertainty random forest | 0.657 | 0.245 | 65.25% | 66.50% |
| All-feature random forest | 0.647 | 0.236 | **65.50%** | 66.50% |
| Entropy threshold | — | — | **66.00%** | 66.25% |
| Oracle | 1.000 | 1.000 | 73.00% | 77.25% |

This is a negative result for H4 in its current form. The oracle gap at 10% is 7.5
points above the learned router, but current trajectory aggregates do not identify
the right examples reliably enough.

## 8. Early-prefix feasibility

Only BNB4 features visible within a fixed prefix were used. Every completion for a
given prompt remains in the same train/test split.

| Prefix | Best model | Beneficial ROC-AUC | PR-AUC | Accuracy @10% |
|---:|---|---:|---:|---:|
| 16 tokens | Logistic | 0.502 | 0.152 | 64.00% |
| 32 tokens | Random forest | 0.572 | 0.193 | 64.75% |
| 64 tokens | Random forest | 0.639 | 0.278 | 64.75% |

The signal emerges only gradually and does not beat static FP16, entropy routing, or
the full-trajectory router. Token-level switching is therefore not justified by the
present evidence.

## 9. Runtime and token inflation

| Metric | FP16 | BNB4 | Relative result |
|---|---:|---:|---:|
| Mean generated tokens | 152.76 | 165.11 | BNB4 +8.08% |
| Mean tokens/second | 54.09 | 36.64 | BNB4 −32.3% |
| Mean example latency | 2.82 s | 4.50 s | BNB4 +59.5% |

bitsandbytes NF4 reduces weight memory but is not a fast batch-one kernel on this
setup. The experiment logged no peak VRAM, resident model memory, or controller rerun
latency. Consequently, no current result demonstrates a favorable memory-latency-
accuracy trade-off.

An example-level policy that first completes BNB4 generation and then reruns 10% in
FP would have approximate mean generation latency of `4.50 + 0.1 × 2.82 = 4.78 s`,
before model-switching overhead. That is slower than always FP16. A useful system will
need a genuinely efficient low-precision backend, early intervention, sequential model
loading, or a memory-first deployment objective.

The 1.081× token-inflation observation is consistent with the recent
[Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) result, but is not
novel by itself.

## 10. Trajectory structure and qualitative failure modes

BNB4 is longer overall, but extra tokens are not equivalent to useful deliberation.
On the clean subset, BNB4-only rescues average +21.72 tokens relative to FP16 while
FP16-only cases average −14.30 tokens. The 36.02-token contrast has bootstrap 95% CI
2.73–67.56. However, rescue cases do not systematically add arithmetic expressions
or explicit self-correction markers. The safer claim is that precision changes
trajectory length and commitment timing, not that quantization makes the model
“reason more.”

Independent greedy traces commonly diverge at token zero and before the first
arithmetic expression. This makes raw first-divergence position a poor mechanistic
measure; future work should compare logits under a shared prefix.

The paired traces show genuine reasoning changes rather than extraction-only failures:

- In test example 173, FP correctly totals $1,500/month and $18,000/year; BNB4 returns
  $15,600 after omitting or miscombining a payment.
- In example 203, BNB4 incorrectly multiplies three omelets across seven days and
  returns 43 instead of 31.
- In example 219, BNB4 drops the “10 more than half” adjustment and returns 64 instead
  of 54.

Quantization rescues are equally instructive:

- In example 10, FP computes 70% of 180 as 135; BNB4 correctly computes 126 and the
  final total 366.
- In example 171, FP sums 640 + 540 + 30 as 1190; BNB4 correctly returns 1210.

These examples explain why “high precision is always safer” is false at the individual
example level. The policy must estimate intervention utility or abstain.

## 11. Novelty assessment

The original broad framing is not novel:

- [FlexQuant](https://arxiv.org/abs/2506.12024) dynamically switches token/layer
  precision using perplexity entropy, KL divergence, and top-logit tolerance.
- [DP-LLM](https://arxiv.org/abs/2508.06041) chooses layer precision at each decoding
  iteration using lightweight input-dependent quantization-error estimates.
- [Quantization Meets Reasoning](https://arxiv.org/abs/2505.11574) localizes early
  reasoning failure frontiers and restores token margins through targeted tuning.
- [Extreme Low-Bit Inference in Reasoning Models](https://arxiv.org/abs/2606.02011)
  detects 2-bit generation pathologies and selectively invokes FP16 planning or rescue.
- [ReQAT](https://arxiv.org/abs/2606.15682) identifies critical low-entropy symbolic
  tokens and performs reasoning-centric W4A4KV4 training.
- [DynamicPTQ](https://arxiv.org/abs/2606.12487) selects higher activation precision
  for layers identified through residual-stream dynamics.

### Potentially defensible contribution

A narrower thesis can still be valuable:

> A calibrated, quantized-only controller that predicts the signed counterfactual
> utility of precision recovery for reasoning, explicitly models harmful interventions,
> and trades critical-error risk against precision budget and abstention.

This differs from maintaining token-level distribution similarity or minimizing
quantization error. It asks whether precision should be spent because it changes task
correctness in the desired direction. To claim this contribution, the project must
actually implement and beat strong baselines using this utility-aware objective.

## 12. Required next experiments

Do these before collecting another large identical BNB4 sample:

1. Replicate on one pre-registered independent setting, preferably a small MATH subset.
2. Reuse the paired outcomes, clean filter, length contrast, and oracle analysis.
3. Fix stopping so arithmetic expressions cannot be cut after the first number.
4. Log peak VRAM, resident model memory, time-to-first-token, and policy latency.
5. Do not implement token/layer kernels unless an early router becomes useful.

## 13. Temperature control

The completed control uses 100 prompts and three FP16 samples at each temperature:

| Condition | Per-completion accuracy | Pass@3 | Majority accuracy | BNB4 rescue coverage: any / majority | Mean tokens |
|---|---:|---:|---:|---:|---:|
| FP16 T=0.3 | 66.00% | 83.00% | 67.00% | 66.67% / 11.11% | 156.33 |
| FP16 T=0.7 | 69.67% | 89.00% | 70.00% | 100.00% / 55.56% | 149.90 |
| FP16 greedy | 69.00% | — | — | — | 145.16 |
| BNB4 greedy | 63.00% | — | — | — | 168.99 |

There are only nine BNB4-rescue prompts in this subset. Temperature 0.7 produces at
least one correct FP16 sample for all nine and a correct majority for five. The rescue
Jaccard is 37.5% because FP16 sampling rescues many additional prompts. A grouped
correctness model gains −0.0009 ROC-AUC and +0.0004 log loss when precision mode is
added after temperature and trajectory features—no measurable improvement.

This weakens H3: much of the correctness complementarity is reproducible by ordinary
sampling. It does not prove full equivalence because BNB4 remains substantially longer
and majority overlap is incomplete.

## 14. Bottom line

There is a real and statistically credible research signal. The strongest result is
not that BNB4 is worse—it is that quantization-induced correctness flips are
predictable across official train/test splits, while a perfect sparse selector has a
large 14.25-point upside over static BNB4.

The current controller and backend do not realize that upside efficiently. The
project should continue only through one independent pre-registered replication. Its
strongest defensible framing is an empirical study of precision-induced trajectory
shifts. The strong control-variable and learned-routing claims are unsupported.
