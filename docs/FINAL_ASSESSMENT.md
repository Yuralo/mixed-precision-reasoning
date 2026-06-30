# Final evidence assessment: is this thesis direction good enough?

Assessment date: 30 June 2026

## Executive decision

**Recommendation: only continue as empirical analysis, with one final independent
replication.**

**Strongest defensible framing today:**

> Empirical Study of Quantization-Induced Reasoning Trajectory Shifts

The original strong framing—**Precision as a Control Variable for Reasoning**—is not
supported by the current evidence. Numerical precision changes the observed greedy
trajectory, answer, length, and uncertainty profile, but the temperature experiment
shows that much of the apparent correctness complementarity is also accessible by
ordinary FP16 sampling. The learned utility controller does not beat entropy or
always-FP16, and early-token features are too weak for dynamic switching.

This is a good research result because it falsifies the easy version of the thesis.
It is not yet a strong standalone PhD thesis contribution.

## 1. Evidence inventory

The assessment uses only saved experiment artifacts:

- 400 paired FP16/BNB4 generations from the GSM8K train split;
- 400 paired FP16/BNB4 generations from the GSM8K test split;
- token-level entropy, probability, margin, token ID, latency, and length logs;
- strict explicit-answer/non-truncation subsets;
- four-outcome utility-controller evaluation;
- 16-, 32-, and 64-token prefix prediction;
- 600 FP16 sampled completions on 100 prompts: three at temperature 0.3 and three at
  temperature 0.7;
- 91,871 token rows from the temperature experiment.

No claim below relies on the original contaminated 200-example pilot.

## 2. Static FP16 versus BNB4

| Result | Train, n=400 | Test, n=400 |
|---|---:|---:|
| FP16 accuracy | 81.25% | 69.00% |
| BNB4 accuracy | 72.50% | 63.00% |
| FP16 minus BNB4 | 8.75 points | 6.00 points |
| Paired bootstrap 95% CI | 4.50–13.00 | 1.50–10.50 |
| Exact McNemar p | 0.000103 | 0.0149 |
| FP16-only wins | 57 | 57 |
| BNB4-only wins | 22 | 33 |
| Answer-flip rate | 28.25% | 38.75% |
| Oracle selector | 86.75% | 77.25% |

The key positive result is not merely the six-point loss. On test, 90 examples are
discordant: FP16 alone solves 57 and BNB4 alone solves 33. The success sets are not
nested. Aggregate accuracy hides much larger answer churn.

The strict test subset contains 325 examples where both models emit explicit answers
without hitting the token ceiling:

| Clean outcome | Count |
|---|---:|
| Both correct | 186 |
| FP16 only | 47 |
| BNB4 only | 25 |
| Both wrong | 67 |

Clean FP16 accuracy is 71.69%, clean BNB4 accuracy is 64.92%, and exact McNemar
p=0.0128. Therefore, formatting and truncation do not explain away the principal
paired effect.

### Interpretation

- **Supported:** quantization changes which problems greedy decoding solves.
- **Supported:** the effect is larger than the aggregate accuracy gap suggests.
- **Not supported:** quantization is beneficial on average; FP16 remains better.
- **Not supported:** a BNB4 win proves a uniquely useful quantized reasoning process.

## 3. Length and trajectory structure

Across all 400 test examples:

- FP16 mean length: 152.76 tokens;
- BNB4 mean length: 165.11 tokens;
- paired mean difference: +12.35 tokens;
- bootstrap 95% CI: +4.21 to +20.72;
- BNB4 total-token ratio: 1.081×;
- BNB4 is longer in 59.75% of non-tied pairs.

The clean subset reduces the mean difference to +4.84 tokens with a confidence
interval spanning zero. The most interesting clean contrast is outcome-dependent:

| Clean outcome | n | Mean BNB4 − FP16 tokens |
|---|---:|---:|
| Both correct | 186 | +8.08 |
| FP16 only | 47 | −14.30 |
| BNB4 only | 25 | +21.72 |
| Both wrong | 67 | +3.00 |

The BNB4-rescue versus BNB4-induced-failure contrast is +36.02 tokens, bootstrap 95%
CI +2.73 to +67.56. This is compatible with an exploration/commitment interpretation:
successful perturbations extend the trajectory, while harmful perturbations sometimes
commit earlier.

However, the structure proxies do not establish “more reasoning”:

- clean BNB4 rescues add only +0.20 lines on average;
- they contain 0.56 fewer detected arithmetic expressions;
- the self-correction-marker difference is only +0.04;
- final-answer position moves only about +1.44 percentage points later.

The safe conclusion is **trajectory-length and commitment shift**, not deeper or
better reasoning. This distinction matters because recent work already reports
reasoning-token inflation under quantization.

## 4. First divergence

Independent FP16 and BNB4 greedy traces frequently diverge at token zero, before an
arithmetic expression appears. Median normalized first-divergence position is zero in
every outcome group.

This does not localize a causal reasoning failure. Small formatting or lexical changes
can make independent autoregressive trajectories incomparable. A valid mechanistic
follow-up would use a shared prefix and compare both models' logits, token ranks, and
counterfactual continuation outcomes at the same state.

## 5. Utility-aware routing

The practical action is an FP16 rerun. Its signed utility is:

`correct(FP16) - correct(BNB4)`

The controller must distinguish beneficial FP16-only cases from harmful BNB4-only
cases. At a 10% maximum rerun budget:

| Policy | Accuracy | Beneficial selected | Harmful selected |
|---|---:|---:|---:|
| Always BNB4 | 63.00% | — | — |
| Length threshold | 65.00% | — | — |
| Best learned router | 65.50% | 12 | 2 |
| Entropy threshold | **66.00%** | — | — |
| Always FP16 | **69.00%** | — | — |
| Oracle | 73.00% | 40 | 0 |

The best learned model is the all-feature random forest:

- beneficial-switch ROC-AUC: 0.647;
- PR-AUC: 0.236, versus 14.25% beneficial prevalence;
- beneficial Brier score: 0.131;
- expected calibration error: 0.094.

At 20%, the learned router reaches 66.50%; the oracle reaches 77.25% using only 57
beneficial interventions. The oracle opportunity is large, but the cheap controller
does not realize it.

### Interpretation

- **Oracle feasibility:** strong.
- **Learned practical routing:** failed the predeclared baseline test.
- **Adaptive-system claim:** unsupported.
- **Potential abstention:** useful for selective reliability, but not evidence that
  precision routing itself is the right intervention.

## 6. Early-token feasibility

| Observed BNB4 prefix | Best beneficial ROC-AUC | Best PR-AUC | Accuracy with 10% reruns |
|---:|---:|---:|---:|
| 16 tokens | 0.502 | 0.152 | 64.00% |
| 32 tokens | 0.572 | 0.193 | 64.75% |
| 64 tokens | 0.639 | 0.278 | 64.75% |

The 16-token classifier is chance-level. Some signal appears by 64 tokens, but the
result remains below entropy routing and always-FP16. Current data does not justify
token-level precision kernels or layer switching.

## 7. Is quantization just temperature?

The temperature control uses the first 100 test prompts. Greedy accuracy on this
subset matches the full-test rates: 69% FP16 and 63% BNB4. It contains 9 BNB4-only
rescues and 15 FP16-only cases.

| Condition | Per-completion accuracy | Empirical pass@3 | Majority accuracy | Mean tokens |
|---|---:|---:|---:|---:|
| FP16 greedy | 69.00% | — | — | 145.16 |
| BNB4 greedy | 63.00% | — | — | 168.99 |
| FP16 T=0.3 | 66.00% | 83.00% | 67.00% | 156.33 |
| FP16 T=0.7 | 69.67% | 89.00% | 70.00% | 149.90 |

### BNB4-rescue overlap

| Temperature | BNB4 rescues reproduced by any sample | Reproduced by majority | Any-sample rescue Jaccard |
|---:|---:|---:|---:|
| 0.3 | 6/9 | 1/9 | 33.33% |
| 0.7 | **9/9** | **5/9** | 37.50% |

At T=0.7, the individual FP16 completion accuracy on the nine BNB4-only prompts is
59.26%. Thus these prompts are not generally inaccessible to FP16; greedy decoding
selected a bad trajectory, while ordinary sampling frequently finds the correct one.

The equivalence is incomplete:

- majority vote reproduces only five of nine BNB4 rescues;
- the any-sample rescue set is broader than the BNB4 set, producing low Jaccard;
- FP16 T=0.7 averages 19.09 fewer tokens than BNB4 greedy;
- only 5/15 BNB4-induced failures are also majority failures at T=0.7.

A grouped correctness model using temperature, sampling status, prompt length,
generation length, entropy, and margin obtains ROC-AUC 0.7868. Adding precision mode
changes ROC-AUC by −0.0009 and log loss by +0.0004. This associational diagnostic
finds no incremental predictive value from precision after the included controls.

### Interpretation

H3 is **weakened, not proven false in every mechanistic sense**. Precision produces a
distinct deterministic path and length signature, but the correctness complementarity
is largely not unique to precision. This removes the strongest motivation for using
precision instead of cheaper, simpler sampling or self-consistency baselines.

## 8. Systems result

On the RTX 3090 test run:

| Metric | FP16 | BNB4 | BNB4 relative result |
|---|---:|---:|---:|
| Mean tokens/second | 54.09 | 36.64 | −32.3% |
| Mean example latency | 2.82 s | 4.50 s | +59.5% |
| Mean generated tokens | 152.76 | 165.11 | +8.08% |

Bitsandbytes BNB4 may save resident weight memory, but no peak-VRAM measurement was
recorded, and batch-one generation is slower. A BNB4-then-FP16-rerun policy is not an
efficiency win on this setup. The project must not claim speed or a favorable
latency–accuracy trade-off.

## 9. Hypothesis verdicts

| Hypothesis | Verdict | Reason |
|---|---|---|
| H1: precision changes trajectories non-monotonically | **Supported in this setting** | Both FP16-only and BNB4-only wins survive cleaning |
| H2: quantization changes exploration/commitment | **Partial** | Signed clean length contrast exists; reasoning-structure proxies are weak |
| H3: quantization is distinct from temperature | **Weakened** | FP16 T=0.7 covers 9/9 BNB4 rescues with any of three samples; precision adds no controlled predictive value |
| H4: routing can beat static precision | **Oracle only** | Oracle is strong; learned router loses to entropy and always-FP16 |
| Early token control is feasible | **Not supported** | Chance at 16 tokens and weak utility by 64 |
| Adaptive inference is efficient | **Not supported** | BNB4 is slower and no peak-memory policy measurement exists |

## 10. Novelty after the results

The broad claims are already crowded:

- [FlexQuant](https://arxiv.org/abs/2506.12024) dynamically switches token/layer
  precision using entropy and divergence signals.
- [DP-LLM](https://arxiv.org/abs/2508.06041) performs dynamic layer-wise precision
  assignment with lightweight error estimation.
- [Quantization Meets Reasoning](https://arxiv.org/abs/2505.11574) localizes early
  reasoning failures and performs targeted recovery.
- [Extreme Low-Bit Inference](https://arxiv.org/abs/2606.02011) detects generation
  pathologies and selectively invokes FP16 planning/rescue.
- [ReQAT](https://arxiv.org/abs/2606.15682) targets critical symbolic commitments in
  4-bit reasoning.
- [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) studies token
  inflation and trace changes across multiple reasoning domains.

Therefore, none of the following is sufficiently novel alone:

- quantization hurts reasoning;
- quantization sometimes lengthens reasoning;
- entropy predicts failures;
- dynamic precision can recover quantized models;
- quantized and FP models occasionally solve different examples.

### Surviving differentiated direction

The most promising broader direction is:

> **Reasoning stability under numerical and decoding perturbations:** characterize
> when quantization, rounding, precision, and sampling temperature move a model into
> the same or different semantic solution basins.

This reframes precision as one intervention in a controlled perturbation family, not
as a privileged controller. A substantial thesis would need:

1. a factorial experiment over precision, quantizer, rounding seed, temperature, and
   model family;
2. shared-prefix counterfactual branching rather than independent-string divergence;
3. semantic strategy and arithmetic-state alignment;
4. causal estimates of when perturbations rescue or corrupt a trajectory;
5. comparison against self-consistency, best-of-N, verifier-guided selection, and
   abstention at equal compute cost;
6. replication across GSM8K, MATH, code, and at least two model families.

That could become a good PhD chapter or paper sequence. The current experiment alone
is a credible pilot or empirical thesis, not yet the full contribution.

## 11. How good is the project right now?

| Dimension | Assessment |
|---|---|
| Experimental hygiene | **Good**: paired runs, clean subset, held-out split, bootstrap/McNemar, artifact logging |
| Empirical signal | **Good**: statistically credible two-way trajectory churn |
| Mechanistic explanation | **Weak-to-moderate**: length signal, but no causal localization |
| Learned intervention | **Weak**: loses to entropy and static FP16 |
| Systems contribution | **Weak**: BNB4 slower; memory not measured |
| Original broad novelty | **Low**: substantial overlap with current dynamic-precision literature |
| Narrow empirical novelty | **Moderate**: precision-versus-temperature rescue overlap is interesting but small-scale |
| Master's thesis readiness | **Strong**, with honest empirical framing |
| Standalone PhD-thesis readiness | **Insufficient** without broader perturbation/stability program |

## 12. Final next experiment

Run exactly one pre-registered replication before deciding whether to close or expand
the project:

- model: `Qwen/Qwen2.5-3B-Instruct`;
- dataset: the same 200 GSM8K test prompts first, to isolate model scale;
- conditions: FP16 greedy, BNB4 greedy, FP16 T=0.7 with three samples;
- primary outcomes: clean FP16-only and BNB4-only counts, oracle gain, BNB4-rescue
  any/majority temperature coverage, and token-length contrast;
- no controller retraining until the paired and temperature effects are inspected.

### Decision after replication

- **Expand into reasoning-stability research** only if BNB4-specific rescues remain
  after the temperature control and trajectory signatures replicate.
- **Publish/submit as a compact empirical negative result** if greedy churn and length
  shifts replicate but temperature again explains most correctness complementarity.
- **Close the direction** if BNB4-only wins or the signed length contrast collapse.

## Bottom line

The project found something real, but not the thing originally hoped for. Precision
changes deterministic reasoning trajectories; it has not been shown to be a uniquely
useful or learnably controllable reasoning knob. The honest contribution is now a
careful study of reasoning stability under numerical perturbation, with temperature
as the crucial alternative explanation.
