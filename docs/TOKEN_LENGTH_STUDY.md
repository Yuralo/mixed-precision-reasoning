# Token-length study: does quantization make the model “think” longer?

## Question

The paired experiment contains cases where BNB4 is correct and FP16 is wrong. Does
the quantized model generate more tokens in those cases, suggesting that quantization
causes a longer reasoning trajectory?

This study compares generated-token counts for the same 400 held-out GSM8K prompts.
Each example contributes a paired difference:

```text
token delta = BNB4 generated tokens - FP16 generated tokens
```

Positive values mean that BNB4 produced a longer response. Token count measures
output length, not reasoning depth. Extra tokens may represent useful intermediate
work, verbosity, repetition, instability, or delayed commitment.

## Overall result

| Metric | Result |
|---|---:|
| FP16 mean generated tokens | 152.76 |
| BNB4 mean generated tokens | 165.11 |
| Mean paired increase | +12.35 tokens |
| Bootstrap 95% CI | [+4.21, +20.72] |
| Ratio of total BNB4/FP16 tokens | 1.081× |
| BNB4 longer | 239/400, 59.75% |
| BNB4 shorter | 150/400, 37.50% |
| Equal | 11/400, 2.75% |
| Sign-test p-value, excluding ties | 0.0000075 |

BNB4 produced significantly more tokens overall. The total token increase was 8.08%.

## Results by correctness outcome

| Paired outcome | n | FP16 mean | BNB4 mean | Mean delta | BNB4 longer |
|---|---:|---:|---:|---:|---:|
| Both correct | 219 | 131.05 | 138.98 | +7.93 | 61.6% |
| FP16 correct, BNB4 wrong | 57 | 187.44 | 202.98 | +15.54 | 50.9% |
| FP16 wrong, BNB4 correct | 33 | 154.39 | 173.00 | +18.61 | 63.6% |
| Both wrong | 91 | 182.71 | 201.42 | +18.70 | 59.3% |

The 33 BNB4 rescue cases were longer by 18.61 tokens on average, and BNB4 was longer
in 21 of 33 cases. However, the bootstrap interval for that group is wide
[-11.64, +51.76], and both-wrong examples show nearly the same mean increase. The full
sample does not establish that extra tokens specifically cause rescues.

## Clean-subset analysis

The clean subset includes 325 examples where both outputs were explicit and neither
hit the token ceiling. Across all clean examples, BNB4 generated 3.29% more total
tokens. Its mean increase was +4.84 tokens, with a bootstrap interval of
[-1.16, +10.69]. BNB4 was still longer more often—190 versus 127 examples—but extreme
truncated traces explain part of the larger full-sample inflation.

The outcome-specific clean results are more revealing:

| Clean outcome | n | Mean token delta | Median delta | BNB4 longer |
|---|---:|---:|---:|---:|
| Both correct | 186 | +8.08 | +8 | 60.8% |
| FP16 correct, BNB4 wrong | 47 | **−14.30** | −9 | 42.6% |
| FP16 wrong, BNB4 correct | 25 | **+21.72** | +24 | 68.0% |
| Both wrong | 67 | +3.00 | +9 | 59.7% |

The difference between clean BNB4 rescues and clean quantization-induced failures is
36.02 tokens. A bootstrap 95% interval is [+2.73, +67.56].

This supports an exploratory **deliberation-versus-premature-commitment hypothesis**:

- when BNB4 rescues an FP16 mistake, it often follows a longer trajectory;
- when BNB4 corrupts an FP16-correct answer, its clean trace is often shorter.

This contrast was discovered after inspecting the outcomes and has not been corrected
for multiple exploratory comparisons. It requires preregistered replication on new
examples, another model, and another quantizer before becoming a thesis claim.

## Accuracy by length-change regime

| BNB4 length change | n | FP16 accuracy | BNB4 accuracy | Rescues | Failures |
|---|---:|---:|---:|---:|---:|
| At least 25 tokens shorter | 83 | 68.67% | 53.01% | 6 | 19 |
| 1–24 tokens shorter | 67 | 70.15% | 65.67% | 6 | 9 |
| Equal | 11 | 72.73% | 72.73% | 0 | 0 |
| 1–24 tokens longer | 109 | 73.39% | 69.72% | 7 | 11 |
| At least 25 tokens longer | 130 | 64.62% | 61.54% | 14 | 18 |

Strong shortening is associated with the largest BNB4 accuracy loss. But very long
BNB4 outputs are not automatically good: they contain both 14 rescues and 18 induced
failures, and both models have lower accuracy in that regime. Long traces may also
indicate intrinsically harder prompts.

## Interpretation

The supported conclusion is:

> BNB4 changes generation length and produces modest token inflation overall. In the
> clean data, successful BNB4 rescues are substantially longer relative to their FP16
> counterparts than BNB4-induced failures are.

The unsupported conclusion is:

> Quantization makes the model reason more deeply, and this deeper reasoning causes
> correct answers.

Token count alone cannot distinguish useful computation from verbosity or instability.
To test causality, the next study should measure the number of distinct reasoning
steps, repetition, the first FP/BNB4 divergence, the first incorrect step, and whether
forcing additional tokens changes short BNB4 failures into correct answers.

## Reproduction

```bash
uv run --with matplotlib python -m scripts.study_token_inflation
```

Artifacts:

- `runs/gsm8k_test/token_length_study.json`
- `runs/gsm8k_test/token_length_pairs.csv`
- `runs/gsm8k_test/figures/11_token_length_by_outcome.png`
- `runs/gsm8k_test/figures/11_token_length_by_outcome.pdf`
- `runs/gsm8k_test/figures/12_clean_token_delta.png`
- `runs/gsm8k_test/figures/12_clean_token_delta.pdf`
