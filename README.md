# Think in Mixed Precision — research MVP

This repository tests whether quantization-induced reasoning failures concentrate in
identifiable uncertainty regimes. The implemented MVP covers:

1. matched full-precision and quantized deterministic evaluation;
2. four-way correctness/answer-flip labeling;
3. token entropy, probability, surprisal, and top-1/top-2 margin logging;
4. example-level aggregation and logistic-regression/random-forest predictors;
5. entropy, margin, and random baselines plus recall at 5/10/20% budgets.

The target is strictly **FP correct, quantized wrong**. Predictor training excludes
FP-wrong examples so the negative class is FP-correct/quantized-correct. FP/Q token
disagreement is retained for analysis, but deliberately excluded from the default
learned controller because it is not a cheap quantized-only runtime feature.

## Install

Use Python 3.10+ from the project directory:

```bash
cd mixed_precision_reasoning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Mac: five-example smoke run

The default model is Qwen2.5-1.5B-Instruct. For a much faster code-path smoke test,
override it with a tiny causal LM. `sshleifer/tiny-gpt2` is useful for plumbing but
will not solve the arithmetic examples.

```bash
python -m scripts.run_fp_eval \
  --model sshleifer/tiny-gpt2 --device cpu --dtype fp32 \
  --tiny --limit 5 --max-new-tokens 16

python -m scripts.run_quant_eval \
  --model sshleifer/tiny-gpt2 --device cpu --dtype fp32 \
  --quantization fake --fake-quant-bits 4 \
  --tiny --limit 5 --max-new-tokens 16

python -m scripts.compare_fp_quant
python -m scripts.log_features
python -m scripts.train_predictor
```

For the actual small model on Apple Silicon, use `--model Qwen/Qwen2.5-1.5B-Instruct
--device mps --dtype fp16`. The `fake` backend rounds Linear weights and dequantizes
them in place. It creates quantization noise and is suitable for Mac debugging, but
it does **not** reduce memory or claim a quantized latency result.

Five examples will often have no quantization-induced failures. In that case,
`predictor_metrics.json` reports `insufficient_class_variation`; that is expected.
Move to 100–500 examples before interpreting AUC or intervention recall.

## RTX 3090: real 4-bit run

Install a CUDA-compatible PyTorch build, then install bitsandbytes:

```bash
pip install bitsandbytes>=0.43

python -m scripts.run_fp_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --no-tiny --limit 200 --max-new-tokens 256

python -m scripts.run_quant_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --quantization bnb4 --no-tiny --limit 200 --max-new-tokens 256

python -m scripts.compare_fp_quant
python -m scripts.log_features
python -m scripts.train_predictor
python -m scripts.run_oracle_recovery
```

Use the same seed, model ID, dataset split, prompt, and decoding settings for both
runs. The evaluator is greedy by construction. Each run saves model/backend metadata
so fake quantization cannot be confused with a bitsandbytes result.

## Outputs

- `runs/fp_outputs.jsonl`, `runs/quant_outputs.jsonl`: generations, extracted answers,
  correctness, latency, token counts, and model metadata.
- `runs/comparison.jsonl`, `runs/summary.json`: matched four-way labels and accuracy,
  induced-failure, rescue, flip-rate, and critical-failure metrics.
- `runs/token_features.parquet`: aligned quantized token traces and FP token comparison.
- `runs/example_features.parquet`: cheap aggregate predictor inputs.
- `runs/predictor_metrics.json`, `runs/models/*.joblib`: model/baseline metrics and fitted models.
- `notebooks/analysis.ipynb`: summary tables and an entropy-score/failure-probability
  curve saved as `runs/sensitivity_vs_failure.png`.

## Current scope and next gate

This first cut deliberately omits hidden-state hooks, full-vocabulary FP-vs-Q KL,
and token-level intervention. First establish enough FP-correct examples and induced
failures for a meaningful predictor test. If entropy/margin features and the learned
model show signal, the next implementation should add selected-layer activation
norms/outlier ratios and logit KL, followed by token-level oracle intervention.

Run logic tests with:

```bash
python -m unittest discover -s tests -v
```
