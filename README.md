# Precision as a Control Variable for Reasoning — research MVP

This repository tests whether numerical precision changes a language model's
reasoning trajectory in a way that is distinct from ordinary decoding noise. It
also tests whether the signed value of switching precision can be predicted.
The implemented pipeline now covers:

1. matched full-precision and quantized deterministic evaluation;
2. four-way correctness/answer-flip labeling;
3. token entropy, probability, surprisal, and top-1/top-2 margin logging;
4. judge-free trajectory structure, paired length, entropy, and divergence analyses;
5. four-outcome utility routing that counts both helpful and harmful FP reruns;
6. 16/32/64-token prefix prediction for early-intervention feasibility;
7. an FP16 temperature-sampling control and an evidence-gated decision memo.

The current directed sequence and exact RTX 3090 commands are in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).

The visual project report is in [`docs/index.html`](docs/index.html). Serve the
repository root with `python -m http.server 8000`, then open
`http://localhost:8000/docs/`.

For a from-scratch technical explanation, read
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). The audited Stage 1 evidence and
novelty assessment are in [`docs/RESULTS_AUDIT.md`](docs/RESULTS_AUDIT.md).
The focused paired analysis of whether BNB4 generates longer solutions is in
[`docs/TOKEN_LENGTH_STUDY.md`](docs/TOKEN_LENGTH_STUDY.md).

The old binary target was **FP correct, quantized wrong** among FP-correct examples.
The current controller uses all four paired outcomes and predicts signed utility:
`correct(FP16) - correct(BNB4)`. FP/Q disagreement remains analysis-only because it
requires running both models and is not a cheap quantized-only routing feature.

## Current decision

The present evidence says **continue, but narrow scope**. FP16 and BNB4 solve
meaningfully different subsets, and a per-example oracle reaches 77.25% versus
69% FP16 and 63% BNB4. However, the learned utility router reaches only 65.5% at a
10% rerun budget, below entropy (66%) and always-FP16. Early-prefix prediction is
weak. The strong “precision as a control variable” claim therefore remains a
hypothesis until the temperature control and one independent replication pass.

Read [`results/DECISION_MEMO.md`](results/DECISION_MEMO.md) first.

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
  --no-tiny --limit 200 --max-new-tokens 256 \
  --log-every 1 --checkpoint-every 10

python -m scripts.run_quant_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --quantization bnb4 --no-tiny --limit 200 --max-new-tokens 256 \
  --log-every 1 --checkpoint-every 10

python -m scripts.compare_fp_quant
python -m scripts.log_features
python -m scripts.train_predictor
python -m scripts.run_oracle_recovery
```

Use the same seed, model ID, dataset split, prompt, and decoding settings for both
runs. The evaluator is greedy by construction. Each run saves model/backend metadata
so fake quantization cannot be confused with a bitsandbytes result.

Evaluation logs model/dataset loading and per-example progress, including ETA,
generation speed, and running accuracy. Partial JSONL outputs are atomically saved
every 10 examples by default. Use `--log-every 5`, `--checkpoint-every 25`, or
`--quiet` to tune this behavior.

## Outputs

- `runs/fp_outputs.jsonl`, `runs/quant_outputs.jsonl`: generations, extracted answers,
  correctness, latency, token counts, and model metadata.
- `runs/comparison.jsonl`, `runs/summary.json`: matched four-way labels and accuracy,
  induced-failure, rescue, flip-rate, and critical-failure metrics.
- `runs/token_features.parquet`: aligned quantized token traces and FP token comparison.
- `runs/example_features.parquet`: cheap aggregate predictor inputs.
- `runs/predictor_metrics.json`, `runs/models/*.joblib`: model/baseline metrics and fitted models.
- `runs/predictor_predictions.parquet`: per-example held-out sensitivity scores.
- `runs/diagnostics.json`: truncation, strict-answer, and extraction-quality checks.
- `runs/figures/*.{png,pdf}`: standardized accuracy, failure-concentration,
  predictor, controller, oracle, and generation-quality figures.
- `notebooks/analysis.ipynb`: summary tables and an entropy-score/failure-probability
  curve saved as `runs/sensitivity_vs_failure.png`.
- `results/trajectory/`: paired trajectory records, structure metrics, and summaries.
- `results/utility_controller/`: signed utility models, predictions, and budget curves.
- `results/prefix_prediction/`: 16/32/64-token early-control evaluation.
- `results/figures/`: trajectory, utility, calibration, and prefix figures.
- `results/temperature/`: temperature outputs and overlap analysis after the 3090 run.
- `results/DECISION_MEMO.md`: automatically generated continue/pivot recommendation.

## Analyze the existing 800 paired examples

```bash
python -m scripts.analyze_existing_runs
python -m scripts.train_utility_controller
python -m scripts.run_prefix_prediction
python -m scripts.plot_trajectory_analysis
python -m scripts.plot_control_analysis
python -m scripts.generate_decision_memo
```

`generate_decision_memo` automatically rebuilds missing audit, trajectory, token-
length, utility, and prefix artifacts from `runs/gsm8k_train/` and
`runs/gsm8k_test/`. This preparation is analysis-only and does not load a language
model. Use `--no-prepare-missing` for strict artifact checking.

## RTX 3090: decisive temperature control

This is the next run. It produces 600 completions: 100 prompts × two temperatures ×
three samples. It checkpoints and prints progress throughout.

```bash
python -m scripts.run_temperature_baseline \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype fp16 \
  --dataset-split test --limit 100 --offset 0 \
  --temperatures 0.3,0.7 --samples 3 \
  --max-new-tokens 512 --checkpoint-every 20 --log-every 1

python -m scripts.analyze_temperature_experiment
python -m scripts.plot_temperature_analysis
python -m scripts.generate_decision_memo
```

The central comparison is not sampled accuracy alone. Inspect how much the FP16
temperature rescue set overlaps the deterministic BNB4 rescue set. High overlap
weakens the precision-specific thesis; low overlap supports it.

## Current scope and next gate

Do not add custom kernels or layer switching yet. The next gates are the temperature
control and one independent model/dataset/quantizer replication. Current prefix
features do not justify token-level systems work.

Run logic tests with:

```bash
python -m unittest discover -s tests -v
```
