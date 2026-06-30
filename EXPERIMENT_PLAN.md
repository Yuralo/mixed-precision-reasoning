# Directed experiment plan

Run these stages in order. Do not start the next stage until the previous result has
been inspected.

## Stage 0 — audit the completed 200-example run (no GPU rerun)

```bash
python -m scripts.diagnose_runs --assumed-max-new-tokens 256
python -m scripts.run_oracle_recovery
python -m scripts.make_figures
```

Send back `runs/diagnostics.json`, `runs/oracle_recovery.json`, and the figures in
`runs/figures/`. This tells us whether the 54% answer-flip rate is partly caused by
truncation or fallback answer extraction.

## Stage 0.5 — clean 20-example generation pilot

Before the held-out experiment, verify the revised concise prompt, explicit-answer
parser, and answer-aware stopping on 20 examples. Keep these outputs separate from
the legacy run.

```bash
python -m scripts.run_fp_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --dataset-split test --no-tiny --limit 20 --max-new-tokens 384 \
  --stop-after-answer --output runs/pilot_v2/fp_outputs.jsonl \
  --token-output runs/pilot_v2/fp_token_features.jsonl

python -m scripts.run_quant_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --quantization bnb4 --dataset-split test --no-tiny --limit 20 \
  --max-new-tokens 384 --stop-after-answer \
  --output runs/pilot_v2/quant_outputs.jsonl \
  --token-output runs/pilot_v2/quant_token_features.jsonl

python -m scripts.diagnose_runs \
  --fp runs/pilot_v2/fp_outputs.jsonl \
  --quant runs/pilot_v2/quant_outputs.jsonl \
  --assumed-max-new-tokens 384 --output runs/pilot_v2/diagnostics.json
```

Do not proceed unless both runs contain 20 examples, explicit-answer rate is high,
and token-cap rate is low.

## Stage 1 — held-out BNB4 validation on the RTX 3090

Use 400 official training examples only to fit the sensitivity predictor, then test
once on 400 disjoint official test examples. FP and quantized runs must use identical
generation settings.

### Generate the controller-training split

```bash
python -m scripts.run_fp_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --dataset-split train --no-tiny --limit 400 --max-new-tokens 512 \
  --stop-after-answer --output runs/gsm8k_train/fp_outputs.jsonl \
  --token-output runs/gsm8k_train/fp_token_features.jsonl

python -m scripts.run_quant_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --quantization bnb4 --dataset-split train --no-tiny --limit 400 \
  --max-new-tokens 512 --stop-after-answer \
  --output runs/gsm8k_train/quant_outputs.jsonl \
  --token-output runs/gsm8k_train/quant_token_features.jsonl

python -m scripts.compare_fp_quant \
  --fp runs/gsm8k_train/fp_outputs.jsonl \
  --quant runs/gsm8k_train/quant_outputs.jsonl \
  --output runs/gsm8k_train/comparison.jsonl \
  --summary runs/gsm8k_train/summary.json

python -m scripts.log_features \
  --fp-tokens runs/gsm8k_train/fp_token_features.jsonl \
  --quant-tokens runs/gsm8k_train/quant_token_features.jsonl \
  --comparisons runs/gsm8k_train/comparison.jsonl \
  --token-output runs/gsm8k_train/token_features.parquet \
  --example-output runs/gsm8k_train/example_features.parquet
```

### Generate the locked test split

```bash
python -m scripts.run_fp_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --dataset-split test --no-tiny --limit 400 --max-new-tokens 512 \
  --stop-after-answer --output runs/gsm8k_test/fp_outputs.jsonl \
  --token-output runs/gsm8k_test/fp_token_features.jsonl

python -m scripts.run_quant_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --quantization bnb4 --dataset-split test --no-tiny --limit 400 \
  --max-new-tokens 512 --stop-after-answer \
  --output runs/gsm8k_test/quant_outputs.jsonl \
  --token-output runs/gsm8k_test/quant_token_features.jsonl

python -m scripts.compare_fp_quant \
  --fp runs/gsm8k_test/fp_outputs.jsonl \
  --quant runs/gsm8k_test/quant_outputs.jsonl \
  --output runs/gsm8k_test/comparison.jsonl \
  --summary runs/gsm8k_test/summary.json

python -m scripts.log_features \
  --fp-tokens runs/gsm8k_test/fp_token_features.jsonl \
  --quant-tokens runs/gsm8k_test/quant_token_features.jsonl \
  --comparisons runs/gsm8k_test/comparison.jsonl \
  --token-output runs/gsm8k_test/token_features.parquet \
  --example-output runs/gsm8k_test/example_features.parquet
```

### Fit on train, evaluate once on test, and create figures

```bash
python -m scripts.train_predictor \
  --features runs/gsm8k_train/example_features.parquet \
  --test-features runs/gsm8k_test/example_features.parquet \
  --output-dir runs/gsm8k_test/models \
  --metrics runs/gsm8k_test/predictor_metrics.json \
  --predictions runs/gsm8k_test/predictor_predictions.parquet

python -m scripts.diagnose_runs \
  --fp runs/gsm8k_test/fp_outputs.jsonl \
  --quant runs/gsm8k_test/quant_outputs.jsonl \
  --assumed-max-new-tokens 512 --output runs/gsm8k_test/diagnostics.json

python -m scripts.run_oracle_recovery \
  --comparisons runs/gsm8k_test/comparison.jsonl \
  --output runs/gsm8k_test/oracle_recovery.json

python -m scripts.make_figures \
  --comparisons runs/gsm8k_test/comparison.jsonl \
  --features runs/gsm8k_test/example_features.parquet \
  --predictor-metrics runs/gsm8k_test/predictor_metrics.json \
  --oracle runs/gsm8k_test/oracle_recovery.json \
  --diagnostics runs/gsm8k_test/diagnostics.json \
  --output-dir runs/gsm8k_test/figures
```

As a robustness check, repeat predictor evaluation using only non-truncated examples
where both models emitted an explicit marked answer:

```bash
python -m scripts.train_predictor \
  --features runs/gsm8k_train/example_features.parquet \
  --test-features runs/gsm8k_test/example_features.parquet \
  --quality-filter clean --output-dir runs/gsm8k_test/models_clean \
  --metrics runs/gsm8k_test/predictor_metrics_clean.json \
  --predictions runs/gsm8k_test/predictor_predictions_clean.parquet
```

## Stage 2 — artifact-only trajectory and utility analysis (complete)

These commands reuse the existing 800 paired generations:

```bash
python -m scripts.analyze_existing_runs
python -m scripts.train_utility_controller
python -m scripts.run_prefix_prediction
python -m scripts.plot_trajectory_analysis
python -m scripts.plot_control_analysis
python -m scripts.generate_decision_memo
```

Observed gate result: the paired precision effect survives, but the learned signed-
utility router does not beat entropy at 10%, and 16/32-token prefix prediction is
weak. This blocks token-level systems work for now.

## Stage 3 — precision versus temperature (next RTX 3090 run)

Start with 100 test prompts, two temperatures, and three samples. This is 600 FP16
completions. It is intentionally smaller than the paired run because the immediate
question is success-set overlap, not a final leaderboard number.

```bash
python -m scripts.run_temperature_baseline \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype fp16 \
  --dataset-split test --offset 0 --limit 100 \
  --temperatures 0.3,0.7 --samples 3 \
  --max-new-tokens 512 --checkpoint-every 20 --log-every 1

python -m scripts.analyze_temperature_experiment
python -m scripts.plot_temperature_analysis
python -m scripts.generate_decision_memo
```

Inspect:

- fraction of deterministic BNB4 rescues reproduced by any FP16 sample;
- Jaccard overlap between BNB4 and temperature rescue sets;
- per-completion, pass-at-3, and majority-vote accuracy;
- generated length and answer diversity.

Precision-specific evidence is stronger if BNB4 rescues have low overlap with FP16
temperature rescues and retain a distinct trajectory signature. High overlap weakens
the control-variable claim.

## Stage 4 — one independent replication

Run only after inspecting Stage 3. Prefer one of:

1. a small MATH subset with the same Qwen2.5-1.5B model; or
2. Qwen2.5-3B-Instruct on 200 GSM8K examples; or
3. BNB8/AWQ/GPTQ on the same 200 examples for a quantizer/precision dose response.

The preferred scientific replication is a new dataset. A second GSM8K slice is not
independent enough. Stay on the 3090 unless the chosen model genuinely does not fit.

## Current decision gate

- **Continue strongly** only if temperature does not reproduce the precision effect,
  independent replication preserves two-way success-set churn, oracle routing stays
  materially above both static modes, and routing/abstention becomes useful.
- **Continue narrowly** if the paired phenomenon replicates but learned control stays
  weak. Frame the work as an empirical trajectory study.
- **Pivot** if BNB4 rescues are mostly extraction/truncation artifacts, temperature
  reproduces them, or complementarity collapses in replication.

Do not implement custom CUDA, layer switching, or large A100 sweeps before these
gates. The generated [`results/DECISION_MEMO.md`](results/DECISION_MEMO.md) records
the current recommendation.
