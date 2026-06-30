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

### Stage 1 decision gate

Continue if most of the following hold on the locked test split:

- ROC-AUC is at least 0.70 and PR-AUC clearly exceeds positive prevalence.
- The learned controller beats entropy at a 10% intervention budget.
- Net end-to-end accuracy improves after harmful FP interventions are counted.
- The signal remains after excluding malformed/truncated generations.

## Stage 2 — precision dose response

Run BNB8 on the same train/test examples. Reuse the existing FP outputs; do not rerun
them. Compare BNB8 and BNB4 induced-failure rates, predictability, and oracle upside.
This tests whether sensitivity scales coherently with quantization severity.

## Stage 3 — model replication

Only if Stage 1 succeeds, repeat 200 train + 200 test examples with
Qwen2.5-3B-Instruct. Do not move to A40/A100 yet.
