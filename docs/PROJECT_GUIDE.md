# Project guide: Precision as a Control Variable for Reasoning

For the consolidated result tables, hypothesis verdicts, novelty assessment, and
continue/pivot recommendation, read [`FINAL_ASSESSMENT.md`](FINAL_ASSESSMENT.md).

## 1. The problem

Quantization stores model weights or activations with fewer bits. It reduces memory,
but numerical perturbations can change token probabilities and send autoregressive
generation down a different reasoning path. A small early change may cascade into a
wrong final answer.

The project now asks a stronger but more falsifiable question than “does
quantization hurt accuracy?”:

> Does numerical precision alter reasoning trajectories in a way that is distinct
> from ordinary decoding randomness, and can those changes be predicted or used?

The control hypothesis is that precision changes exploration, commitment, and
self-correction—not merely numerical fidelity. The intended policy has three actions:

- continue at low precision;
- use high precision for a sensitive example, token, or layer;
- abstain when reliable recovery is too expensive or uncertain.

The MVP currently implements example-level detection and accounting. It does **not**
yet switch individual tokens or layers during real inference.

## 2. Experimental definitions

Every prompt is run once with FP16 weights and once with bitsandbytes 4-bit weights
using greedy decoding. The final numeric answers define four paired outcomes:

| Label | Meaning | Intervention utility |
|---|---|---:|
| FP correct, Q correct | Both models succeed | 0 |
| FP correct, Q wrong | Quantization-induced failure | +1 |
| FP wrong, Q correct | Quantization rescue | -1 |
| FP wrong, Q wrong | Neither succeeds | 0 |

The `+1/-1` distinction matters. Blindly replacing every quantized answer with FP can
destroy correct quantized answers. A deployable controller should predict **signed
counterfactual utility**, not merely whether an example looks difficult.

Three evaluation populations are used:

1. **Sensitivity prediction population:** FP-correct examples only. Positive means
   quantized wrong; negative means quantized correct. This directly tests whether
   quantization-induced failures are predictable.
2. **Controller population:** all examples. This measures net end-to-end accuracy
   after both helpful and harmful FP interventions are counted.
3. **Early-control population:** all examples, but features are truncated after 16,
   32, or 64 quantized tokens. This tests whether a decision is possible before the
   whole output and its correctness are already observable.

## 3. Model and hardware

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- FP reference: FP16 on CUDA
- Quantized model: bitsandbytes NF4 4-bit weights with FP16 compute
- Hardware: one NVIDIA RTX 3090
- Task: GSM8K arithmetic word problems
- Stage 1 data: first 400 examples from official train split and first 400 examples
  from official test split
- Decoding: greedy, maximum 512 new tokens
- Prompt: concise reasoning with an explicit `FINAL_ANSWER: <number>` request
- Answer-aware stopping: enabled with a four-token grace period

The BNB4 model is weight-only quantized. These experiments are not W4A4 or KV-cache
quantization experiments.

## 4. Software pipeline

```text
dataset + prompt
      |
      +--> FP16 generation --------+
      |                            |
      +--> BNB4 generation --------+--> answer extraction
                                           |
                                           +--> paired correctness labels
                                           |
token entropy, margin, probability --------+--> example feature aggregation
                                                   |
                                                   +--> predictor training
                                                   +--> oracle/controller curves
                                                   +--> figures and audit
```

Important modules:

| File | Responsibility |
|---|---|
| `src/load_model.py` | FP, fake, BNB4, and BNB8 model loading |
| `src/datasets.py` | GSM8K loading and prompt construction |
| `src/generation.py` | Greedy generation and token telemetry |
| `src/answer_extraction.py` | Numeric and explicit-answer parsing |
| `src/compare_outputs.py` | Four-way paired labels |
| `src/feature_logging.py` | Token alignment and example aggregation |
| `src/train_sensitivity_model.py` | Predictors, baselines, and intervention curves |
| `src/oracle_recovery.py` | Correct at-most-budget oracle |
| `src/diagnostics.py` | Truncation and answer-quality checks |
| `src/visualization.py` | Standard PNG/PDF figure bundle |
| `scripts/audit_results.py` | Dependency-light paired statistical audit |
| `scripts/study_token_inflation.py` | Paired output-length study by correctness outcome |
| `src/trajectory_metrics.py` | Judge-free reasoning structure and commitment proxies |
| `src/paired_analysis.py` | Paired trajectory records and first-divergence summaries |
| `src/utility_controller.py` | Four-outcome signed-utility routing and calibration |
| `src/prefix_prediction.py` | Early quantized-prefix feasibility analysis |
| `src/temperature_experiment.py` | Temperature/quantization success-set overlap |
| `src/decision_memo.py` | Evidence-gated framing and continue/pivot recommendation |

## 5. Runtime features

The current controller uses only quantized-generation aggregates:

- maximum and mean next-token entropy;
- minimum and mean top-1/top-2 logit margin;
- minimum and mean selected-token probability;
- prompt length;
- generated length.

Generated length and aggregate statistics are known only after low-precision
generation completes. The current controller is therefore a **rerun/abstention
controller**, not an early token-level switching controller. This distinction must be
preserved in every claim.

FP/Q token disagreement is stored for analysis but deliberately excluded from the
cheap controller because it requires running FP.

## 6. Models and baselines

Learned predictors:

- class-balanced logistic regression;
- class-balanced random forest.

Baselines:

- maximum-entropy ranking;
- minimum-margin ranking;
- seeded random ranking;
- always quantized;
- always FP16;
- perfect paired oracle.

Metrics:

- ROC-AUC and PR-AUC with 500-sample bootstrap intervals;
- precision and recall at 5%, 10%, and 20% intervention budgets;
- end-to-end accuracy at 0%, 5%, 10%, 20%, 50%, and 100% FP budgets;
- number of beneficial and harmful interventions selected;
- paired accuracy difference and exact McNemar test;
- answer flip rate, token inflation, latency, and tokens/second.

The utility controller additionally reports beneficial-switch precision/recall,
harmful-switch rate, avoided-harm rate, Brier score, calibration error, and
risk–coverage under abstention.

## 7. Current evidence

The held-out 400-example test split gives:

| Quantity | Result |
|---|---:|
| FP16 / BNB4 accuracy | 69.0% / 63.0% |
| FP16-only / BNB4-only wins | 57 / 33 |
| Answer-flip rate | 38.75% |
| Oracle selector | 77.25% |
| Clean FP16-only / BNB4-only wins | 47 / 25 |
| Best learned utility router at 10% | 65.5% |
| Entropy router at 10% | 66.0% |
| Oracle at 10% | 73.0% |

The precision effect is statistically credible for this setting: FP16 minus BNB4 is
6 points with paired bootstrap 95% CI 1.5–10.5 points and exact McNemar p=0.0149.
The success sets are not nested, and the clean subset keeps both directions of flip.

The mechanism and control claims are weaker. On clean outputs, BNB4 rescues are
21.72 tokens longer than FP16 while BNB4-induced failures are 14.30 shorter, but
arithmetic-expression and self-correction proxies do not show that the extra tokens
are better reasoning. At a 10% rerun budget, learned routing does not beat entropy or
always-FP16. Prefix ROC-AUC is 0.50 at 16 tokens, 0.57 at 32, and 0.64 at 64.

## 8. Artifact layout

```text
runs/
  gsm8k_train/
    fp_outputs.jsonl
    quant_outputs.jsonl
    fp_token_features.jsonl
    quant_token_features.jsonl
    comparison.jsonl
    example_features.parquet
  gsm8k_test/
    ...matched test artifacts...
    predictor_metrics.json
    predictor_metrics_clean.json
    predictor_predictions.parquet
    diagnostics.json
    oracle_recovery.json
    figures/
  research_audit.json
results/
  trajectory/
  utility_controller/
  prefix_prediction/
  temperature/
  figures/
  DECISION_MEMO.md
```

`runs/first/` contains the exploratory 200-example experiment. It should not be mixed
with Stage 1 because it used the old long prompt, had severe truncation, and later had
an incomplete default FP file. `runs/pilot_v2/` contains the 20-example prompt and
stopping validation.

## 9. Reproduction

The full staged command sequence is in [`../EXPERIMENT_PLAN.md`](../EXPERIMENT_PLAN.md).
The final Stage 1 audit can be regenerated without pandas:

```bash
python -m scripts.diagnose_runs \
  --fp runs/gsm8k_test/fp_outputs.jsonl \
  --quant runs/gsm8k_test/quant_outputs.jsonl \
  --assumed-max-new-tokens 512 \
  --output runs/gsm8k_test/diagnostics.json

python -m scripts.audit_results \
  --runs-dir runs \
  --output runs/research_audit.json
```

Generate the figure bundle in an environment containing pandas, pyarrow,
scikit-learn, and matplotlib:

```bash
python -m scripts.make_figures \
  --comparisons runs/gsm8k_test/comparison.jsonl \
  --features runs/gsm8k_test/example_features.parquet \
  --predictor-metrics runs/gsm8k_test/predictor_metrics.json \
  --oracle runs/gsm8k_test/oracle_recovery.json \
  --diagnostics runs/gsm8k_test/diagnostics.json \
  --output-dir runs/gsm8k_test/figures
```

Run all new artifact-only analyses without model inference:

```bash
python -m scripts.analyze_existing_runs
python -m scripts.train_utility_controller
python -m scripts.run_prefix_prediction
python -m scripts.plot_trajectory_analysis
python -m scripts.plot_control_analysis
python -m scripts.generate_decision_memo
```

The completed RTX 3090 temperature control was run with:

```bash
python -m scripts.run_temperature_baseline \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype fp16 \
  --dataset-split test --limit 100 --temperatures 0.3,0.7 \
  --samples 3 --max-new-tokens 512 --checkpoint-every 20

python -m scripts.analyze_temperature_experiment
python -m scripts.plot_temperature_analysis
python -m scripts.generate_decision_memo
```

## 10. Decision and scientific boundary

- real token-level or layer-level precision switching;
- a successful controller that beats entropy or static FP16;
- activation norms, outlier ratios, FP/Q KL, or hidden-state distances;
- memory and peak-VRAM measurement;
- a temperature result showing precision-specific correctness beyond FP16 sampling;
- BNB8 precision dose response;
- MATH, critical semantic tasks, or a second model family;
- optimized mixed-precision kernels.

The recommendation is **only continue as empirical analysis**. The defensible framing today is
“Empirical Study of Quantization-Induced Reasoning Trajectory Shifts.” Promote to
“Precision as a Control Variable” only if temperature sampling does not reproduce the
BNB4 rescue/failure pattern and the signed effect replicates independently. Otherwise
keep it empirical or pivot. See [`../results/DECISION_MEMO.md`](../results/DECISION_MEMO.md).

The completed temperature control weakens the strong framing. On 100 prompts,
temperature 0.7 with three FP16 samples covers all nine BNB4-rescue prompts with at
least one correct sample and five of nine by majority vote. Rescue-set Jaccard is
37.5%, so the sets are not identical. The sampled outputs are also 19.09 tokens
shorter than BNB4 greedy. Precision changes the trajectory signature, but the observed
correctness complementarity is not unique to precision.
