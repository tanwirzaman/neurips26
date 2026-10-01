# Baseline audit

## What the references actually establish

| Reference | Mechanism | Evidence and limitation |
|---|---|---|
| Trivial fallback | Always predicts non-robust | Valid interface smoke test; measure class proportions before interpreting accuracy. |
| Representation research baseline | Last input-token representations; linear and RBF kernel prediction of accuracy decay | Provides a reproduction workflow, not a verified private leaderboard target. |
| Submission probe | Linear margins averaged from an artifact | Requires checkpoint-compatible trained artifacts. |
| Uncertainty profiling | One deterministic trace; 14 aggregate token-distribution statistics; fitted decay regressor and threshold | Published grouped OOF accuracy 0.6879, balanced accuracy 0.6262, MAE 0.3429. These are development metrics, not final-track results. |

The [research baseline](https://github.com/aimo-interp/baselines) encodes all layers and compares repeated layer-wise probes and kernels. Its README contains older setup assumptions; use the competition runtime for deployment.

The [uncertainty documentation](https://github.com/aimo-interp/getting-started/blob/main/solutions/uncertainty-profiling/README.md) reports a 500-tree Random Forest, threshold 0.5760, prompt cap 2,048 and output cap 4,096. Its bundled artifact targets DeepSeek-R1-0528-Qwen3-8B; unsupported models return False. It selects using balanced accuracy, whereas competition ranking uses plain accuracy.

## Implementation findings

Inspection of [probe_inference.py](https://github.com/aimo-interp/getting-started/blob/main/solutions/trained-probe/probe_inference.py) shows `_encode_problem` calls `_load_model` for every problem. Missing artifacts return False. The code retains every layer's hidden states, then selects one. These suggest concrete changes:

1. Load once per model group, length-bucket and batch prompts, restore original order, release the checkpoint before switching models. Measure memory and throughput gains.
2. Extract only selected layer outputs where supported; verify hooks across architectures and avoid retaining full sequence activations unnecessarily.
3. Train artifacts for every target model. Treat artifact coverage as a release requirement, not a silent fallback.
4. Keep prompt formatting, layer indexing and token selection identical during training and inference. Padding must not change the selected final non-padding token.

These changes improve efficiency or coverage; none alone proves better robustness prediction.

## Data risk

The [public sample](https://huggingface.co/datasets/aimo-interp/val-sample) currently exposes 28 rows, eight distinct problems and seven model identifiers. It is primarily useful for interface testing, not reliable model selection. The [aggregate sample](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg) has 141 rows and one model identifier. Row count is not independent problem count. Audit family duplication, labels, effort and checkpoint provenance before use.

Do not use base/permuted accuracy, decay, detrimental-permutation counts or supplied perturbation outcomes as prediction features: they are unavailable at inference and encode the target. They may be training targets or provenance only.

No live leaderboard score was recovered: the Codabench response was unusable. There is no justified numerical private-score target yet.
