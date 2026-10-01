# Issue #3: uncertainty-profiling baseline reproduction status

## Outcome

The reference baseline and artifact are pinned and inspected, but the baseline has **not been reproduced** in this workspace. Feature extraction requires the DeepSeek-R1-0528-Qwen3-8B checkpoint and a PyTorch/Transformers runtime. This machine has no NVIDIA runtime or local checkpoint, and the bundled Python lacks PyTorch, Transformers, scikit-learn, joblib, and PyArrow. The evaluation was not started, so inference time, peak memory, prediction coverage, invalid predictions, and OOF predictions from the official artifact are not measured here. No paid compute or package installation was used.

## Pinned reference

| Item | Pin |
| --- | --- |
| Official starter repository | [`aimo-interp/getting-started`](https://github.com/aimo-interp/getting-started/tree/de794053debe75a711400696e66b60937cebbb1b) at commit `de794053debe75a711400696e66b60937cebbb1b` |
| Baseline README blob | `56c6487bad5227c59f589fdb7185fbdbfc59564d` |
| Feature extraction script blob | `2eceea3bf2a520ec463795208bbaf2ca43c796f0` |
| Regressor training script blob | `cbcdc59e91afd3f52366d4dfc6137330ddf5e29e` |
| Inference implementation blob | `0bcffbf32bf9d6bf6d13a17b3c78617a2273b435` |
| Bundled joblib artifact blob | `7d3283e3ef065f5397a512c5a6760722ab05e3c2` |
| Feature dataset | `aimo-interp/augmented-sample-math-agg`, revision `f972ced0705096f8d7ca7fac30825900b8b7fb6a`, config `default`, split `validation` |
| Feature model | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, Hugging Face revision `6e8885a6ff5c1dc5201574c8fd700323f23c25fa` |
| Generation settings | User-only chat template; greedy (`do_sample=false`); 2,048 prompt-token cap; 4,096 generated-token cap; batch size 8; static KV cache; Min-K fraction 0.20; high/low probability cutoffs 0.90/0.10; seed 42 |
| Runtime versions | PyTorch 2.12.1; CUDA 13.2; Accelerate 1.13.0; Hugging Face Hub 1.22.0; joblib 1.5.3; pandas 3.0.3; scikit-learn 1.8.0; tokenizers 0.22.2; Transformers 5.13.0 |

The baseline generates one trace per unique original problem, extracts 14 token-distribution summaries, predicts `absolute_accuracy_decay` with a fitted regressor, and maps scores below threshold 0.5760 to robust. The checked-in artifact is a 500-tree Random Forest (`max_depth=None`, `min_samples_leaf=1`, `max_features=1.0`). The baseline does not use `reasoning_effort` and only has a learned artifact for DeepSeek-R1-0528-Qwen3-8B. Its fallback returns `False` on other models. Thus learned-model coverage is one of five Main Track model IDs and one of four Small Models Track IDs, even though those fallback responses are syntactically valid.

## Reported reference metrics (not independently reproduced)

The upstream README reports five-fold grouped out-of-fold accuracy 0.6879, balanced accuracy 0.6262, and regression MAE 0.3429, with threshold 0.5760. These are the upstream artifact's reported development numbers, not results from this repository's run. The public source has 141 rows and 137 problem IDs.

Our preceding E0 run on the same pinned dataset reported 137 cases and 137 problem-ID groups: the fold-majority control scored 0.7372 accuracy / 0.5000 balanced accuracy, and unigram Naive Bayes scored 0.6934 / 0.5329. This is **not an apples-to-apples comparison**: E0 used a deterministic greedy grouped split with seed 20261001, while the reference trainer uses `StratifiedGroupKFold` with five folds and seed 42. We do not claim the published method beats those controls until both are evaluated on identical folds and inputs.

The official commands to generate missing features and refit are:

```bash
uv run solutions/uncertainty-profiling/scripts/compute_uncertainty_features.py \
  --dataset aimo-interp/augmented-sample-math-agg \
  --revision f972ced0705096f8d7ca7fac30825900b8b7fb6a \
  --feature-model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --local-files-only \
  --cache-dir /path/to/huggingface/cache

uv run solutions/uncertainty-profiling/scripts/train_uncertainty_regressor.py \
  --feature-data-path \
  data/uncertainty-profiling/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B_confidence_features.parquet \
  --n-splits 5 \
  --seed 42
```

These commands have not been run here. The feature parquet and model cache are absent, and the local Python cannot import the required inference/training libraries. A meaningful reproduction should run the feature command on a free compatible GPU environment, then evaluate both the artifact and E0 controls on the same saved `StratifiedGroupKFold` indices. If that requires a paid GPU, stop before provisioning and obtain an explicit budget decision.

## Runtime and alternatives

No official-baseline prediction was run, so runtime and peak-memory measurements are unavailable, not zero. The upstream competition image documents one RTX PRO 6000 with 96 GiB VRAM and a 3,600-second prediction-run limit; that describes the submission runtime, not resources available for feature generation in this workspace. The 8B checkpoint is plausible for that target, but end-to-end time with 137 prompts or the eventual private case count remains unmeasured. The baseline is currently practical only for its single trained model; unsupported models fall back to the non-robust class.

Alternatives inspected: the official representation-probe baseline needs model forward passes and per-model hidden-state artifacts; it is also blocked by the absent checkpoint/runtime here. The CPU text-only controls are fully runnable and have already been recorded, but they do not replace a reproduced model-based reference. The relevant paper, *Tracing Uncertainty in Language Model “Reasoning”*, motivates token-level uncertainty traces; the competition baseline is a simpler 14-feature aggregate adaptation, not a faithful reproduction of that paper's temporal trace-profile method.

## Local preflight evidence

- `nvidia-smi` is unavailable.
- No local Hugging Face Hub model cache was found for the pinned DeepSeek checkpoint.
- The bundled Python had `pandas` and `numpy`; `torch`, `transformers`, `scikit-learn`, `joblib`, and `pyarrow` were absent.
- No model generation, artifact loading, or training command was executed.
- Paid compute: **$0**.

