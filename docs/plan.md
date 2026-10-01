# Plan to improve on the baseline

All proposals below are hypotheses to test. The first goal is a reproducible improvement over the strongest reproduced baseline on the same held-out problems, followed by confirmation on Codabench.

## Experiment sequence

| Stage | Experiment | Decision gate |
|---|---|---|
| E0 | Always True/False; train-only model/effort priors; cheap text-length/difficulty controls | Establish class imbalance and shortcut strength. |
| E1 | Reproduce linear probe, RBF kernel and uncertainty references | Same folds, labels, inference settings and runtime measurement; quantify unsupported models. |
| E2 | Regularized logistic probes on selected prompt layers; compare last-token and masked mean pooling | Beat E1 on held-out families; reject layer sweeps that only improve tuning folds. |
| E3 | Separate per-model heads; effort-aware calibration with shared/shrunk calibration when data is sparse | Verify gains per model and aggregate; avoid fitting tiny effort subsets. |
| E4 | Add short trace uncertainty: early/middle/late entropy, slope, spikes, truncation and completion flags | Incremental gain versus E2 at matched runtime; compare 128/256/512 token budgets. |
| E5 | Out-of-fold stacking of probe and trace scores using a small regularized classifier | Beat each component on an untouched holdout; no training on in-sample component scores. |
| E6 | Optional cheap equivalence-preserving perturbations and representation stability | Keep only if validated semantic equivalence and accuracy gain justify extra forward passes. |

Prioritize E0–E3. Delay sparse autoencoders, transcoders and large-scale test-time resampling until a cheap system works. They increase engineering and compute requirements without evidence they will help this dataset.

A reasonable candidate uses a fast probe for every case and trace features only on uncertain cases. Calibrate and validate the entire routing policy, including its time cutoff; selecting only hard cases changes the distribution seen by the second stage. Use the fast prediction as an explicit, measured time-budget fallback.

## Validation that can support a claim

1. Freeze source revisions and audit labels, model IDs, reasoning effort, class balance, repeated texts and problem-family IDs. Keep official robustness booleans when supplied; do not invent the organizers' hidden rule from decay.
2. Group all variants and all model/effort rows from one underlying problem into the same fold. Normalize exact duplicates and manually inspect near duplicates. Where possible group common templates too.
3. Reserve a family-disjoint final local holdout before tuning. Use nested grouped CV on the remaining data for layer, regularization, threshold and ensemble selection. Fit scaling/PCA only on training folds.
4. Optimize plain accuracy, with balanced accuracy, confusion matrices and per-model/per-effort accuracy as diagnostics. Report train-derived majority controls.
5. Compute paired confidence intervals by resampling independent problem families. With very few families, report uncertainty and avoid declaring victory from one or two flipped cases.
6. Include shuffled-label and text-only controls to expose accidental leakage or superficial difficulty prediction. A useful classifier need not establish a causal mechanism; make the scientific claim match the experiment.
7. Promote a candidate only when improvement survives the untouched holdout, full model coverage and complete offline timing. Aim for a paired interval above zero; if data cannot support it, label the gain provisional and seek more independent data.

Extra labeled data can be generated during training. Proposed process: collect licensed public math problems disjoint from validation families, build mathematically checked equivalent variants, sample each actual target checkpoint/configuration, and record correctness and perturbation stability. Use the resulting labels as noisy auxiliary supervision, not as the undisclosed official criterion. Reserve an independent family split before generation. Estimate cost on a pilot before scaling.

## Compute and deployment

Training needs CUDA GPU access and storage for checkpoint caches. Profile the smallest model first, sequentially loading checkpoints; prompt-only features are much cheaper than long generated traces. No training-hour or dollar estimate is defensible until problem count, token lengths and measured throughput are known.

Use this planning inequality:

`sum(model_load_seconds + prompt_forward_seconds + optional_decode_seconds) + overhead < 3000 seconds`

The 3,000-second target is our proposed safety margin, not an organizer rule. For N cases and measured non-generation overhead H, average remaining budget is `(3000-H)/N`. Sweep realistic case counts until organizers confirm N. Share one elapsed-time budget across successive groups; do not reset it per call. Avoid letting early groups consume all optional generation time.

Cache extraction by checkpoint revision, effort, template hash, problem hash, layer, token cap and precision. Do not retain GPU models for all groups simultaneously. Test quantized gpt-oss loading separately rather than forcing ordinary `.to()` behavior onto it.

Validate the final ZIP in the actual offline image: exact booleans/order, empty batch, repeated prompts, long inputs, every checkpoint/effort, no missing artifacts, no downloads, stable repeated predictions, complete-run time and peak memory. Use the official harness for scores. Keep a known-good bundle and its SHA-256.

## October execution schedule

- October 1–4: accounts, data/model coverage audit, compute access, smoke submission and baseline reproduction setup.
- October 5–11: freeze folds, extract features, reproduce E0/E1; identify data gaps before broad sweeps.
- October 12–18: E2/E3 and controlled E4 experiments; collect auxiliary data only if pilot costs justify it.
- October 19–24: E5/E6 only when warranted; full-track timing and limited leaderboard checks.
- October 25–29: freeze best validated candidate, offline reproduction and report draft.
- October 30–31: submit verified bundles with time for retry. Do not rely on an unconfirmed November 1 timezone.
- November 2–15: finish reproducible report and release materials; prepare peer reviews.

Deliverables needed for a real entry: trained artifacts for all selected models, data provenance, reproducible feature/training scripts, family-level evaluation report, measured offline bundle and Codabench acceptance. None of these trained-system milestones is claimed complete by this planning repository.
