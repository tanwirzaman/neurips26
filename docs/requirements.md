# Participation requirements

Checked 2026-10-01. Website rules take precedence over the proposal; the starter specifies the runtime.

## Rules and dates

- Predict robustness for a problem/model pair, not merely whether an answer is correct. Ranking uses classification accuracy; speed breaks ties.
- Final submission: November 1, 2026. Confirm the exact closing time and timezone.
- Award candidates: open implementation and reproducible technical report by November 15; review up to three reports November 15–30.
- Additional labeled training data is allowed following the September 19 clarification. Evaluation-data leakage is forbidden. White-box and budget-compliant behavioral methods are permitted.
- Main and Small tracks require separate submissions. Small excludes models at or above 10B parameters.
- Create/sign in to a Codabench account and join the active competition. Email-update registration is a separate form. Compute support is available by proposal, not guaranteed.

Source: [competition rules, timeline and FAQ](https://aimo-interp.github.io/).

## Submission contract

`are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]`

Preserve order and length; return Python booleans. ZIP root needs `solution.py`, helpers and trained artifacts; `small.txt` is required only for Small. No evaluation internet or dependency installation. Do not bundle datasets, credentials or supplied model weights. Validate full coverage and zero invalid predictions in the official offline container.

Workers expose one 96 GB RTX PRO 6000. The 3,600-second prediction limit covers all groups together. Inputs include neither labels nor perturbation metadata. Current checkpoint IDs:

- `Qwen/Qwen3.5-4B`
- `Skywork/Skywork-OR1-Math-7B`
- `allenai/Olmo-3-7B-Think`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- `openai/gpt-oss-120b` (Main only)

Source: [starter interface and evaluation environment](https://github.com/aimo-interp/getting-started).

The [runtime Dockerfile](https://github.com/aimo-interp/getting-started/blob/main/Dockerfile.competition) pins PyTorch 2.12.1/CUDA 13.2, Transformers 5.13.0, scikit-learn 1.8.0, Accelerate 1.13.0, joblib 1.5.3, pandas 3.0.3, tokenizers 0.22.2, safetensors 0.8.0 and huggingface-hub 1.22.0. Match serialized-artifact versions and test on this image.

## Resolve before spending substantial compute

- Confirm final case count or a safe upper bound: it determines generation budgets.
- Confirm active leaderboard baseline scores, submission quotas, ZIP size limits, eligibility details and exact cutoff time in Codabench. These were not verified here.
- Clarify effort values: website mentions high effort, while starter describes default/low/medium. Support metadata explicitly and confirm actual evaluation configurations.
- Ask where current labels for all five checkpoints are available. Public sample model names differ from current worker IDs; do not silently equate unrelated checkpoints.
- Confirm whether supplied public labels include sufficient independent problem families for meaningful evaluation.

No accounts were registered, messages sent, compute rented, or submissions uploaded.
