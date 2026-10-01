# Experiment record template

Create one Markdown record per run with:

- Run ID, hypothesis, UTC timestamp and Git commit.
- Dataset revisions, licenses, family split hash and label definition.
- Checkpoint revisions, effort, chat template, feature configuration and random seeds.
- Training/runtime package versions and hardware.
- Exact reproduction commands and artifact hashes.
- Out-of-fold and untouched-holdout accuracy, balanced accuracy, confusion matrices, per-model results and paired family-bootstrap interval against E1.
- Total loading/inference time, peak VRAM, supported model count, coverage and invalid predictions.
- Conclusion: retain, reject or inconclusive; next experiment justified by evidence.

Keep large artifacts and private/local data outside Git. Record their paths and hashes, never credentials. Do not overwrite previous results when retuning.
