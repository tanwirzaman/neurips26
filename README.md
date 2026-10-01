# AIMO Interpretability entry

**Current implementation:** a zero-compute minimal entry is built and locally
validated. See [minimal entry](docs/minimal-entry.md) for ZIPs, reproduction and
results. No Codabench score has been obtained yet. Earlier analysis and the
readiness audit describe the state before this implementation.

Research and participation plan, checked 2026-10-01. Project repository: [tanwirzaman/neurips26](https://github.com/tanwirzaman/neurips26). No model has been trained or baseline beaten.

Start with [requirements](docs/requirements.md), [baseline audit](docs/baselines.md), and [experiment plan](docs/plan.md).

Recommended direction: Small Models Track first; train inexpensive, model-specific representation probes, validate on held-out problem families, then test a calibrated ensemble with short reasoning-trace features. Add the Main Track only after its additional checkpoint passes memory and whole-run timing checks. This is a proposed strategy, not a measured result.

## Repository contents

- `docs/requirements.md`: entry checklist and unresolved organizer questions.
- `docs/baselines.md`: inspected baselines, reported scores, and concrete weaknesses.
- `docs/plan.md`: experiments, validation design, compute budget, and October schedule.
- `experiments/README.md`: required experiment records.
- `sources.md`: official sources and environment limitations.

## First execution session

1. Obtain a working Git HTTPS installation, Python 3.11+, uv, and a Linux CUDA training environment.
2. Clone the official starter and baselines into ignored `external/` directories. Record actual commit SHAs before experiments; do not assume mutable `main` is reproducible.
3. In the starter, run `uv sync`, then `uv run scripts/import_hf_dataset.py`.
4. Run the starter's local evaluation on its trivial example to verify the contract. Reproduce trained baselines separately; missing-artifact fallback is not a trained baseline.
5. Follow the experiment plan before training or selecting a final submission.

Official entry points: [starter](https://github.com/aimo-interp/getting-started), [baselines](https://github.com/aimo-interp/baselines), [competition](https://aimo-interp.github.io/).
