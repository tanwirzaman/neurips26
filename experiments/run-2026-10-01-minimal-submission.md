# Run log: minimal submission and validation

**Run ID:** E0-smoke  
**Date:** 2026-10-01  
**Purpose:** Build and validate a zero-cost submission package, confirm Codabench accepts it, and establish a measured starting point before attempting baseline improvements.  
**Outcome:** Submission pipeline succeeded. No trained baseline was reproduced or beaten.

## What was run

1. Implemented an always-True predictor in `solution.py`, using only Python's standard library. This was selected as an interface smoke test before looking at the small public sample's labels; it is not a candidate classifier.
2. Built Small Models and Main Track ZIPs. Both bundles were checked for archive integrity, stable SHA-256 hashes, correct root files and rejection on the wrong track.
3. Prepared the official ingestion, scoring and dataset-import scripts from pinned Git blob SHAs. Downloaded the public sample at dataset revision `1ae454ec1fad9727084eda8f9f3c9ae2239b21de` and recorded the response hash.
4. Ran local public-sample scoring and 60 submission-contract permutations across model, effort and input combinations. Also checked empty, repeated, long Unicode-containing inputs and deterministic predictions.
5. Uploaded the Small Models ZIP to Codabench competition 16180, task 36241. Codabench finished submission 955339 and displayed accuracy 0.50. One of ten daily submissions was used. The Codabench result did not provide evaluation logs.
6. No GPU training, paid compute, external API calls, Main Track Codabench submission or Docker/organizer-image test was performed.

## Results

The public source contained 28 rows; the official importer reduced these to 24 distinct cases (9 robust, 15 non-robust). Both ZIPs scored 0.375 accuracy, 1.0 coverage and 0 invalid predictions on that same local sample. The always-True method was not selected using those labels. Always-False would score 0.625 on this tiny sample, but was not submitted because this run's purpose was pipeline validation.

| Track bundle | SHA-256 | Local accuracy | Coverage | Invalid |
|---|---|---:|---:|---:|
| Small Models | `8190cb7b5f565fbf742dfe18e91580971885fbb6661b84367071654daf488c0a` | 0.375 | 1.0 | 0 |
| Main | `664f9cd23e93c8424530898f3423153c25ff9c4593a98d0ab1152e3efac93c6e` | 0.375 | 1.0 | 0 |

Codabench result: Small Models, submission 955339, status Finished, displayed accuracy 0.50. This score is recorded separately from the local public-sample result in `codabench-955339.json`.

## Interpretation and next work

This run verifies packaging, interface behavior and one end-to-end Codabench evaluation. It does not test a learned predictor and does not establish improvement over the published trained baseline. There is no model-training experiment to report yet.

The planned next experiments are in `../docs/plan.md`: establish majority and shortcut controls, reproduce the published baselines under fixed family splits, then test representation probes and only retain changes that improve a family-disjoint holdout with complete runtime and coverage. Each future run should get its own record rather than overwriting this one.

## Reproduction and source records

- Validator: `../scripts/validate_minimal.py`
- Full machine-readable measurements and pinned harness hashes: `minimal-validation.json`
- Codabench receipt: `codabench-955339.json`
- Submission description and reproduction steps: `../docs/minimal-entry.md`
- Public dataset revision and source details: see the validation JSON and `../sources.md`
- Paid compute: $0
- Docker / organizer GPU image: not tested
