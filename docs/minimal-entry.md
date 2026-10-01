# Option 1: zero-cost minimal entry

Implemented an always-True predictor using only built-in Python. It does not
load a checkpoint, train a model, call an API or use labels at prediction time.
This establishes the submission pipeline, not a competitive method.

## Bundles

- Small Models: `submissions/minimal-always-true-small.zip` (root `solution.py`, `small.txt`).
- Main: `submissions/minimal-always-true-main.zip` (root `solution.py`).

Select only the corresponding Codabench task when uploading. Start with Small.
These tiny ZIPs are intentionally tracked in Git; no datasets or upstream model
weights are included. Their hashes are recorded in the validation JSON.

## Reproduce

With Python 3.11+ and network access for preparation:

```sh
python scripts/validate_minimal.py --prepare
```

Subsequent runs need no network:

```sh
python scripts/validate_minimal.py
```

An optional `--work-dir PATH` selects a scratch directory. Preparation downloads
the official ingestion, scoring and importer by immutable Git blob SHA, verifies
each blob, and downloads the public sample without credentials. It checks the
dataset revision before and after the rows request. A changed revision or a
partial response stops preparation. If the network fails, retry preparation.

## Measured local result

On 2026-10-01, the official importer reduced 28 public source rows to 24 distinct
cases: 9 robust and 15 non-robust. Both ZIP variants scored **0.375 accuracy,
1.0 coverage, 0 invalid predictions** on this same sample. This does not estimate
separate private-track performance. Always-False would score 15/24 = 0.625 here;
the always-True entry was chosen in advance as an interface smoke test, not
selected to maximize this small sample's score.

Validation includes 60 model/effort/input combinations, empty input, duplicate
and long Unicode-containing test inputs, repeat predictions, deterministic ZIP
hashes, archive integrity, and rejection of each bundle on the wrong track.
The official harness runs in isolated Python subprocesses without third-party
packages. Docker and the organizer GPU image were **not** tested locally.

Full results: `experiments/minimal-validation.json`.

## Remote status

No Codabench score or submission ID has been obtained yet. Authentication and
the competition submission UI must be accessible to finish this step. Null
remote fields in the validation JSON deliberately distinguish local testing
from an accepted remote submission.

External compute spending: **$0**. No paid resources were provisioned.
