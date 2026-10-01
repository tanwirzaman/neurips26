# Competition readiness audit

Checked 2026-10-01 against the official website, starter instructions and remote `main`.

## Verdict

**Analysis upload: PASS. Competition readiness: NOT READY.**

All seven original tracked files were fetched from GitHub and their contents compared with local files: seven exact matches. The root `solution.py` is absent on GitHub. The uploaded analysis contains no implementation, trained artifacts, submission archive or evaluation results. GitHub publication does not enter the competition.

| Uploaded file | Audit result |
|---|---|
| README.md | Present; correct repository link; clearly labels the work as a plan |
| docs/requirements.md | Present; captures published requirements and unresolved questions |
| docs/baselines.md | Present; distinguishes development metrics from private scores |
| docs/plan.md | Present; proposed experiments, not claimed results |
| experiments/README.md | Present; recording template, no completed runs |
| sources.md | Present; limitations recorded |
| .gitignore | Present; excludes data, artifacts and archives from ordinary Git adds |

No analysis file was lost during upload. There is no basis to claim a competitive score.

## Required submission work

- [ ] Implement `are_robust(model_id, reasoning_effort, problems)`; return ordered Python booleans, one per input.
- [ ] Package a ZIP with root `solution.py`, imported helpers and every artifact actually required.
- [ ] Include root `small.txt` for Small only; omit it for Main.
- [ ] Exclude datasets, private labels, supplied model weights, environments, caches and credentials.
- [ ] Test offline using the official runtime: coverage 1.0, zero invalid predictions, deterministic repeated results.
- [ ] Measure the complete prediction run below 3,600 seconds on the single 96 GB GPU.
- [ ] Upload the correct ZIP to the chosen Codabench task and verify scoring completes.

Source: [official starter](https://github.com/aimo-interp/getting-started).

Artifacts are conditional: a constant smoke-test method needs none, but our proposed trained method does. A smoke-test entry would establish packaging only, not baseline improvement. A `requirements.txt` alone cannot fix unavailable evaluation dependencies.

## Participation and award work

- [ ] Verify the user's Codabench account and competition participation status.
- [ ] Submit by November 1, 2026; exact cutoff time remains unverified.
- [ ] For award consideration, submit the technical report and open implementation by November 15.
- [ ] Document training data, code, bundle and exact reproduction instructions.
- [ ] Be prepared to review up to three reports during November 15–30.
- [ ] Keep evaluation data isolated; additional labeled training data is permitted.

Source: [official rules](https://aimo-interp.github.io/).

The update form/Discord and compute-support proposal are useful options, not substitutes for a scored submission. Prior participation in the separate AIMO math-solving competition is unnecessary.

## Additional project work recommended

1. Reproduce a reference baseline and establish family-disjoint evaluation before claiming improvement.
2. Create a reproducible environment, training/feature scripts and package builder.
3. Pin source, dataset and checkpoint revisions; record licenses and data provenance.
4. Establish artifact delivery. Our `.gitignore` excludes `artifacts/`, `dist/`, `*.pkl` and `*.joblib`: normal Git adds will omit these. Publish necessary artifacts through a documented release/download mechanism and explicitly include them in the offline ZIP.
5. Select an appropriate code license before the open implementation release. A public repository by itself does not state reuse rights; the reviewed rules do not prescribe a specific license.
6. Reconcile local and remote Git histories before a future native push. The previous upload used the GitHub API, creating different commit history from the local repository. Do not force-push over remote work.

These are engineering recommendations, not additional organizer-mandated filenames. No LICENSE or placeholder solution was added merely to make the repository look complete.

## Remaining verification limits

Codabench's browser page loaded its title and navigation but not the competition details, and showed a Login link. Search access returned unrelated content. Therefore account participation, submission receipts, quotas, ZIP limits, phase status and exact deadline timezone could not be verified. No account action or competition submission was made.

There is also a documentation discrepancy: the website mentions high reasoning effort; the starter enumerates default/low/medium. Resolve actual evaluation configurations before release. The earlier requirements document already flags this.

Next milestone: implement and validate a minimal official-interface entry, then obtain a successful Codabench score. The present request audited readiness; it did not train or submit a method.
