# Sources and verification scope

Accessed 2026-10-01:

- https://aimo-interp.github.io/
- https://github.com/aimo-interp/getting-started
- https://github.com/aimo-interp/baselines
- https://github.com/aimo-interp/getting-started/blob/main/solutions/trained-probe/probe_inference.py
- https://github.com/aimo-interp/getting-started/blob/main/solutions/uncertainty-profiling/README.md
- https://github.com/aimo-interp/getting-started/blob/main/Dockerfile.competition
- https://huggingface.co/datasets/aimo-interp/val-sample
- https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg
- https://www.codabench.org/competitions/16180/ (unusable response; standings not verified)

The uncertainty README pins its training data to `f972ced0705096f8d7ca7fac30825900b8b7fb6a`. Current dataset viewer counts need not equal counts at that historical revision. GitHub commit SHAs could not be verified, so source URLs are mutable and must be pinned before implementation.

Local Git 2.53.0.windows.3 is available but HTTPS cloning failed because its remote-https helper was unavailable. A shell HTTP request was blocked by network permissions. Browser-accessible official source files were inspected instead. Python, uv, Docker and nvidia-smi were not found on this shell's PATH; that does not establish they are absent elsewhere on the machine.

This deliverable is an analysis repository. GPU execution, data download, baseline reproduction, live standings, container tests and competition registration remain unperformed. No upstream implementation was vendored.
