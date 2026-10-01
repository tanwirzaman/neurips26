# Public grouped controls

Run `E0-controls-2026-10-01` evaluates majority and text-only controls on three pinned public AIMO-Interp datasets. It is a local CPU experiment, not a Codabench or private-track score. The run is reproducible with:

```powershell
python scripts/evaluate_public_controls.py --output experiments/public-controls-2026-10-01.json
```

The full run record contains source revisions, fold counts, confusion matrices, per-model/effort results, coverage, invalid-prediction counts, and family-bootstrap intervals. The classifier sees only the original problem text. It does not see labels, model outputs, robustness drops, model identity, effort, or perturbation metadata. Vocabulary and token counts are learned separately inside each training fold. The text controls are multinomial Naive Bayes with unigram and unigram-plus-bigram features; no external ML package is required.

| Dataset | Labeled cases | Groups | Majority accuracy | Unigram NB accuracy / balanced accuracy | Unigram + bigram NB accuracy / balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train-main-v2` | 71 | 33 provisional prompt groups | 73.2% | 45.1% / 44.1% | 47.9% / 42.7% |
| `augmented-sample-math-agg-filtered` | 52 | 52 problem IDs | 78.8% | 65.4% / 44.8% | 69.2% / 43.9% |
| `augmented-sample-math-agg` | 137 | 137 problem IDs | 73.7% | 69.3% / 53.3% | 67.9% / 50.5% |

The majority control predicts the most common class from the training fold; ties fall back to the overall training-fold majority. On all three datasets, this equals the always-majority class for aggregate accuracy. On the 137-case aggregate dataset, unigram NB has the strongest balanced accuracy (53.3%), but lower accuracy than majority (69.3% vs. 73.7%). That small balanced-accuracy lift is exploratory; it does not establish a reliable gain. The text controls do not beat majority accuracy on any dataset.

The folds are deterministic, grouped, and nonempty. Problem IDs define groups for the two aggregate datasets. `train-main-v2` has no family identifiers, so its folds group exact normalized prompt text; related variants can still land in separate folds. Treat those results as provisional. The labeled train-main subset has 33 exact-text groups, while the dataset card describes 28 source problems and the broader row audit found 36 prompt strings; there is not enough public metadata to reconcile these counts or build a verified family mapping. The aggregate datasets also have class imbalance, so the 0.5 balanced accuracy of their constant predictors is more informative than majority accuracy alone.

All predictions were valid (coverage 100%, invalid predictions 0); the evaluation targets are already binary labels, so this does not test a participant submission parser. Family-bootstrap 95% intervals and full per-configuration metrics are in [`experiments/public-controls-2026-10-01.json`](../experiments/public-controls-2026-10-01.json). These public controls are a low-cost reference for later modeling, not evidence that the private leaderboard baseline has been beaten. No GPU, paid service, hosted inference, or package installation was used.

