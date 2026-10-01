# Public data audit

Audit date: 2026-10-01. This is a metadata and label audit only; no model was
trained, no GPU or paid compute was used, and no data was added to a submission.
The full machine-readable inventory and repeatable stdlib-only fetcher are
[`experiments/data-audit-2026-10-01.json`](../experiments/data-audit-2026-10-01.json)
and [`scripts/audit_public_data.py`](../scripts/audit_public_data.py).

## Findings

The organizer's Hugging Face account currently exposes seven public, ungated
datasets. The main labeled training resource is `aimo-interp/train-main-v2` at
revision `23791d9b120132087877b0459f953d5959fc8455`. Its 82 rows cover all five
current competition checkpoint IDs and four effort values. The data card says
the rows cover 28 underlying problems: 52 robust, 19 non-robust, and 11
unlabelled cases. The unlabelled cases have `max_drop` between 0.11 and 0.21;
the card assigns labels only for `max_drop <= 0.10` (robust) and
`max_drop >= 0.25` (non-robust).

| Configuration | Robust | Non-robust | Unlabelled |
|---|---:|---:|---:|
| Qwen/Qwen3.5-4B · default | 8 | 4 | 2 |
| Skywork/Skywork-OR1-Math-7B · default | 2 | 1 | 0 |
| allenai/Olmo-3-7B-Think · default | 7 | 2 | 1 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B · default | 7 | 3 | 0 |
| openai/gpt-oss-120b · high | 13 | 2 | 3 |
| openai/gpt-oss-120b · low | 5 | 2 | 1 |
| openai/gpt-oss-120b · medium | 10 | 5 | 4 |

There is no problem or family ID in this training schema. Exact Unicode-NFKC,
case-folded, whitespace-normalized text hashing produces 36 distinct prompt
strings, while the data card reports 28 underlying problems. Those counts are
not interchangeable: model/configuration prompts may be variants of the same
source problem. The public data does not provide the mapping needed to build a
defensible family-disjoint split. Do not split rows randomly or claim the 36
strings are independent families. Request or construct a reviewed mapping
before using cross-validation or a final holdout.

The machine audit checked exact normalized-text overlap with `train-main-v2`.
Five of the eight problem texts in `val-sample` also occur in `train-main-v2`;
the 558-row `aimo-interp-challenge-sample-full` has five exact overlaps among
its ten distinct problem texts. Near-duplicate, paraphrase, and shared-source
problem overlap was not tested, so zero exact overlap is not proof of an
independent holdout.

| Dataset and pinned revision | Split / rows | Problem and label summary | Model coverage |
|---|---:|---|---|
| [`train-main-v2`](https://huggingface.co/datasets/aimo-interp/train-main-v2/tree/23791d9b120132087877b0459f953d5959fc8455) `23791d9b120132087877b0459f953d5959fc8455` | train / 82 | 28 underlying problems per card; 52 true, 19 false, 11 null | All five current checkpoint IDs; default, plus low/medium/high for gpt-oss |
| [`val-sample`](https://huggingface.co/datasets/aimo-interp/val-sample/tree/1ae454ec1fad9727084eda8f9f3c9ae2239b21de) `1ae454ec1fad9727084eda8f9f3c9ae2239b21de` | validation / 28 | 8 problem IDs; 24 unique problem/model cases after repeated rows; 9 true, 15 false | 7 older or alias IDs; none exactly matches a current evaluation ID |
| [`augmented-sample-math-agg-filtered`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg-filtered/tree/0607cd06cbd789dd33655ff718bbfae0d3edba7c) `0607cd06cbd789dd33655ff718bbfae0d3edba7c` | validation / 54 | 52 unique problems; 11 true, 41 false after repeated rows are grouped | Exact current DeepSeek checkpoint only |
| [`augmented-sample-math-agg`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg/tree/f972ced0705096f8d7ca7fac30825900b8b7fb6a) `f972ced0705096f8d7ca7fac30825900b8b7fb6a` | validation / 141 | 137 unique problem IDs; 36 true, 101 false after repeated rows are grouped | One legacy `qwen3-8b:low` ID |
| [`augmented-sample-math`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math/tree/e1aaca9895b31fc9ce3d8d0d4e2f3009f6ca8090) `e1aaca9895b31fc9ce3d8d0d4e2f3009f6ca8090` | validation / 156 | 49 unique problems; all 49 grouped labels are false | One legacy Llama 3.1 ID |
| [`augmented-sample-math-full-filtered`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-full-filtered/tree/f09e41ed0648b0ff7cac220c6ce9150a36e2e8b0) `f09e41ed0648b0ff7cac220c6ce9150a36e2e8b0` | validation / 259 | 69 unique problems; no `model_is_robust` label column | One legacy `qwen3-8b:low` ID |
| [`aimo-interp-challenge-sample-full`](https://huggingface.co/datasets/aimo-interp/aimo-interp-challenge-sample-full/tree/c0ffb7e678294cc8819d50d979ba28ba49bd02d0) `c0ffb7e678294cc8819d50d979ba28ba49bd02d0` | validation / 558 | 10 unique problems; no robustness label column | 8 older or alias IDs |

Repeated rows in the augmented/full datasets often represent perturbation-level
measurements for the same problem/model pair, not independent examples. Their
row counts must not be used as sample sizes without grouping. The
`augmented-sample-math-agg-filtered` data is the only listed auxiliary labeled
set with an exact current model ID; its terms and intended relationship to the
current train set still need confirmation before relying on it.

## Leakage and data-use rules

- Group every model/effort row and every known variant of one underlying
  problem into the same validation fold. Use the supplied `problem_id` where
  available. `train-main-v2` lacks it, so its 28-family mapping remains open.
- Exact-text hashes in the JSON are overlap checks, not semantic-family labels.
  They normalize Unicode, case and whitespace only; no fuzzy matching was run.
- `max_drop`, base/permuted accuracies, decay fields, perturbation lists and
  robustness labels are audit/target information. Do not use them as
  prediction-time features; the evaluator supplies only model ID, effort and
  original problem text.
- The current competition rules allow additional labeled training data, but
  this does not settle dataset licensing. Six repositories expose no license
  field; `train-main-v2` says `other` but provides no plain-language license
  terms in its card. Confirm rights before redistributing or relying on data
  outside the competition.
- The public `val-sample` is small and overlaps the training problem set; it is
  an interface/development sample, not an untouched holdout.

The competition website describes the available signals and the evaluation
interface; the starter repository documents the public validation sample and
the current checkpoint list. See [competition rules](https://aimo-interp.github.io/),
[official starter](https://github.com/aimo-interp/getting-started), and the
[organizer's Hugging Face datasets](https://huggingface.co/aimo-interp).

## Reproduce the audit

With Python 3.11+ and network access, run:

```sh
python scripts/audit_public_data.py --output experiments/data-audit-2026-10-01.json
```

The script uses only Python's standard library, fetches rows at the revisions
listed above, checks completeness, and hashes a projection of IDs, labels,
measurement columns and normalized prompt hashes. The reported hash is not a
Parquet file checksum. The JSON records the projection hash and source revision
for each dataset. No source dataset rows are saved into the repository.

