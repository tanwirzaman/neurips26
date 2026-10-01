#!/usr/bin/env python3
"""Run grouped, no-GPU controls on pinned public AIMO-Interp labels.

All text models are fitted within each fold using only the original problem.
This is a development control, not a private-track score.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import random
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import Any


DATASETS = [
    {
        "id": "aimo-interp/train-main-v2",
        "revision": "23791d9b120132087877b0459f953d5959fc8455",
        "split": "train",
        "family_key": "normalized exact problem text (provisional; 28 card-reported source problems map to 36 prompt strings)",
    },
    {
        "id": "aimo-interp/augmented-sample-math-agg-filtered",
        "revision": "0607cd06cbd789dd33655ff718bbfae0d3edba7c",
        "split": "validation",
        "family_key": "dataset_id + problem_id, cross-checked by exact normalized problem text",
    },
    {
        "id": "aimo-interp/augmented-sample-math-agg",
        "revision": "f972ced0705096f8d7ca7fac30825900b8b7fb6a",
        "split": "validation",
        "family_key": "dataset_id + problem_id, cross-checked by exact normalized problem text",
    },
]
N_FOLDS = 5
SEED = 20261001
BOOTSTRAP_REPLICATES = 2000


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "aimo-interp-public-controls/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def get_page(dataset: str, revision: str, split: str, offset: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "revision": revision,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": 100,
        }
    )
    return request_json(f"https://datasets-server.huggingface.co/rows?{query}")


def fetch_rows(dataset: str, revision: str, split: str) -> list[dict[str, Any]]:
    first = get_page(dataset, revision, split, 0)
    total = first["num_rows_total"]
    offsets = list(range(100, total, 100))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        pages = list(executor.map(lambda offset: get_page(dataset, revision, split, offset), offsets))
    rows = [entry["row"] for page in [first, *pages] for entry in page.get("rows", [])]
    if len(rows) != total:
        raise RuntimeError(f"{dataset}@{revision}: expected {total} rows, received {len(rows)}")
    return rows


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def problem_key(row: dict[str, Any], dataset: str) -> str:
    text = normalize_text(row.get("problem") or row.get("original_problem") or "")
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if row.get("problem_id"):
        return f"{dataset}:{row['problem_id']}"
    return f"text:{text_sha}"


def get_effort(row: dict[str, Any]) -> str:
    if row.get("reasoning_effort"):
        return row["reasoning_effort"]
    model = row.get("model_id", "")
    for separator in (":", "_"):
        if separator in model and model.rsplit(separator, 1)[1] in {"default", "low", "medium", "high"}:
            return model.rsplit(separator, 1)[1]
    return "unspecified"


def prepare_cases(spec: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = fetch_rows(spec["id"], spec["revision"], spec["split"])
    schema = set().union(*(row.keys() for row in rows))
    label_col = "is_robust" if "is_robust" in schema else "model_is_robust" if "model_is_robust" in schema else None
    if not label_col:
        raise RuntimeError(f"No robustness label column in {spec['id']}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    unlabeled = 0
    for row in rows:
        label = row.get(label_col)
        if not isinstance(label, bool):
            unlabeled += 1
            continue
        family = problem_key(row, spec["id"])
        config = f"{row.get('model_id', 'unknown')} | {get_effort(row)}"
        grouped[(family, config, label_col)].append(row)

    cases = []
    conflicts = 0
    for (family, config, _), members in grouped.items():
        labels = {row[label_col] for row in members}
        texts = {normalize_text(row.get("problem") or row.get("original_problem") or "") for row in members}
        if len(labels) != 1:
            conflicts += 1
            continue
        # Repeated measurement rows are one problem/config case. They must not
        # inflate the number of independent examples.
        if len(texts) != 1:
            raise RuntimeError(f"One family/config has differing prompt text: {family} / {config}")
        sample = members[0]
        cases.append(
            {
                "family": family,
                "config": config,
                "text": sample.get("problem") or sample.get("original_problem") or "",
                "label": next(iter(labels)),
            }
        )
    if conflicts:
        raise RuntimeError(f"Found {conflicts} problem/config groups with conflicting labels")
    cases.sort(key=lambda case: (case["family"], case["config"]))
    metadata = {
        "dataset": spec["id"],
        "revision": spec["revision"],
        "split": spec["split"],
        "source_rows": len(rows),
        "unlabelled_rows_excluded": unlabeled,
        "cases_after_grouping": len(cases),
        "problem_families": len({case["family"] for case in cases}),
        "model_effort_configurations": sorted({case["config"] for case in cases}),
        "family_key_rule": spec["family_key"],
        "family_group_leakage_across_folds": 0,
        "repeated_rows_collapsed": len(rows) - unlabeled - len(cases),
    }
    return cases, metadata


def assign_folds(cases: list[dict[str, Any]], n_folds: int = N_FOLDS) -> dict[str, int]:
    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for case in cases:
        by_family[case["family"]].append(case)
    if len(by_family) < n_folds:
        raise RuntimeError(f"Need at least {n_folds} families; found {len(by_family)}")

    totals: collections.Counter[tuple[str, bool]] = collections.Counter()
    vectors: dict[str, collections.Counter[tuple[str, bool]]] = {}
    for family, members in by_family.items():
        vector = collections.Counter((case["config"], case["label"]) for case in members)
        vectors[family] = vector
        totals.update(vector)

    def priority(family: str) -> tuple[float, int, str]:
        vector = vectors[family]
        rarity = sum(count / totals[key] for key, count in vector.items())
        return (-rarity, -sum(vector.values()), hashlib.sha256(f"{SEED}:{family}".encode()).hexdigest())

    fold_vectors = [collections.Counter() for _ in range(n_folds)]
    fold_loads = [0] * n_folds
    assignment: dict[str, int] = {}
    ordered_families = sorted(by_family, key=priority)

    # Seed every fold so a local greedy choice cannot leave test folds empty.
    for fold, family in enumerate(ordered_families[:n_folds]):
        assignment[family] = fold
        fold_vectors[fold].update(vectors[family])
        fold_loads[fold] += len(by_family[family])

    for family in ordered_families[n_folds:]:
        vector = vectors[family]
        options = []
        for fold in range(n_folds):
            delta = 0.0
            for stratum, total in totals.items():
                target = total / n_folds
                before = fold_vectors[fold][stratum] - target
                after = fold_vectors[fold][stratum] + vector[stratum]
                after -= target
                delta += (after * after - before * before) / (target + 1)
            load_target = len(cases) / n_folds
            before_load = fold_loads[fold] - load_target
            after_load = fold_loads[fold] + len(by_family[family]) - load_target
            delta += (after_load * after_load - before_load * before_load) / (load_target + 1)
            tie = hashlib.sha256(f"{SEED}:{family}:{fold}".encode()).hexdigest()
            options.append((delta, fold_loads[fold], tie, fold))
        chosen = min(options)[-1]
        assignment[family] = chosen
        fold_vectors[chosen].update(vector)
        fold_loads[chosen] += len(by_family[family])
    return assignment


def tokens(text: str, bigrams: bool = False) -> collections.Counter[str]:
    words = re.findall(r"[^\W_]+", normalize_text(text), flags=re.UNICODE)
    features = [f"u:{word}" for word in words]
    if bigrams:
        features.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    return collections.Counter(features)


class MultinomialNB:
    def __init__(self, bigrams: bool = False, alpha: float = 1.0) -> None:
        self.bigrams = bigrams
        self.alpha = alpha

    def fit(self, train: list[dict[str, Any]]) -> None:
        self.class_docs = collections.Counter(case["label"] for case in train)
        self.word_counts = {False: collections.Counter(), True: collections.Counter()}
        self.total_tokens = {False: 0, True: 0}
        self.vocabulary: set[str] = set()
        for case in train:
            counts = tokens(case["text"], self.bigrams)
            self.word_counts[case["label"]].update(counts)
            self.total_tokens[case["label"]] += sum(counts.values())
            self.vocabulary.update(counts)
        if not self.class_docs[False] or not self.class_docs[True]:
            # The classifier falls back to the training-fold majority if a fold
            # lacks a class; this guard makes that behavior explicit.
            self.available = False
        else:
            self.available = True

    def predict(self, case: dict[str, Any], fallback: bool) -> bool:
        if not self.available:
            return fallback
        counts = tokens(case["text"], self.bigrams)
        vocabulary_size = max(len(self.vocabulary), 1)
        scores = {}
        for label in (False, True):
            scores[label] = math.log(self.class_docs[label] / sum(self.class_docs.values()))
            denominator = self.total_tokens[label] + self.alpha * vocabulary_size
            for word, count in counts.items():
                if word in self.vocabulary:
                    numerator = self.word_counts[label][word] + self.alpha
                    scores[label] += count * math.log(numerator / denominator)
        return scores[True] > scores[False] if scores[True] != scores[False] else fallback


def confusion(labels: list[bool], predictions: list[bool]) -> dict[str, int]:
    tp = sum(y and p for y, p in zip(labels, predictions))
    tn = sum((not y) and (not p) for y, p in zip(labels, predictions))
    fp = sum((not y) and p for y, p in zip(labels, predictions))
    fn = sum(y and (not p) for y, p in zip(labels, predictions))
    return {"tp_robust": tp, "fn_robust": fn, "tn_non_robust": tn, "fp_non_robust": fp}


def metrics(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    cm = confusion(labels, predictions)
    accuracy = (cm["tp_robust"] + cm["tn_non_robust"]) / max(len(labels), 1)
    tpr = cm["tp_robust"] / max(cm["tp_robust"] + cm["fn_robust"], 1)
    tnr = cm["tn_non_robust"] / max(cm["tn_non_robust"] + cm["fp_non_robust"], 1)
    return {
        "n_cases": len(labels),
        "accuracy": accuracy,
        "balanced_accuracy": (tpr + tnr) / 2,
        "confusion": cm,
        "coverage": 1.0,
        "invalid_predictions": 0,
    }


def bootstrap_ci(cases: list[dict[str, Any]], predictions: list[bool]) -> dict[str, list[float]]:
    by_family: dict[str, list[int]] = collections.defaultdict(list)
    for index, case in enumerate(cases):
        by_family[case["family"]].append(index)
    families = sorted(by_family)
    rng = random.Random(SEED)
    accuracy_values = []
    balanced_values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        chosen = [rng.choice(families) for _ in families]
        indices = [index for family in chosen for index in by_family[family]]
        labels = [cases[index]["label"] for index in indices]
        preds = [predictions[index] for index in indices]
        cm = confusion(labels, preds)
        accuracy_values.append((cm["tp_robust"] + cm["tn_non_robust"]) / len(indices))
        if cm["tp_robust"] + cm["fn_robust"] and cm["tn_non_robust"] + cm["fp_non_robust"]:
            tpr = cm["tp_robust"] / (cm["tp_robust"] + cm["fn_robust"])
            tnr = cm["tn_non_robust"] / (cm["tn_non_robust"] + cm["fp_non_robust"])
            balanced_values.append((tpr + tnr) / 2)

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {"accuracy_95pct_family_bootstrap_ci": interval(accuracy_values), "balanced_accuracy_95pct_family_bootstrap_ci": interval(balanced_values)}


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    family_to_fold = assign_folds(cases)
    predictions = {name: [False] * len(cases) for name in ("always_false", "always_true", "fold_majority", "configuration_majority", "text_unigram_nb", "text_unigram_bigram_nb")}
    fold_summary = []

    for fold in range(N_FOLDS):
        test_indices = [i for i, case in enumerate(cases) if family_to_fold[case["family"]] == fold]
        train_indices = [i for i, case in enumerate(cases) if family_to_fold[case["family"]] != fold]
        train = [cases[i] for i in train_indices]
        test = [cases[i] for i in test_indices]
        overall = collections.Counter(case["label"] for case in train)
        global_majority = overall[True] > overall[False]
        by_config: dict[str, collections.Counter[bool]] = collections.defaultdict(collections.Counter)
        for case in train:
            by_config[case["config"]][case["label"]] += 1

        unigram = MultinomialNB(False)
        unigram.fit(train)
        uni_bigram = MultinomialNB(True)
        uni_bigram.fit(train)
        for index in test_indices:
            case = cases[index]
            predictions["always_false"][index] = False
            predictions["always_true"][index] = True
            predictions["fold_majority"][index] = global_majority
            counts = by_config[case["config"]]
            predictions["configuration_majority"][index] = counts[True] > counts[False] if counts[True] != counts[False] else global_majority
            predictions["text_unigram_nb"][index] = unigram.predict(case, global_majority)
            predictions["text_unigram_bigram_nb"][index] = uni_bigram.predict(case, global_majority)
        fold_summary.append(
            {
                "fold": fold,
                "train_cases": len(train),
                "test_cases": len(test),
                "train_families": len({case["family"] for case in train}),
                "test_families": len({case["family"] for case in test}),
                "train_labels": {str(key): value for key, value in overall.items()},
                "test_labels": {str(key): sum(case["label"] == key for case in test) for key in (False, True)},
            }
        )

    result = {"folds": fold_summary, "controls": {}}
    for name, preds in predictions.items():
        result["controls"][name] = {**metrics([case["label"] for case in cases], preds), **bootstrap_ci(cases, preds)}

    result["per_configuration"] = {}
    configs = sorted({case["config"] for case in cases})
    for config in configs:
        indices = [i for i, case in enumerate(cases) if case["config"] == config]
        result["per_configuration"][config] = {
            "n_families": len({cases[i]["family"] for i in indices}),
            "label_counts": {str(label): sum(cases[i]["label"] == label for i in indices) for label in (False, True)},
            "controls": {
                name: metrics([cases[i]["label"] for i in indices], [preds[i] for i in indices])
                for name, preds in predictions.items()
            },
        }
    result["fold_group_leakage"] = 0
    result["n_problem_families"] = len({case["family"] for case in cases})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional path for the JSON experiment result")
    args = parser.parse_args()
    result = {
        "run_id": "E0-controls-2026-10-01",
        "date": "2026-10-01",
        "seed": SEED,
        "folds": N_FOLDS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "method": "Five-fold deterministic greedy stratified group CV; majority controls and stdlib multinomial Naive Bayes unigram / unigram+bigram controls. Vocabulary and token counts fit on each training fold only.",
        "text_model_features": "Original problem text only. No model outputs, labels, max_drop, accuracy/decay fields, permutation metadata, model ID, or effort are classifier features.",
        "limitations": [
            "Development cross-validation on public labeled datasets is not a private-track score.",
            "train-main-v2 has no family IDs. Exact normalized prompt hashes are a provisional grouping and may split related variants; its results are exploratory.",
            "For legacy aggregate datasets, dataset_id + problem_id is the group key; family-level bootstrap intervals assume these IDs represent independent base problems.",
            "No model weights, GPU, external ML packages, or paid compute were used.",
        ],
        "datasets": [],
    }
    for spec in DATASETS:
        cases, metadata = prepare_cases(spec)
        entry = {**metadata, **evaluate(cases)}
        result["datasets"].append(entry)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

