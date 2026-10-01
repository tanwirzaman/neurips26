#!/usr/bin/env python3
"""Audit the organizer's public robustness datasets using only the stdlib.

Fetches pinned Hugging Face dataset-server rows and emits descriptive metadata.
It does not train a model or save source rows. Use --output to save the report.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import unicodedata
import urllib.parse
import urllib.request
from typing import Any


DATASETS = [
    ("aimo-interp/val-sample", "1ae454ec1fad9727084eda8f9f3c9ae2239b21de", "validation"),
    ("aimo-interp/augmented-sample-math", "e1aaca9895b31fc9ce3d8d0d4e2f3009f6ca8090", "validation"),
    ("aimo-interp/augmented-sample-math-full-filtered", "f09e41ed0648b0ff7cac220c6ce9150a36e2e8b0", "validation"),
    ("aimo-interp/augmented-sample-math-agg-filtered", "0607cd06cbd789dd33655ff718bbfae0d3edba7c", "validation"),
    ("aimo-interp/aimo-interp-challenge-sample-full", "c0ffb7e678294cc8819d50d979ba28ba49bd02d0", "validation"),
    ("aimo-interp/augmented-sample-math-agg", "f972ced0705096f8d7ca7fac30825900b8b7fb6a", "validation"),
    ("aimo-interp/train-main-v2", "23791d9b120132087877b0459f953d5959fc8455", "train"),
]

EVALUATION_MODELS = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b",
}

DATASET_CARD_PROBLEM_COUNTS = {"aimo-interp/train-main-v2": 28}


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "aimo-interp-data-audit/1.0"})
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


def fetch_rows(dataset: str, revision: str, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_page = get_page(dataset, revision, split, 0)
    total = first_page["num_rows_total"]
    page_offsets = list(range(100, total, 100))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        pages = list(executor.map(lambda offset: get_page(dataset, revision, split, offset), page_offsets))
    all_pages = [first_page, *pages]
    rows = [item["row"] for page in all_pages for item in page.get("rows", [])]
    truncations = [cell for page in all_pages for cell in page.get("truncated_cells", [])]
    if len(rows) != total:
        raise RuntimeError(f"{dataset}@{revision}: expected {total} rows, received {len(rows)}")
    return rows, truncations


def normalized_text(row: dict[str, Any]) -> str:
    value = row.get("problem") or row.get("original_problem") or ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def text_hash(row: dict[str, Any]) -> str | None:
    text = normalized_text(row)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def summarize(dataset: str, revision: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, truncations = fetch_rows(dataset, revision, split)
    schema = sorted({key for row in rows for key in row})
    missing = {
        key: sum(key not in row or row[key] is None or row[key] == "" for row in rows)
        for key in schema
    }
    label_field = "is_robust" if "is_robust" in schema else "model_is_robust" if "model_is_robust" in schema else None
    label_counts: collections.Counter[str] = collections.Counter()
    if label_field:
        label_counts.update(str(row.get(label_field)) for row in rows)

    models = sorted({row["model_id"] for row in rows if row.get("model_id")})
    efforts = sorted({row["reasoning_effort"] for row in rows if row.get("reasoning_effort")})
    if not efforts:
        parsed_efforts = set()
        for model in models:
            for separator in (":", "_"):
                if separator in model and model.rsplit(separator, 1)[1] in {"default", "low", "medium", "high"}:
                    parsed_efforts.add(model.rsplit(separator, 1)[1])
        efforts = sorted(parsed_efforts)

    text_fingerprints = [text_hash(row) for row in rows]
    problem_ids = {row.get("problem_id") for row in rows if row.get("problem_id")}
    data_ids = sorted({row.get("dataset_id") for row in rows if row.get("dataset_id")})
    groups: collections.Counter[tuple[str, str, str]] = collections.Counter()
    labels_by_group: dict[tuple[str, str, str], set[Any]] = collections.defaultdict(set)
    labels_by_configuration: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    exact_records: collections.Counter[str] = collections.Counter()
    projection_rows = []

    for row, fingerprint in zip(rows, text_fingerprints):
        problem_key = row.get("problem_id") or fingerprint or "<missing-problem>"
        dataset_key = row.get("dataset_id") or "<no-dataset-id>"
        effort = row.get("reasoning_effort")
        if effort is None:
            model = row.get("model_id", "")
            for separator in (":", "_"):
                if separator in model and model.rsplit(separator, 1)[1] in {"default", "low", "medium", "high"}:
                    effort = model.rsplit(separator, 1)[1]
                    break
        group_key = (dataset_key, str(problem_key), str(row.get("model_id", "<missing-model>")) + "/" + str(effort or "default"))
        groups[group_key] += 1
        if label_field:
            labels_by_group[group_key].add(row.get(label_field))
            labels_by_configuration[(str(row.get("model_id", "<missing-model>")), str(effort or "default"))][str(row.get(label_field))] += 1

        # Hash an auditable projection rather than large nested perturbation text.
        projection = {
            key: row.get(key)
            for key in (
                "dataset_id", "problem_id", "model_id", "reasoning_effort", "is_robust",
                "model_is_robust", "max_drop", "base_accuracy", "permuted_accuracy",
                "absolute_accuracy_decay", "relative_accuracy_decay", "n_base_predictions",
                "n_permuted_predictions", "n_detrimental_permutations", "permutation_type",
                "permutation_source",
            )
            if key in row
        }
        projection["normalized_problem_sha256"] = fingerprint
        projection_rows.append(projection)
        exact_records[json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))] += 1

    duplicate_case_rows = sum(count - 1 for count in groups.values() if count > 1)
    contradictory_group_count = sum(len(labels) > 1 for labels in labels_by_group.values())
    grouped_label_counts: collections.Counter[str] = collections.Counter()
    for labels in labels_by_group.values():
        if len(labels) == 1:
            grouped_label_counts[str(next(iter(labels)))] += 1
    truncation_counts = collections.Counter(cell.get("column_name", "unknown") for cell in truncations)
    report = {
        "dataset": dataset,
        "revision": revision,
        "split": split,
        "rows": len(rows),
        "schema": schema,
        "missing_or_empty_by_column": missing,
        "label_field": label_field,
        "label_counts": dict(label_counts),
        "unique_problem_model_effort_label_counts": dict(grouped_label_counts),
        "row_label_counts_by_model_effort": {
            f"{model} | {effort}": dict(counts)
            for (model, effort), counts in sorted(labels_by_configuration.items())
        },
        "model_ids": models,
        "reasoning_efforts": efforts,
        "dataset_ids": data_ids,
        "distinct_problem_ids": len(problem_ids),
        "distinct_normalized_problem_texts": len({item for item in text_fingerprints if item}),
        "problem_model_effort_groups": len(groups),
        "repeated_rows_for_same_problem_model_effort": duplicate_case_rows,
        "exact_duplicate_audit_projection_rows": sum(count - 1 for count in exact_records.values() if count > 1),
        "groups_with_conflicting_labels": contradictory_group_count,
        "dataset_server_truncated_cells_by_column": dict(truncation_counts),
        "audit_projection_sha256": hashlib.sha256(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in projection_rows).encode("utf-8")
        ).hexdigest(),
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional path to write the JSON audit report")
    args = parser.parse_args()

    reports = []
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for dataset, revision, split in DATASETS:
        report, rows = summarize(dataset, revision, split)
        metadata = request_json(f"https://huggingface.co/api/datasets/{dataset}/revision/{revision}")
        report["public"] = not metadata.get("private", True)
        report["gated"] = metadata.get("gated", False)
        report["license_metadata"] = metadata.get("cardData", {}).get("license")
        report["metadata_tags"] = metadata.get("tags", [])
        if dataset in DATASET_CARD_PROBLEM_COUNTS:
            report["dataset_card_reported_problem_count"] = DATASET_CARD_PROBLEM_COUNTS[dataset]
        reports.append(report)
        row_sets[dataset] = rows

    text_hashes = {
        dataset: {text_hash(row) for row in rows if text_hash(row)}
        for dataset, rows in row_sets.items()
    }
    train_hashes = text_hashes["aimo-interp/train-main-v2"]
    for report in reports:
        report["exact_normalized_problem_text_overlap_with_train_main_v2"] = len(
            text_hashes[report["dataset"]] & train_hashes
        )
        report["exact_evaluation_model_id_matches"] = sorted(
            set(report["model_ids"]) & EVALUATION_MODELS
        )

    result = {
        "audit_date": "2026-10-01",
        "source": "Hugging Face Hub metadata API and datasets-server rows API",
        "family_proxy": "dataset_id + problem_id where available; normalized exact problem text hash for cross-dataset overlap and rows without problem_id. This is not semantic-family annotation.",
        "text_normalization": "Unicode NFKC, casefold, collapse whitespace; no fuzzy or mathematical-equivalence deduplication.",
        "prediction_feature_policy": "Labels and perturbation outcomes are inspected only for this data audit; do not pass them to prediction-time code.",
        "evaluation_model_ids": sorted(EVALUATION_MODELS),
        "datasets": reports,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

