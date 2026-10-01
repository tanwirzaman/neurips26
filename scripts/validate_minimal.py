"""Build deterministic ZIPs and validate with the official CPU-only harness.

Python 3.11+ standard library only. --prepare downloads public validation data
and hash-pinned official scripts; subsequent runs are fully local.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
DATA_REVISION = "1ae454ec1fad9727084eda8f9f3c9ae2239b21de"
BLOBS = {
    "ingestion.py": "aba27e901b4595404c53c2763c6d21be2ae8d772",
    "scoring.py": "0070d8fffa4351b7b12fb85648af0b0b88c04540",
    "import_hf_dataset.py": "90992e4975f2d7a06625400a602db68f1e484ea6",
}
MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B",
          "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
          "openai/gpt-oss-120b"]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def fetch(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


def blob_sha(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    for filename, sha in BLOBS.items():
        reply = json.loads(fetch(f"https://api.github.com/repos/aimo-interp/getting-started/git/blobs/{sha}"))
        payload = base64.b64decode(reply["content"])
        check(blob_sha(payload) == sha, f"Upstream hash mismatch: {filename}")
        (work / filename).write_bytes(payload)
    # The rows service serves current data, so explicitly check its repository
    # revision before AND after fetching. Do not assume a revision query pins it.
    info_url = "https://huggingface.co/api/datasets/aimo-interp/val-sample"
    check(json.loads(fetch(info_url))["sha"] == DATA_REVISION, "Dataset changed; review revision before continuing")
    payload = fetch("https://datasets-server.huggingface.co/rows?dataset=aimo-interp%2Fval-sample&config=default&split=validation&offset=0&length=100")
    data = json.loads(payload)
    check(not data.get("partial") and len(data["rows"]) == data["num_rows_total"] == 28,
          "Incomplete or changed validation sample")
    check(json.loads(fetch(info_url))["sha"] == DATA_REVISION, "Dataset changed during download")
    (work / "rows.json").write_bytes(payload)


def run_script(work, filename, *arguments, expect_success=True):
    result = subprocess.run([sys.executable, "-I", str(work / filename), *map(str, arguments)],
                            capture_output=True, text=True, encoding="utf-8")
    if expect_success:
        check(result.returncode == 0, result.stderr or result.stdout)
    else:
        check(result.returncode != 0, "Wrong-track ZIP was unexpectedly accepted")
        check("small.txt" in result.stderr, "Wrong-track failure was unrelated to marker")
    return result


def build(track):
    destination = ROOT / "submissions" / f"minimal-always-true-{track}.zip"
    destination.parent.mkdir(exist_ok=True)
    entries = {"solution.py": (ROOT / "solution.py").read_bytes().replace(b"\r\n", b"\n")}
    if track == "small":
        entries["small.txt"] = b""
    with ZipFile(destination, "w", compression=ZIP_STORED) as archive:
        for name, payload in sorted(entries.items()):
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload)
    return destination


def validate(work):
    for filename, sha in BLOBS.items():
        check(blob_sha((work / filename).read_bytes()) == sha, f"Changed harness: {filename}")
    run_script(work, "import_hf_dataset.py", "--source-json", work / "rows.json",
               "--output-dir", work / "sample", "--revision", DATA_REVISION)
    metadata = json.loads((work / "sample/metadata.json").read_bytes())
    spec = importlib.util.spec_from_file_location("minimal", ROOT / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = 0
    for model in MODELS:
        for effort in ("default", "low", "medium", "high"):
            for problems in ([], ["x"], ["repeat", "repeat", "Unicode: π", "x" * 100000]):
                result = module.are_robust(model, effort, problems)
                check(type(result) is list and len(result) == len(problems)
                      and all(type(x) is bool and x is True for x in result), "Interface mismatch")
                check(result == module.are_robust(model, effort, problems), "Non-deterministic output")
                calls += 1
    report = {"method": "always_true", "dataset_revision": DATA_REVISION,
              "rows_sha256": hashlib.sha256((work / "rows.json").read_bytes()).hexdigest(),
              "harness_blob_shas": BLOBS, "python": sys.version,
              "contract_cases": calls, "docker_tested": False,
              "codabench_submission_id": None, "codabench_score": None,
              "paid_compute_usd": 0, "tracks": {}}
    report["sample"] = {key: metadata[key] for key in
                        ("source_rows", "cases", "duplicate_rows_removed", "label_counts")}
    for track in ("main", "small"):
        archive = build(track)
        first_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        check(hashlib.sha256(build(track).read_bytes()).hexdigest() == first_hash, "ZIP is not reproducible")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            submission = tmp / "submission"
            with ZipFile(archive) as bundle:
                expected = {"solution.py", "small.txt"} if track == "small" else {"solution.py"}
                check(set(bundle.namelist()) == expected, "Unexpected ZIP contents")
                check(bundle.testzip() is None, "Corrupt ZIP")
                bundle.extractall(submission)
            start = time.perf_counter()
            run_script(work, "ingestion.py", work / "sample/input", tmp / "res", submission, "--track", track)
            prediction_bytes = (tmp / "res/predictions.jsonl").read_bytes()
            run_script(work, "ingestion.py", work / "sample/input", tmp / "res", submission, "--track", track)
            check(prediction_bytes == (tmp / "res/predictions.jsonl").read_bytes(), "Predictions changed")
            import shutil
            shutil.copytree(work / "sample/reference", tmp / "ref")
            run_script(work, "scoring.py", tmp, tmp / "scores", "--track", track)
            scores = json.loads((tmp / "scores/scores.json").read_bytes())
            check(scores["coverage"] == 1.0 and scores["invalid_predictions"] == 0, "Invalid predictions")
            elapsed = time.perf_counter() - start
            wrong_track = "small" if track == "main" else "main"
            run_script(work, "ingestion.py", work / "sample/input", tmp / "wrong", submission,
                       "--track", wrong_track, expect_success=False)
        report["tracks"][track] = {"zip": archive.name, "sha256": first_hash,
                                    "bytes": archive.stat().st_size, "local_scores": scores,
                                    "two_ingestions_plus_scoring_seconds": elapsed,
                                    "wrong_track_rejected": True}
    report["note"] = "Both variants use 24 unique public cases from 28 source rows; these are not private track scores."
    path = ROOT / "experiments/minimal-validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work/minimal")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if args.prepare:
        prepare(work)
    validate(work)
