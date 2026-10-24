"""Download immutable official GSM8K and SVAMP files and record integrity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from math550.topics.reasoning import load_gsm8k_jsonl, load_svamp_json


USER_AGENT = "MATH550-reasoning-course/1.0 (educational reproducibility exercise)"


def _download(url: str, path: Path) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
        resolved_url = response.geturl()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "requested_url": url,
        "resolved_url": resolved_url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def run(data_root: Path = Path("data/reasoning"), *, force: bool = False) -> Path:
    sources = json.loads((data_root / "sources.json").read_text(encoding="utf-8"))
    records: dict[str, object] = {"sources_file": str(data_root / "sources.json"), "files": {}}
    for name, source in sources["datasets"].items():
        path = data_root / source["local_path"]
        if force or not path.is_file():
            record = _download(source["download_url"], path)
            record["status"] = "downloaded"
        else:
            payload = path.read_bytes()
            record = {
                "path": str(path),
                "requested_url": source["download_url"],
                "resolved_url": source["download_url"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": "existing",
            }
        expected = source.get("sha256")
        if expected and record["sha256"] != expected:
            raise RuntimeError(
                f"checksum mismatch for {name}: expected {expected}, observed {record['sha256']}"
            )
        records["files"][name] = record

    train = load_gsm8k_jsonl(data_root / sources["datasets"]["gsm8k_train"]["local_path"], split="train")
    test = load_gsm8k_jsonl(data_root / sources["datasets"]["gsm8k_test"]["local_path"], split="test")
    svamp = load_svamp_json(data_root / sources["datasets"]["svamp"]["local_path"])
    expected_counts = {"gsm8k_train": 7_473, "gsm8k_test": 1_319, "svamp": 1_000}
    observed_counts = {"gsm8k_train": len(train), "gsm8k_test": len(test), "svamp": len(svamp)}
    if observed_counts != expected_counts:
        raise RuntimeError(f"dataset count mismatch: {observed_counts}")
    records["validated_examples"] = observed_counts
    records["validated_numeric_answers"] = all(
        example.answer is not None for example in train + test + svamp
    )
    manifest = data_root / "download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reasoning"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(args.data_root, force=args.force))


if __name__ == "__main__":
    main()
