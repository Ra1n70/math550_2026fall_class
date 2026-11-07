"""Prepare the fixed, public instruction records for LLM homework Problem 4.

Run from the repository root:
    uv run --extra llm python homework/llm/prepare_instruction_data.py

This helper prepares data, not a model or an instructor solution.  It uses the
course's fixed ByteTokenizer and instruction template; it does not train a
tokenizer, change an existing book corpus, or download pretrained weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from math550.topics.llm import ByteTokenizer, InstructionExample
from math550.topics.llm.data import format_instruction_prompt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "llm" / "raw" / "homework" / "rasbt_instruction_data.json"
DEFAULT_MANIFEST = Path(__file__).with_name("instruction_manifest.json")
SOURCE_COMMIT = "f49cad747931fe6af0a7d2802cd480d1c1a0ec8e"
SOURCE_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
    f"{SOURCE_COMMIT}/ch07/01_main-chapter-code/instruction-data.json"
)
SOURCE_SHA256 = "7e09a9af353722ca15c18bc580f093f9c39d3178335a75177319cb1e1d163944"
TEST_INDICES = (1, 2, 3, 4, 8, 9, 10, 11)  # Zero-based source JSON indices.
CONTEXT_LENGTH = 256
MAX_PROMPT_TOKENS = 192


def _validate_source(payload: bytes) -> list[dict[str, str]]:
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            "Instruction data SHA256 mismatch. The existing cache is not modified. "
            f"Expected {SOURCE_SHA256}, found {actual_hash}."
        )
    records = json.loads(payload)
    if not isinstance(records, list) or len(records) != 1100:
        raise ValueError("The pinned source must contain exactly 1,100 records.")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) for key in ("instruction", "input", "output")
        ):
            raise ValueError(f"Invalid instruction/input/output fields at source index {index}.")
    return records


def load_source_records(
    cache_path: Path = DEFAULT_CACHE, *, download: bool = True
) -> list[dict[str, str]]:
    """Return the verified source records, downloading only if the cache is absent."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        return _validate_source(cache_path.read_bytes())
    if not download:
        raise FileNotFoundError(
            f"Missing {cache_path}. Run homework/llm/prepare_instruction_data.py first."
        )
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "MATH550-instruction-homework/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    records = _validate_source(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a pre-existing corpus/cache, including one created meanwhile.
    try:
        with cache_path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError:
        return _validate_source(cache_path.read_bytes())
    return records


def as_instruction_example(record: dict[str, str]) -> InstructionExample:
    """Map the book's instruction/input/output fields to the course data contract."""

    return InstructionExample(
        instruction=record["instruction"],
        context=record["input"],
        response=record["output"],
        category="raschka-ch07-homework",
    )


def _prompt_key(record: dict[str, str]) -> tuple[str, str]:
    return tuple(
        " ".join(record[key].split()).casefold() for key in ("instruction", "input")
    )


def token_lengths(record: dict[str, str]) -> dict[str, int]:
    """Count the exact course prompt and response tokens, including response EOS."""

    tokenizer = ByteTokenizer()
    example = as_instruction_example(record)
    prompt_length = len(tokenizer.encode(format_instruction_prompt(example)))
    response_length = len(tokenizer.encode(example.response.strip(), add_eos=True))
    return {
        "prompt_tokens": prompt_length,
        "response_tokens_including_eos": response_length,
        "combined_tokens": prompt_length + response_length,
    }


def load_instruction_splits(
    cache_path: Path = DEFAULT_CACHE, *, download: bool = True
) -> tuple[dict[str, list[InstructionExample]], dict[str, object]]:
    """Return train/validation/test course examples and their reproducible metadata.

    Reserve eight specified test prompts, exclude normalized duplicate prompts,
    and take the first 40 remaining eligible source records: 32 training and eight
    validation.  All combined sequences fit without SFTDataset truncation.
    """

    records = load_source_records(cache_path, download=download)
    lengths = [token_lengths(record) for record in records]

    def eligible(index: int) -> bool:
        counts = lengths[index]
        return (
            counts["prompt_tokens"] <= MAX_PROMPT_TOKENS
            and counts["combined_tokens"] <= CONTEXT_LENGTH + 1
        )

    test_keys = {_prompt_key(records[index]) for index in TEST_INDICES}
    if len(test_keys) != len(TEST_INDICES):
        raise ValueError("The fixed test set contains duplicate normalized prompts.")
    if not all(eligible(index) for index in TEST_INDICES):
        raise ValueError("A fixed test record exceeds the no-truncation limits.")
    seen = set(test_keys)
    selected: list[int] = []
    for index, record in enumerate(records):
        key = _prompt_key(record)
        if key in seen or not eligible(index):
            continue
        selected.append(index)
        seen.add(key)
        if len(selected) == 40:
            break
    if len(selected) != 40:
        raise ValueError("Not enough distinct eligible source records for the fixed split.")
    source_indices = {
        "train": selected[:32],
        "validation": selected[32:],
        "test": list(TEST_INDICES),
    }
    splits = {
        name: [as_instruction_example(records[index]) for index in indices]
        for name, indices in source_indices.items()
    }
    metadata: dict[str, object] = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_record_count": len(records),
        "index_convention": "zero-based source JSON indices",
        "tokenizer": "course ByteTokenizer, vocabulary size 261; no fitting",
        "context_length": CONTEXT_LENGTH,
        "eligibility": {
            "maximum_prompt_tokens": MAX_PROMPT_TOKENS,
            "maximum_combined_tokens_including_response_eos": CONTEXT_LENGTH + 1,
            "normalization_for_duplicate_prompts": "casefold and collapse whitespace in instruction/input",
            "selection": "reserve test prompts; first 40 remaining eligible unique prompts; first 32 train, next 8 validation",
        },
        "splits": {
            name: {
                "count": len(indices),
                "source_indices": indices,
                "token_lengths": [
                    {"source_index": index, **lengths[index]} for index in indices
                ],
            }
            for name, indices in source_indices.items()
        },
    }
    return splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--offline", action="store_true", help="Require the existing verified cache.")
    args = parser.parse_args()
    existed = args.cache.is_file()
    _, metadata = load_instruction_splits(args.cache, download=not args.offline)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{'Reused verified cache' if existed else 'Downloaded and verified'}: {args.cache}")
    print(f"Source SHA256: {SOURCE_SHA256}")
    for name, split in metadata["splits"].items():
        print(f"{name}: {split['count']} records; source indices {split['source_indices']}")
    print(f"Wrote split metadata: {args.manifest}")


if __name__ == "__main__":
    main()
