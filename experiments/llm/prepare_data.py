"""Train the course tokenizer and build leakage-resistant token streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from math550.topics.llm import BytePairTokenizer, ByteTokenizer


def selected_books(
    data_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """Use the exact randomized selection recorded by the downloader."""

    manifest_path = data_root / "download_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        books = manifest.get("books")
        if books:
            return [dict(book) for book in books], manifest.get("book_selection")
    raise FileNotFoundError(
        f"{manifest_path} has no selected books; run "
        "experiments.llm.download_data --dataset books first"
    )


def _finish_stream(
    path: Path, count: int, minimum: int, maximum: int
) -> dict[str, object]:
    if count < 2:
        raise ValueError(f"token stream {path} contains fewer than two tokens")
    return {
        "path": str(path),
        "tokens": count,
        "dtype": "uint16",
        "minimum_id": minimum,
        "maximum_id": maximum,
        "bytes": path.stat().st_size,
    }


def run(
    data_root: Path,
    *,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 512,
    tokenizer_characters: int = 500_000,
) -> Path:
    books, selection = selected_books(data_root)
    split_counts = {
        split: sum(book.get("split") == split for book in books)
        for split in ("train", "validation")
    }
    if split_counts["train"] < 1 or split_counts["validation"] < 1:
        raise ValueError(
            "the selected corpus must contain training and validation books"
        )
    missing = [
        book["id"]
        for book in books
        if not (data_root / "raw" / "books" / f"pg{book['id']}.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing Project Gutenberg books {missing}; "
            "run experiments.llm.download_data first"
        )
    train_texts = (
        (data_root / "raw" / "books" / f"pg{book['id']}.txt").read_text(
            encoding="utf-8"
        )
        for book in books
        if book["split"] == "train"
    )
    tokenizer = (
        BytePairTokenizer.train(
            train_texts,
            vocab_size=vocab_size,
            max_characters=tokenizer_characters,
        )
        if tokenizer_kind == "bpe"
        else ByteTokenizer()
    )
    processed = data_root / "processed"
    tokenizer_path = processed / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    stream_paths = {
        split: processed / f"{split}.bin" for split in ("train", "validation")
    }
    stream_counts = {split: 0 for split in stream_paths}
    stream_minimum = {split: np.iinfo(np.uint16).max for split in stream_paths}
    stream_maximum = {split: 0 for split in stream_paths}
    per_book: list[dict[str, object]] = []
    handles = {split: path.open("wb") for split, path in stream_paths.items()}
    try:
        for book in books:
            path = data_root / "raw" / "books" / f"pg{book['id']}.txt"
            text = path.read_text(encoding="utf-8")
            token_ids = tokenizer.encode(text, add_eos=True)
            array = np.asarray(token_ids)
            if array.ndim != 1 or len(array) < 2:
                raise ValueError(f"PG{book['id']} produced fewer than two tokens")
            if array.min() < 0 or array.max() > np.iinfo(np.uint16).max:
                raise ValueError("course token files require ids in uint16 range")
            split = str(book["split"])
            array.astype(np.uint16).tofile(handles[split])
            stream_counts[split] += len(array)
            stream_minimum[split] = min(stream_minimum[split], int(array.min()))
            stream_maximum[split] = max(stream_maximum[split], int(array.max()))
            per_book.append(
                {
                    "id": book["id"],
                    "title": book["title"],
                    "author": book.get("author", "Unknown"),
                    "split": split,
                    "selection_source": book.get(
                        "selection_source", "fixed_course_list"
                    ),
                    "utf8_bytes": path.stat().st_size,
                    "tokens": len(token_ids),
                    "bytes_per_token": path.stat().st_size / len(token_ids),
                }
            )
    finally:
        for handle in handles.values():
            handle.close()
    stream_metadata = {
        split: _finish_stream(
            stream_paths[split],
            stream_counts[split],
            stream_minimum[split],
            stream_maximum[split],
        )
        for split in stream_paths
    }
    metadata = {
        "tokenizer": tokenizer_kind,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_training_characters": (
            tokenizer_characters if tokenizer_kind == "bpe" else None
        ),
        "split_policy": (
            "manifest-selected whole books; tokenizer trained on training books only"
        ),
        "book_count": len(books),
        "training_book_count": split_counts["train"],
        "validation_book_count": split_counts["validation"],
        "download_selection": selection,
        "streams": stream_metadata,
        "books": per_book,
    }
    metadata_path = processed / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--tokenizer", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-characters", type=int, default=500_000)
    args = parser.parse_args()
    print(
        run(
            args.data_root,
            tokenizer_kind=args.tokenizer,
            vocab_size=args.vocab_size,
            tokenizer_characters=args.tokenizer_characters,
        )
    )


if __name__ == "__main__":
    main()
