"""Data contracts and sequence packing for pretraining and post-training."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenization import BytePairTokenizer, ByteTokenizer


Tokenizer = ByteTokenizer | BytePairTokenizer


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    response: str
    context: str = ""
    category: str = "unknown"


@dataclass(frozen=True)
class PreferenceExample:
    chosen: str
    rejected: str


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg's distribution header and footer when present."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    start = 0
    end = len(lines)
    for index, line in enumerate(lines[:600]):
        upper = line.upper()
        if "*** START OF THE PROJECT GUTENBERG EBOOK" in upper or (
            "*** START OF THIS PROJECT GUTENBERG EBOOK" in upper
        ):
            start = index + 1
            break
    for index in range(len(lines) - 1, max(start, len(lines) - 800), -1):
        upper = lines[index].upper()
        if "*** END OF THE PROJECT GUTENBERG EBOOK" in upper or (
            "*** END OF THIS PROJECT GUTENBERG EBOOK" in upper
        ):
            end = index
            break
    body = "\n".join(lines[start:end])
    body = "\n".join(line.rstrip() for line in body.splitlines())
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")
    return body.strip() + "\n"


def save_token_stream(token_ids: Sequence[int], path: Path) -> dict[str, object]:
    """Write compact uint16 token IDs and return integrity metadata."""

    array = np.asarray(token_ids)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("token stream must contain at least two ids")
    if array.min() < 0 or array.max() > np.iinfo(np.uint16).max:
        raise ValueError("course token files require ids in uint16 range")
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(np.uint16).tofile(path)
    return {
        "path": str(path),
        "tokens": int(len(array)),
        "dtype": "uint16",
        "minimum_id": int(array.min()),
        "maximum_id": int(array.max()),
        "bytes": int(path.stat().st_size),
    }


class TokenStream:
    """Memory-mapped next-token batches sampled without loading the corpus as int64."""

    def __init__(self, path: Path, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least two")
        self.path = Path(path)
        self.block_size = int(block_size)
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= self.block_size:
            raise ValueError("token file is shorter than one training block")

    def batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0,
            len(self.tokens) - self.block_size,
            (batch_size,),
            generator=generator,
        ).tolist()
        windows = np.stack(
            [np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64) for start in starts]
        )
        batch = torch.from_numpy(windows).to(device=device, dtype=torch.long)
        return batch[:, :-1], batch[:, 1:]


def format_instruction_prompt(example: InstructionExample) -> str:
    context = ""
    if example.context.strip():
        context = f"\n\nContext:\n{example.context.strip()}"
    return (
        "<|system|>\nYou are a concise, careful teaching assistant.<|endoftext|>"
        f"<|user|>\n{example.instruction.strip()}{context}<|endoftext|>"
        "<|assistant|>\n"
    )


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length SFT examples with loss only on assistant response tokens."""

    def __init__(
        self,
        examples: Sequence[InstructionExample],
        tokenizer: Tokenizer,
        max_length: int,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length is too small for instruction formatting")
        self.rows: list[dict[str, torch.Tensor]] = []
        for example in examples:
            prompt_ids = tokenizer.encode(format_instruction_prompt(example))
            response_ids = tokenizer.encode(example.response.strip(), add_eos=True)
            if len(response_ids) >= max_length:
                response_ids = response_ids[: max_length - 1] + [tokenizer.eos_id]
                prompt_ids = []
            elif len(prompt_ids) + len(response_ids) > max_length + 1:
                keep_prompt = max_length + 1 - len(response_ids)
                prompt_ids = prompt_ids[-keep_prompt:]
            combined = prompt_ids + response_ids
            if len(combined) < 2:
                continue
            input_ids = combined[:-1][:max_length]
            targets = combined[1:][:max_length]
            response_start = max(len(prompt_ids) - 1, 0)
            labels = [
                token if index >= response_start else -100
                for index, token in enumerate(targets)
            ]
            attention = [1] * len(input_ids)
            padding = max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_id] * padding)
            labels.extend([-100] * padding)
            attention.extend([0] * padding)
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "targets": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.bool),
                }
            )
        if not self.rows:
            raise ValueError("no usable SFT examples were produced")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized chosen/rejected conversations for reward modeling."""

    def __init__(
        self,
        examples: Sequence[PreferenceExample],
        tokenizer: Tokenizer,
        max_length: int,
    ) -> None:
        self.rows: list[dict[str, torch.Tensor]] = []
        for example in examples:
            row: dict[str, torch.Tensor] = {}
            for name, text in (("chosen", example.chosen), ("rejected", example.rejected)):
                token_ids = tokenizer.encode(text, add_eos=True)[-max_length:]
                mask = [1] * len(token_ids)
                padding = max_length - len(token_ids)
                token_ids.extend([tokenizer.pad_id] * padding)
                mask.extend([0] * padding)
                row[f"{name}_ids"] = torch.tensor(token_ids, dtype=torch.long)
                row[f"{name}_mask"] = torch.tensor(mask, dtype=torch.bool)
            self.rows.append(row)
        if not self.rows:
            raise ValueError("no preference examples were provided")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def load_dolly_jsonl(path: Path, *, limit: int | None = None) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            examples.append(
                InstructionExample(
                    instruction=record["instruction"],
                    context=record.get("context", ""),
                    response=record["response"],
                    category=record.get("category", "unknown"),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_hh_preferences(
    path: Path, *, limit: int | None = None
) -> list[PreferenceExample]:
    opener = gzip.open if path.suffix == ".gz" else open
    examples: list[PreferenceExample] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            examples.append(PreferenceExample(record["chosen"], record["rejected"]))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def deterministic_split(
    examples: Sequence[object], validation_fraction: float = 0.1, seed: int = 550
) -> tuple[list[object], list[object]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    cut = max(1, round(len(indices) * validation_fraction))
    validation = [examples[index] for index in indices[:cut]]
    training = [examples[index] for index in indices[cut:]]
    return training, validation


def write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
