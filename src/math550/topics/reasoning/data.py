"""Public math-reasoning data contracts, prompts, and SFT batches."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import random
import re
from typing import Any, Literal, Sequence

import torch
from torch.utils.data import Dataset


FINAL_ANSWER_MARKER = "####"
ReasoningPromptStyle = Literal["direct", "cot"]
DIRECT_SYSTEM_PROMPT = (
    "You are a careful mathematics tutor. Answer directly without showing intermediate steps and finish "
    "with exactly '#### ' followed by the numeric answer."
)
COT_SYSTEM_PROMPT = (
    "You are a careful mathematics tutor. Work through the problem step by step, "
    "check the arithmetic, and finish with exactly '#### ' followed by the numeric answer."
)
# Backward-compatible name for the course's default worked-solution behavior.
SYSTEM_PROMPT = COT_SYSTEM_PROMPT
_CALCULATION_ANNOTATION = re.compile(r"<<.*?>>")
_MARKED_ANSWER = re.compile(r"####\s*([^\n\r]+)")
_BOXED_ANSWER = re.compile(r"\\boxed\{([^{}]+)\}")
_LABELED_ANSWER = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([-+$€£\d][^\n\r.;]*)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?[$€£]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*(?:\.\d+)?)?%?")


@dataclass(frozen=True)
class MathReasoningExample:
    """One question, worked rationale, and verifiable final answer."""

    identifier: str
    question: str
    rationale: str
    answer: str
    source: str
    equation: str | None = None

    @property
    def response(self) -> str:
        rationale = self.rationale.strip()
        if rationale:
            return f"{rationale}\n{FINAL_ANSWER_MARKER} {self.answer}"
        return f"{FINAL_ANSWER_MARKER} {self.answer}"


def canonicalize_numeric_answer(value: str | int | float | Decimal | None) -> str | None:
    """Map common numeric spellings to a stable exact-match representation."""

    if value is None:
        return None
    text = str(value).strip().strip("`*_").rstrip(".,;:! ")
    text = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    if not text:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", maxsplit=1)
            denominator_value = Decimal(denominator)
            if denominator_value == 0:
                return None
            number = Decimal(numerator) / denominator_value
        else:
            number = Decimal(text)
        if is_percent:
            number /= Decimal(100)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_final_answer(text: str, *, allow_fallback: bool = True) -> str | None:
    """Extract a numeric answer, preferring the course's explicit ``####`` contract."""

    marked = _MARKED_ANSWER.findall(text)
    candidates: list[str] = []
    if marked:
        candidates.extend(reversed(marked))
    else:
        boxed = _BOXED_ANSWER.findall(text)
        labeled = _LABELED_ANSWER.findall(text)
        candidates.extend(reversed(boxed or labeled))
    for candidate in candidates:
        numbers = _NUMBER.findall(candidate)
        if numbers:
            normalized = canonicalize_numeric_answer(numbers[-1])
            if normalized is not None:
                return normalized
    if allow_fallback:
        numbers = _NUMBER.findall(text)
        if numbers:
            return canonicalize_numeric_answer(numbers[-1])
    return None


def remove_calculator_annotations(text: str) -> str:
    """Remove GSM8K's ``<<expression=result>>`` tool annotations, retaining prose results."""

    cleaned = _CALCULATION_ANNOTATION.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return "\n".join(line.strip() for line in cleaned.splitlines()).strip()


def _split_gsm8k_answer(answer_text: str) -> tuple[str, str]:
    if FINAL_ANSWER_MARKER not in answer_text:
        raise ValueError("GSM8K answer is missing the #### final-answer marker")
    rationale, raw_answer = answer_text.rsplit(FINAL_ANSWER_MARKER, maxsplit=1)
    answer = canonicalize_numeric_answer(raw_answer)
    if answer is None:
        raise ValueError(f"GSM8K final answer is not numeric: {raw_answer!r}")
    return remove_calculator_annotations(rationale), answer


def load_gsm8k_jsonl(path: Path, *, split: str) -> list[MathReasoningExample]:
    """Load an official GSM8K JSONL split with strict schema checks."""

    examples: list[MathReasoningExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            if set(record) != {"question", "answer"}:
                raise ValueError(f"unexpected GSM8K fields on line {index + 1}: {sorted(record)}")
            rationale, answer = _split_gsm8k_answer(record["answer"])
            examples.append(
                MathReasoningExample(
                    identifier=f"gsm8k-{split}-{index:05d}",
                    question=record["question"].strip(),
                    rationale=rationale,
                    answer=answer,
                    source="GSM8K",
                )
            )
    if not examples:
        raise ValueError(f"no GSM8K examples found in {path}")
    return examples


def load_svamp_json(path: Path) -> list[MathReasoningExample]:
    """Load the official SVAMP challenge set used by the transfer homework."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("SVAMP must be a JSON list")
    examples: list[MathReasoningExample] = []
    for index, record in enumerate(records):
        answer = canonicalize_numeric_answer(record.get("Answer"))
        if answer is None:
            raise ValueError(f"SVAMP record {index} has a non-numeric answer")
        body = str(record.get("Body", "")).strip()
        question = str(record.get("Question", "")).strip()
        examples.append(
            MathReasoningExample(
                identifier=f"svamp-{record.get('ID', index)}",
                question=f"{body} {question}".strip(),
                rationale="",
                answer=answer,
                source="SVAMP",
                equation=str(record.get("Equation", "")).strip() or None,
            )
        )
    if not examples:
        raise ValueError(f"no SVAMP examples found in {path}")
    return examples


def split_training_validation(
    examples: Sequence[MathReasoningExample],
    *,
    validation_size: int = 512,
    seed: int = 550,
) -> tuple[list[MathReasoningExample], list[MathReasoningExample]]:
    """Create a reproducible validation split while leaving official test data untouched."""

    if not 0 < validation_size < len(examples):
        raise ValueError("validation_size must be between zero and the dataset size")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_size])
    training = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return training, validation


def reasoning_messages(
    question: str,
    *,
    prompt_style: ReasoningPromptStyle = "cot",
) -> list[dict[str, str]]:
    if prompt_style not in ("direct", "cot"):
        raise ValueError("prompt_style must be 'direct' or 'cot'")
    system_prompt = COT_SYSTEM_PROMPT if prompt_style == "cot" else DIRECT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def format_reasoning_prompt(
    tokenizer: Any,
    question: str,
    *,
    prompt_style: ReasoningPromptStyle = "cot",
) -> str:
    """Format a direct-answer or zero-shot-CoT prompt with the native chat template."""

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            reasoning_messages(question, prompt_style=prompt_style),
            tokenize=False,
            add_generation_prompt=True,
        )
    system_prompt = COT_SYSTEM_PROMPT if prompt_style == "cot" else DIRECT_SYSTEM_PROMPT
    return f"System: {system_prompt}\nUser: {question.strip()}\nAssistant:"


def _response_ids_preserving_answer(
    tokenizer: Any,
    example: MathReasoningExample,
    available: int,
) -> list[int]:
    eos_id = tokenizer.eos_token_id
    suffix_text = f"\n{FINAL_ANSWER_MARKER} {example.answer}"
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    rationale_ids = tokenizer(example.rationale.strip(), add_special_tokens=False)["input_ids"]
    reserve = len(suffix_ids) + (1 if eos_id is not None else 0)
    if available <= reserve:
        kept = suffix_ids[-max(available - (1 if eos_id is not None else 0), 1) :]
    else:
        kept = rationale_ids[: available - reserve] + suffix_ids
    if eos_id is not None and len(kept) < available:
        kept.append(eos_id)
    return kept[:available]


class ReasoningSFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize worked solutions and mask loss on prompt and padding tokens."""

    def __init__(
        self,
        examples: Sequence[MathReasoningExample],
        tokenizer: Any,
        *,
        max_length: int = 384,
    ) -> None:
        if max_length < 64:
            raise ValueError("max_length must be at least 64")
        self.rows: list[dict[str, torch.Tensor]] = []
        self.identifiers: list[str] = []
        for example in examples:
            prompt = format_reasoning_prompt(tokenizer, example.question)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) >= max_length - 8:
                prompt_ids = prompt_ids[: max_length // 2]
            response_ids = _response_ids_preserving_answer(
                tokenizer, example, max_length - len(prompt_ids)
            )
            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids.copy()
            if not response_ids or len(input_ids) > max_length:
                continue
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                }
            )
            self.identifiers.append(example.identifier)
        if not self.rows:
            raise ValueError("no usable reasoning SFT examples were produced")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


@dataclass(frozen=True)
class ReasoningCollator:
    """Right-pad variable-length causal-LM batches."""

    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def __call__(self, rows: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        maximum = max(int(row["input_ids"].numel()) for row in rows)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            maximum = ((maximum + multiple - 1) // multiple) * multiple
        batch: dict[str, list[torch.Tensor]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for row in rows:
            padding = maximum - int(row["input_ids"].numel())
            batch["input_ids"].append(
                torch.nn.functional.pad(row["input_ids"], (0, padding), value=self.pad_token_id)
            )
            batch["labels"].append(
                torch.nn.functional.pad(row["labels"], (0, padding), value=-100)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(row["attention_mask"], (0, padding), value=0)
            )
        return {name: torch.stack(tensors) for name, tensors in batch.items()}
