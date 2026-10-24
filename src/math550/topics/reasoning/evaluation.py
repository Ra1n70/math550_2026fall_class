"""Exact-answer, sampling, and calibration-aware reasoning evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

from .data import MathReasoningExample, canonicalize_numeric_answer, extract_final_answer


@dataclass(frozen=True)
class CompletionRecord:
    identifier: str
    sample_index: int
    completion: str
    reference_answer: str
    predicted_answer: str | None
    correct: bool
    marked_format: bool
    generated_tokens: int | None = None
    latency_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def majority_vote(answers: Iterable[str | None]) -> str | None:
    """Return a deterministic majority answer; ties follow first occurrence."""

    usable = [answer for answer in answers if answer is not None]
    if not usable:
        return None
    counts = Counter(usable)
    maximum = max(counts.values())
    return next(answer for answer in usable if counts[answer] == maximum)


def verifiable_reward(
    completion: str,
    reference_answer: str,
    *,
    answer_reward: float = 1.0,
    format_bonus: float = 0.1,
    invalid_penalty: float = -0.1,
) -> float:
    """Binary exact-answer reward plus a small, independently visible format signal."""

    predicted = extract_final_answer(completion, allow_fallback=True)
    if predicted is None:
        return invalid_penalty
    correct = predicted == canonicalize_numeric_answer(reference_answer)
    marked = "####" in completion and extract_final_answer(completion, allow_fallback=False) is not None
    return (answer_reward if correct else 0.0) + (format_bonus if marked else 0.0)


def build_completion_records(
    examples: Sequence[MathReasoningExample],
    completions: Sequence[Sequence[str]],
    *,
    generated_tokens: Sequence[Sequence[int]] | None = None,
    latencies: Sequence[Sequence[float]] | None = None,
) -> list[CompletionRecord]:
    if len(examples) != len(completions):
        raise ValueError("examples and completion groups must have matching lengths")
    records: list[CompletionRecord] = []
    for problem_index, (example, group) in enumerate(zip(examples, completions, strict=True)):
        reference = canonicalize_numeric_answer(example.answer)
        if reference is None:
            raise ValueError(f"reference answer is not numeric: {example.answer!r}")
        for sample_index, completion in enumerate(group):
            predicted = extract_final_answer(completion, allow_fallback=True)
            records.append(
                CompletionRecord(
                    identifier=example.identifier,
                    sample_index=sample_index,
                    completion=completion,
                    reference_answer=reference,
                    predicted_answer=predicted,
                    correct=predicted == reference,
                    marked_format=(
                        "####" in completion
                        and extract_final_answer(completion, allow_fallback=False) is not None
                    ),
                    generated_tokens=(
                        generated_tokens[problem_index][sample_index]
                        if generated_tokens is not None
                        else None
                    ),
                    latency_seconds=(
                        latencies[problem_index][sample_index] if latencies is not None else None
                    ),
                )
            )
    return records


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials < 1:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_completion_records(records: Sequence[CompletionRecord]) -> dict[str, object]:
    """Summarize greedy or multi-sample results without conflating sample and problem units."""

    if not records:
        raise ValueError("at least one completion record is required")
    groups: dict[str, list[CompletionRecord]] = {}
    for record in records:
        groups.setdefault(record.identifier, []).append(record)
    first_correct = [int(group[0].correct) for group in groups.values()]
    oracle_correct = [int(any(record.correct for record in group)) for group in groups.values()]
    majority_correct: list[int] = []
    agreement: list[float] = []
    for group in groups.values():
        vote = majority_vote(record.predicted_answer for record in group)
        majority_correct.append(int(vote == group[0].reference_answer))
        parsed = [record.predicted_answer for record in group if record.predicted_answer is not None]
        agreement.append(max(Counter(parsed).values()) / len(group) if parsed else 0.0)
    successes = sum(first_correct)
    low, high = _wilson_interval(successes, len(first_correct))
    token_counts = [record.generated_tokens for record in records if record.generated_tokens is not None]
    latencies = [record.latency_seconds for record in records if record.latency_seconds is not None]
    return {
        "problems": len(groups),
        "samples_per_problem": max(len(group) for group in groups.values()),
        "first_sample_accuracy": successes / len(first_correct),
        "first_sample_accuracy_wilson_95_low": low,
        "first_sample_accuracy_wilson_95_high": high,
        "majority_vote_accuracy": sum(majority_correct) / len(majority_correct),
        "pass_at_k_oracle_accuracy": sum(oracle_correct) / len(oracle_correct),
        "parse_rate": sum(record.predicted_answer is not None for record in records) / len(records),
        "marked_format_rate": sum(record.marked_format for record in records) / len(records),
        "mean_group_agreement": sum(agreement) / len(agreement),
        "mean_generated_tokens": (
            sum(token_counts) / len(token_counts) if token_counts else None
        ),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
    }
