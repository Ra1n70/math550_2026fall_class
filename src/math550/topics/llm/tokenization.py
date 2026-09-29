"""Transparent byte and byte-pair tokenizers for the LLM course module.

The implementation deliberately avoids pretrained tokenizers.  UTF-8 bytes make
every input representable; byte-pair encoding (BPE) then learns a compact list of
frequent adjacent-byte merges from the training corpus only.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)
SPECIAL_TOKEN_IDS = {token: 256 + index for index, token in enumerate(SPECIAL_TOKENS)}
_PRETOKEN_PATTERN = re.compile(r"\s+|[^\w\s]+|\w+", flags=re.UNICODE)
_SPECIAL_PATTERN = re.compile(
    "(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")"
)


def _split_special(text: str) -> list[str]:
    return [piece for piece in _SPECIAL_PATTERN.split(text) if piece]


def _merge_sequence(
    sequence: Sequence[int], pair: tuple[int, int], merged_id: int
) -> tuple[int, ...]:
    """Replace non-overlapping instances of ``pair`` from left to right."""

    result: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            result.append(merged_id)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return tuple(result)


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with five course control tokens."""

    vocab_size = 256 + len(SPECIAL_TOKENS)
    eos_id = SPECIAL_TOKEN_IDS["<|endoftext|>"]
    pad_id = SPECIAL_TOKEN_IDS["<|pad|>"]
    system_id = SPECIAL_TOKEN_IDS["<|system|>"]
    user_id = SPECIAL_TOKEN_IDS["<|user|>"]
    assistant_id = SPECIAL_TOKEN_IDS["<|assistant|>"]

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        for piece in _split_special(text):
            if piece in SPECIAL_TOKEN_IDS:
                token_ids.append(SPECIAL_TOKEN_IDS[piece])
            else:
                token_ids.extend(piece.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, show_special: bool = True) -> str:
        inverse_special = {value: key for key, value in SPECIAL_TOKEN_IDS.items()}
        pieces: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if pending:
                pieces.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in token_ids:
            value = int(token_id)
            if 0 <= value < 256:
                pending.append(value)
            elif value in inverse_special:
                flush()
                if show_special:
                    pieces.append(inverse_special[value])
            else:
                raise ValueError(f"token id {value} is outside the tokenizer vocabulary")
        flush()
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"kind": "byte", "special_tokens": SPECIAL_TOKENS}, indent=2),
            encoding="utf-8",
        )


class BytePairTokenizer(ByteTokenizer):
    """Byte-level BPE with explicit, serializable merge ranks.

    BPE never merges across the simple regex pre-token boundaries.  This keeps
    training understandable and prevents a frequent space or punctuation pattern
    from absorbing unrelated neighboring words.
    """

    def __init__(self, merges: Sequence[tuple[int, int, int]]) -> None:
        self.merges = tuple((int(a), int(b), int(c)) for a, b, c in merges)
        expected = 256 + len(SPECIAL_TOKENS)
        vocabulary: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for rank, (left, right, merged) in enumerate(self.merges):
            if merged != expected + rank:
                raise ValueError("BPE merged token ids must be contiguous and rank ordered")
            if left not in vocabulary or right not in vocabulary:
                raise ValueError("BPE merge references a token not yet defined")
            vocabulary[merged] = vocabulary[left] + vocabulary[right]
        self._token_bytes = vocabulary
        self._merge_rank = {
            (left, right): (rank, merged)
            for rank, (left, right, merged) in enumerate(self.merges)
        }
        self.vocab_size = expected + len(self.merges)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 512,
        max_characters: int | None = 500_000,
        min_pair_frequency: int = 2,
    ) -> "BytePairTokenizer":
        """Learn merge rules from text without using a tokenizer library."""

        base_size = 256 + len(SPECIAL_TOKENS)
        if vocab_size < base_size:
            raise ValueError(f"vocab_size must be at least {base_size}")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be positive")

        counts: Counter[tuple[int, ...]] = Counter()
        characters_seen = 0
        stop = False
        for text in texts:
            for piece in _PRETOKEN_PATTERN.findall(text):
                if max_characters is not None:
                    remaining = max_characters - characters_seen
                    if remaining <= 0:
                        stop = True
                        break
                    piece = piece[:remaining]
                if not piece:
                    continue
                characters_seen += len(piece)
                counts[tuple(piece.encode("utf-8"))] += 1
            if stop:
                break
        if not counts:
            raise ValueError("cannot train BPE on empty text")

        merges: list[tuple[int, int, int]] = []
        next_id = base_size
        while next_id < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence, frequency in counts.items():
                sequence_pairs = Counter(zip(sequence, sequence[1:]))
                for pair, occurrences in sequence_pairs.items():
                    pair_counts[pair] += frequency * occurrences
            if not pair_counts:
                break
            best_pair, best_frequency = min(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if best_frequency < min_pair_frequency:
                break
            updated: Counter[tuple[int, ...]] = Counter()
            for sequence, frequency in counts.items():
                updated[_merge_sequence(sequence, best_pair, next_id)] += frequency
            counts = updated
            merges.append((best_pair[0], best_pair[1], next_id))
            next_id += 1
        return cls(merges)

    def _encode_piece(self, piece: str) -> list[int]:
        sequence = list(piece.encode("utf-8"))
        while len(sequence) > 1:
            candidates = [
                (self._merge_rank[pair][0], pair, self._merge_rank[pair][1])
                for pair in zip(sequence, sequence[1:])
                if pair in self._merge_rank
            ]
            if not candidates:
                break
            _, pair, merged_id = min(candidates, key=lambda item: item[0])
            sequence = list(_merge_sequence(sequence, pair, merged_id))
        return sequence

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        for special_piece in _split_special(text):
            if special_piece in SPECIAL_TOKEN_IDS:
                token_ids.append(SPECIAL_TOKEN_IDS[special_piece])
                continue
            for piece in _PRETOKEN_PATTERN.findall(special_piece):
                token_ids.extend(self._encode_piece(piece))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, show_special: bool = True) -> str:
        inverse_special = {value: key for key, value in SPECIAL_TOKEN_IDS.items()}
        pieces: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if pending:
                pieces.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in token_ids:
            value = int(token_id)
            if value in self._token_bytes:
                pending.extend(self._token_bytes[value])
            elif value in inverse_special:
                flush()
                if show_special:
                    pieces.append(inverse_special[value])
            else:
                raise ValueError(f"token id {value} is outside the tokenizer vocabulary")
        flush()
        return "".join(pieces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "byte_bpe",
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
            "merges": [list(merge) for merge in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "byte_bpe":
            raise ValueError("tokenizer file is not byte-level BPE")
        if tuple(payload.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("tokenizer special-token contract does not match this course")
        return cls([tuple(merge) for merge in payload["merges"]])


def load_tokenizer(path: Path | None = None) -> ByteTokenizer | BytePairTokenizer:
    """Load a saved tokenizer, or return the no-training byte baseline."""

    if path is None:
        return ByteTokenizer()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "byte":
        return ByteTokenizer()
    return BytePairTokenizer.load(path)


def tokenizer_sha256(path: Path) -> str:
    """Return the exact tokenizer-file fingerprint stored with new checkpoints."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_generation_vocab_size(
    *,
    model_vocab_size: int,
    tokenizer_vocab_size: int,
    checkpoint_metadata: Mapping[str, object] | None = None,
    tokenizer_fingerprint: str | None = None,
) -> int:
    """Validate a checkpoint/tokenizer pair and return the sampleable prefix.

    A model may deliberately have extra output rows, but every generated token
    must be defined by the tokenizer.  When checkpoint metadata records the
    tokenizer contract, a different sidecar file is rejected rather than being
    mistaken for harmless model padding.
    """

    model_size = int(model_vocab_size)
    tokenizer_size = int(tokenizer_vocab_size)
    if model_size < 1 or tokenizer_size < 1:
        raise ValueError("model and tokenizer vocabulary sizes must be positive")
    if tokenizer_size > model_size:
        raise ValueError(
            f"tokenizer vocabulary ({tokenizer_size}) exceeds checkpoint model "
            f"vocabulary ({model_size}); use the tokenizer saved with this checkpoint"
        )

    metadata = checkpoint_metadata or {}
    declared_size = metadata.get("tokenizer_vocab_size")
    if declared_size is None:
        corpus = metadata.get("corpus_metadata")
        if isinstance(corpus, Mapping):
            declared_size = corpus.get("vocab_size")
    if declared_size is not None:
        try:
            expected_size = int(declared_size)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint tokenizer_vocab_size is invalid") from error
        if expected_size != tokenizer_size:
            raise ValueError(
                f"checkpoint expects tokenizer vocabulary {expected_size}, but the "
                f"selected tokenizer defines {tokenizer_size}; pass --tokenizer with "
                "the tokenizer.json used for this training run"
            )

    declared_fingerprint = metadata.get("tokenizer_sha256")
    if (
        declared_fingerprint is not None
        and tokenizer_fingerprint is not None
        and str(declared_fingerprint) != tokenizer_fingerprint
    ):
        raise ValueError(
            "tokenizer SHA-256 does not match the checkpoint; pass --tokenizer with "
            "the tokenizer.json used for this training run"
        )
    return tokenizer_size
