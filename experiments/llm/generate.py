"""Sample from a course GPT checkpoint without an inference framework."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
import sys

import torch

from math550.topics.llm import (
    load_gpt_checkpoint,
    load_tokenizer,
    resolve_generation_vocab_size,
    resolve_torch_device,
    seed_everything,
    tokenizer_sha256,
)


def sampling_seed(requested_seed: int | None) -> int:
    """Return an explicit seed or fresh process-independent sampling entropy."""

    if requested_seed is not None:
        if not 0 <= requested_seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        return requested_seed
    return secrets.randbits(32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        help="reproduce a sample; omitted means a fresh random seed on every run",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer.json used for training; defaults to the checkpoint directory",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    actual_seed = sampling_seed(args.seed)
    device = resolve_torch_device(args.device)
    model, payload = load_gpt_checkpoint(args.checkpoint, device)
    model.eval()
    tokenizer_path = args.tokenizer or args.checkpoint.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    valid_vocab_size = resolve_generation_vocab_size(
        model_vocab_size=model.config.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        checkpoint_metadata=payload.get("metadata"),
        tokenizer_fingerprint=tokenizer_sha256(tokenizer_path),
    )
    if valid_vocab_size < model.config.vocab_size:
        print(
            f"model_vocab_size={model.config.vocab_size} "
            f"tokenizer_vocab_size={valid_vocab_size}; masking "
            f"{model.config.vocab_size - valid_vocab_size} padded output rows",
            file=sys.stderr,
        )
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    # Seed immediately before sampling so checkpoint construction cannot consume
    # values from the stream that controls generated tokens.
    seed_everything(actual_seed)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
        valid_vocab_size=valid_vocab_size,
    )
    print(f"sampling_seed={actual_seed}", file=sys.stderr)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
