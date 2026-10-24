"""Fast public-data and GRPO-objective verification without downloading model weights."""

from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from math550.topics.reasoning import (
    extract_final_answer,
    grpo_loss,
    group_relative_advantages,
    load_gsm8k_jsonl,
    load_svamp_json,
    response_token_log_probs,
    verifiable_reward,
)


def run(output_root: Path = Path("output/reasoning")) -> Path:
    started = time.perf_counter()
    train = load_gsm8k_jsonl(Path("data/reasoning/gsm8k_train.jsonl"), split="train")
    test = load_gsm8k_jsonl(Path("data/reasoning/gsm8k_test.jsonl"), split="test")
    svamp = load_svamp_json(Path("data/reasoning/svamp.json"))
    rewards = torch.tensor([1.1, 0.1, 1.1, -0.1, 0.1, 0.1, 0.1, 0.1])
    advantages = group_relative_advantages(rewards, group_size=4)
    shape = (8, 5)
    new_log_probs = torch.zeros(shape, requires_grad=True)
    old_log_probs = torch.zeros(shape)
    reference_log_probs = torch.full(shape, -0.02)
    action_mask = torch.ones(shape, dtype=torch.bool)
    objective = grpo_loss(
        new_log_probs,
        old_log_probs,
        reference_log_probs,
        advantages,
        action_mask,
    )
    objective.total.backward()
    logits = torch.randn(2, 6, 13, requires_grad=True)
    input_ids = torch.randint(0, 13, (2, 6))
    response_mask = torch.tensor(
        [[False, False, False, True, True, True], [False, False, True, True, True, False]]
    )
    selected, shifted_mask = response_token_log_probs(logits, input_ids, response_mask)
    (selected * shifted_mask).sum().backward()
    result = {
        "scope": "public-data parsing plus synthetic objective/autograd smoke",
        "datasets": {
            "gsm8k_training_rows": len(train),
            "gsm8k_test_rows": len(test),
            "svamp_rows": len(svamp),
            "first_gsm8k_answer": train[0].answer,
            "first_svamp_answer": svamp[0].answer,
        },
        "answer_contract": {
            "marked_extract": extract_final_answer("work\n#### 72"),
            "correct_reward": verifiable_reward("work\n#### 72", "72"),
            "wrong_reward": verifiable_reward("work\n#### 71", "72"),
            "invalid_reward": verifiable_reward("I do not know", "72"),
        },
        "grpo": {
            "advantages": advantages.tolist(),
            "all_equal_group_is_zero": advantages[4:].abs().max().item() == 0.0,
            "loss": float(objective.total.detach()),
            "reference_kl": float(objective.reference_kl.detach()),
            "policy_gradient_finite": bool(torch.isfinite(new_log_probs.grad).all()),
            "token_log_probability_gradient_finite": bool(torch.isfinite(logits.grad).all()),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "smoke_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return path


if __name__ == "__main__":
    print(run())
