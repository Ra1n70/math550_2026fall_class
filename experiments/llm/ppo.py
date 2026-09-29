"""Run a compact PPO-style RLHF lab using an SFT policy and reward model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from math550.topics.llm import (
    GPT,
    GPTConfig,
    PolicyWithValueHead,
    RewardModel,
    build_token_rewards,
    generalized_advantage_estimate,
    load_dolly_jsonl,
    load_gpt_checkpoint,
    load_tokenizer,
    ppo_loss,
    resolve_generation_vocab_size,
    resolve_torch_device,
    seed_everything,
    stage_tokenizer_for_checkpoints,
    token_log_probs,
    tokenizer_sha256,
)


def _load_reward(path: Path, device: torch.device) -> RewardModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = RewardModel(GPT(GPTConfig(**payload["backbone_config"])))
    model.load_state_dict(payload["reward_model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run(
    policy_checkpoint: Path,
    reward_checkpoint: Path,
    *,
    data_root: Path,
    output_dir: Path,
    device: str = "auto",
    updates: int = 20,
    batch_size: int = 4,
    response_tokens: int = 32,
    kl_coefficient: float = 0.05,
) -> Path:
    seed_everything(550)
    resolved = resolve_torch_device(device)
    policy_gpt, policy_payload = load_gpt_checkpoint(policy_checkpoint, resolved)
    reference = copy.deepcopy(policy_gpt).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy = PolicyWithValueHead(policy_gpt).to(resolved)
    reward_model = _load_reward(reward_checkpoint, resolved)
    tokenizer_path = policy_checkpoint.parent / "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_fingerprint = tokenizer_sha256(tokenizer_path)
    valid_vocab_size = resolve_generation_vocab_size(
        model_vocab_size=policy_gpt.config.vocab_size,
        tokenizer_vocab_size=tokenizer.vocab_size,
        checkpoint_metadata=policy_payload.get("metadata"),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    stage_tokenizer_for_checkpoints(tokenizer_path, output_dir)
    examples = load_dolly_jsonl(data_root / "raw" / "databricks-dolly-15k.jsonl", limit=max(64, updates * batch_size))
    prompts = [
        f"<|user|>\n{example.instruction.strip()}<|endoftext|><|assistant|>\n"
        for example in examples
    ]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    history: list[dict[str, float]] = []

    for update in range(updates):
        selected = prompts[update * batch_size : (update + 1) * batch_size]
        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        policy.policy.eval()
        for prompt_text in selected:
            prompt_ids = tokenizer.encode(prompt_text)[-(policy.policy.config.block_size - response_tokens) :]
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=resolved)
            generated = policy.policy.generate(
                prompt,
                max_new_tokens=response_tokens,
                temperature=0.9,
                top_k=40,
                eos_id=tokenizer.eos_id,
                valid_vocab_size=valid_vocab_size,
            )[0].tolist()
            sequences.append(generated[: policy.policy.config.block_size])
            prompt_lengths.append(len(prompt_ids))
        max_length = max(len(sequence) for sequence in sequences)
        full = torch.full((len(sequences), max_length), tokenizer.pad_id, dtype=torch.long, device=resolved)
        full_mask = torch.zeros_like(full, dtype=torch.bool)
        action_mask = torch.zeros((len(sequences), max_length - 1), dtype=torch.bool, device=resolved)
        for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
            full[row, : len(sequence)] = torch.tensor(sequence, device=resolved)
            full_mask[row, : len(sequence)] = True
            action_mask[row, prompt_length - 1 : len(sequence) - 1] = True
        inputs, targets = full[:, :-1], full[:, 1:]
        with torch.no_grad():
            old_logits, old_values = policy(inputs)
            old_log_probs = token_log_probs(old_logits, targets)
            reference_log_probs = token_log_probs(reference(inputs).logits, targets)
            terminal_rewards = reward_model(full, full_mask)
            rewards = build_token_rewards(
                old_log_probs,
                reference_log_probs,
                terminal_rewards,
                action_mask,
                kl_coefficient=kl_coefficient,
            )
            advantages, returns = generalized_advantage_estimate(
                rewards, old_values, action_mask
            )
        policy.train()
        logits, values = policy(inputs)
        new_log_probs = token_log_probs(logits, targets)
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        objective = ppo_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            values,
            old_values,
            returns,
            action_mask,
            entropy=entropy,
        )
        optimizer.zero_grad(set_to_none=True)
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        row = {
            "update": update + 1,
            "mean_reward": float(terminal_rewards.mean().cpu()),
            "ppo_total": float(objective.total.detach().cpu()),
            "policy_loss": float(objective.policy.detach().cpu()),
            "value_loss": float(objective.value.detach().cpu()),
            "approximate_kl": float(objective.approximate_kl.detach().cpu()),
            "clip_fraction": float(objective.clip_fraction.detach().cpu()),
        }
        history.append(row)
        print(
            f"update={update + 1:03d} reward={row['mean_reward']:+.3f} "
            f"kl={row['approximate_kl']:.4f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ppo_policy.pt"
    torch.save(
        {
            "model_config": policy.policy.config.to_dict(),
            "model_state": policy.policy.state_dict(),
            "value_head_state": policy.value_head.state_dict(),
            "history": history,
            "metadata": {
                "stage": "PPO-style RLHF teaching run",
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_sha256": tokenizer_fingerprint,
                "kl_coefficient": kl_coefficient,
                "warning": "small-model reward hacking is likely; inspect outputs and KL",
            },
        },
        output_path,
    )
    (output_dir / "ppo_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_checkpoint", type=Path)
    parser.add_argument("reward_checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llm/ppo"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--response-tokens", type=int, default=32)
    parser.add_argument("--kl-coefficient", type=float, default=0.05)
    args = parser.parse_args()
    print(run(**vars(args)))


if __name__ == "__main__":
    main()
