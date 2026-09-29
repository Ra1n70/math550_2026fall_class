"""Fast CPU verification spanning tokenization, pretraining, SFT, reward, and PPO."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from math550.topics.llm import (
    BigramLanguageModel,
    BytePairTokenizer,
    GPT,
    InstructionExample,
    PreferenceDataset,
    PreferenceExample,
    RewardModel,
    SFTDataset,
    TokenStream,
    TrainingConfig,
    generalized_advantage_estimate,
    ppo_loss,
    preference_accuracy,
    preference_loss,
    resolve_torch_device,
    save_token_stream,
    seed_everything,
    train_language_model,
    train_supervised_batches,
)

from .config import ExperimentConfig


TRAIN_TEXT = ("""
The careful student reads the evidence, checks the calculation, and states the limitation.
A language model predicts the next token from tokens that came before it.
Attention compares a query with earlier keys, then averages their values.
Pretraining learns broad continuation patterns; supervised fine-tuning teaches a response format.
Preference learning fits a score from chosen and rejected responses.
Reinforcement learning changes the policy, so a reference-model penalty limits drift.
""" * 120).strip()

VALIDATION_TEXT = ("""
The careful analyst separates measured evidence from an unsupported claim.
A causal decoder cannot use a future token when predicting the present token.
Post-training changes behavior, but it does not create reliable knowledge from nothing.
""" * 45).strip()


@torch.no_grad()
def _stream_loss(model, stream: TokenStream, batch_size: int, seed: int, batches: int = 5) -> float:
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    model.eval()
    for _ in range(batches):
        inputs, targets = stream.batch(batch_size, generator=generator, device=torch.device("cpu"))
        output = model(inputs, targets)
        losses.append(float(output.loss))
    model.train()
    return sum(losses) / len(losses)


def _train_bigram(stream: TokenStream, validation: TokenStream, vocab_size: int, steps: int = 30) -> dict[str, float]:
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(551)
    initial = _stream_loss(model, validation, 8, 901)
    for _ in range(steps):
        inputs, targets = stream.batch(8, generator=generator, device=torch.device("cpu"))
        output = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()
    return {"initial_validation_loss": initial, "final_validation_loss": _stream_loss(model, validation, 8, 902)}


@torch.no_grad()
def _sft_loss(model: GPT, loader: DataLoader) -> float:
    model.eval()
    values = [float(model(batch["input_ids"], batch["targets"]).loss) for batch in loader]
    model.train()
    return sum(values) / len(values)


def run(output_root: Path = Path("output")) -> Path:
    seed_everything(550)
    started = time.perf_counter()
    analysis_dir = output_root / "llm" / "analysis"
    smoke_data = output_root / "llm" / "smoke_data"
    checkpoint_dir = output_root / "llm" / "smoke_checkpoints"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BytePairTokenizer.train(
        [TRAIN_TEXT], vocab_size=300, max_characters=120_000, min_pair_frequency=2
    )
    tokenizer.save(smoke_data / "tokenizer.json")
    train_ids = tokenizer.encode(TRAIN_TEXT, add_eos=True)
    validation_ids = tokenizer.encode(VALIDATION_TEXT, add_eos=True)
    save_token_stream(train_ids, smoke_data / "train.bin")
    save_token_stream(validation_ids, smoke_data / "validation.bin")
    experiment = ExperimentConfig(preset="smoke", output_root=output_root)
    model_config = experiment.model(tokenizer.vocab_size)
    training_config = experiment.training("cpu")
    train_stream = TokenStream(smoke_data / "train.bin", model_config.block_size)
    validation_stream = TokenStream(smoke_data / "validation.bin", model_config.block_size)
    train_generator = torch.Generator().manual_seed(550)
    validation_generator = torch.Generator().manual_seed(551)

    def train_batch():
        return train_stream.batch(training_config.batch_size, generator=train_generator, device=torch.device("cpu"))

    def validation_batch():
        return validation_stream.batch(training_config.batch_size, generator=validation_generator, device=torch.device("cpu"))

    model = GPT(model_config)
    history = train_language_model(
        model,
        train_batch,
        validation_batch,
        training_config,
        output_dir=checkpoint_dir,
        metadata={"stage": "automated smoke verification", "tokenizer_path": str(smoke_data / "tokenizer.json")},
    )
    bigram = _train_bigram(train_stream, validation_stream, tokenizer.vocab_size)

    sft_examples = [
        InstructionExample("State the next-token objective in one sentence.", "Minimize cross-entropy for each next token given earlier tokens."),
        InstructionExample("Why use a causal mask?", "It prevents a prediction from reading future tokens."),
        InstructionExample("What does SFT change?", "It teaches response behavior by imitating curated answers."),
        InstructionExample("Name one RLHF risk.", "A policy can exploit errors in the learned reward model."),
    ]
    sft_data = SFTDataset(sft_examples, tokenizer, model_config.block_size)
    sft_loader = DataLoader(sft_data, batch_size=4, shuffle=False)
    sft_initial = _sft_loss(model, sft_loader)
    sft_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    sft_epoch_losses = [
        train_supervised_batches(model, sft_loader, optimizer=sft_optimizer, device=torch.device("cpu"))
        for _ in range(12)
    ]
    sft_final = _sft_loss(model, sft_loader)

    preferences = [
        PreferenceExample("Human: What does a causal mask do? Assistant: It blocks future tokens.", "Human: What does a causal mask do? Assistant: It makes training random."),
        PreferenceExample("Human: Is a tiny book model reliable? Assistant: No; it is a teaching model with narrow data.", "Human: Is a tiny book model reliable? Assistant: Yes, always."),
        PreferenceExample("Human: What should evaluation report? Assistant: Held-out loss and limitations.", "Human: What should evaluation report? Assistant: Only a favorite sample."),
        PreferenceExample("Human: Can SFT add facts absent from data? Assistant: It cannot guarantee that.", "Human: Can SFT add facts absent from data? Assistant: It guarantees all facts."),
    ]
    preference_data = PreferenceDataset(preferences, tokenizer, model_config.block_size)
    preference_loader = DataLoader(preference_data, batch_size=4, shuffle=False)
    reward_model = RewardModel(copy.deepcopy(model))
    reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)
    batch = next(iter(preference_loader))
    with torch.no_grad():
        chosen_before = reward_model(batch["chosen_ids"], batch["chosen_mask"])
        rejected_before = reward_model(batch["rejected_ids"], batch["rejected_mask"])
        reward_loss_initial = float(preference_loss(chosen_before, rejected_before))
    for _ in range(25):
        chosen = reward_model(batch["chosen_ids"], batch["chosen_mask"])
        rejected = reward_model(batch["rejected_ids"], batch["rejected_mask"])
        loss = preference_loss(chosen, rejected)
        reward_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        reward_optimizer.step()
    with torch.no_grad():
        chosen_after = reward_model(batch["chosen_ids"], batch["chosen_mask"])
        rejected_after = reward_model(batch["rejected_ids"], batch["rejected_mask"])
        reward_loss_final = float(preference_loss(chosen_after, rejected_after))
        reward_accuracy = float(preference_accuracy(chosen_after, rejected_after))

    shape = (3, 5)
    old_log_probs = torch.zeros(shape)
    new_log_probs = torch.full(shape, 0.03, requires_grad=True)
    values = torch.zeros(shape, requires_grad=True)
    old_values = torch.zeros(shape)
    rewards = torch.zeros(shape)
    rewards[:, -1] = torch.tensor([1.0, 0.5, -0.25])
    mask = torch.ones(shape, dtype=torch.bool)
    advantages, returns = generalized_advantage_estimate(rewards, old_values, mask)
    objective = ppo_loss(new_log_probs, old_log_probs, advantages, values, old_values, returns, mask)
    objective.total.backward()

    result = {
        "verification_scope": "fast synthetic CPU smoke run; not the full public-data benchmark",
        "device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "tokenizer": {
            "kind": "byte_bpe",
            "vocab_size": tokenizer.vocab_size,
            "round_trip_exact": tokenizer.decode(tokenizer.encode("Hello, MATH 550!"), show_special=False) == "Hello, MATH 550!",
            "training_tokens": len(train_ids),
            "validation_tokens": len(validation_ids),
        },
        "gpt": {
            "parameter_count": model.parameter_count,
            "block_size": model_config.block_size,
            "layers": model_config.n_layer,
            "heads": model_config.n_head,
            "embedding_width": model_config.n_embd,
            "initial_validation_loss": history[0]["validation_loss"],
            "final_validation_loss": history[-1]["validation_loss"],
            "final_validation_perplexity": history[-1]["validation_perplexity"],
        },
        "bigram_baseline": bigram,
        "sft": {
            "assistant_only_initial_loss": sft_initial,
            "assistant_only_final_loss": sft_final,
            "last_training_epoch_loss": sft_epoch_losses[-1],
            "examples": len(sft_data),
        },
        "reward_model": {
            "initial_preference_loss": reward_loss_initial,
            "final_preference_loss": reward_loss_final,
            "training_preference_accuracy": reward_accuracy,
            "pairs": len(preference_data),
        },
        "ppo_objective": {
            "finite": bool(torch.isfinite(objective.total)),
            "total": float(objective.total.detach()),
            "policy": float(objective.policy.detach()),
            "value": float(objective.value.detach()),
            "approximate_kl": float(objective.approximate_kl.detach()),
            "policy_gradient_finite": bool(torch.isfinite(new_log_probs.grad).all()),
            "value_gradient_finite": bool(torch.isfinite(values.grad).all()),
        },
    }
    output_path = analysis_dir / "smoke_results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return output_path


if __name__ == "__main__":
    print(run())
