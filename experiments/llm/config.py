"""Hardware-aware model and training presets for the LLM labs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from math550.topics.llm import GPTConfig, TrainingConfig


@dataclass(frozen=True)
class ExperimentConfig:
    preset: str = "mps"
    data_root: Path = Path("data/llm")
    output_root: Path = Path("output")
    checkpoint_root: Path = Path("checkpoints/llm")
    seed: int = 550

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "llm" / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "llm" / "figures"

    @property
    def pdf_dir(self) -> Path:
        return self.output_root / "pdf"

    def model(self, vocab_size: int) -> GPTConfig:
        presets = {
            "smoke": dict(block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0),
            "cpu": dict(block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
            "mps": dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1),
            "colab": dict(block_size=256, n_layer=8, n_head=8, n_embd=512, dropout=0.1),
        }
        if self.preset not in presets:
            raise ValueError(f"unknown preset {self.preset!r}")
        return GPTConfig(vocab_size=vocab_size, **presets[self.preset])

    def training(self, device: str = "auto") -> TrainingConfig:
        presets = {
            "smoke": TrainingConfig(
                max_steps=30,
                batch_size=8,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=3,
                weight_decay=0.01,
                eval_interval=10,
                eval_batches=3,
                checkpoint_interval=100,
                device=device,
                seed=self.seed,
            ),
            "cpu": TrainingConfig(
                max_steps=1_000,
                batch_size=8,
                gradient_accumulation_steps=2,
                eval_interval=100,
                eval_batches=10,
                checkpoint_interval=500,
                device=device,
                seed=self.seed,
            ),
            "mps": TrainingConfig(
                max_steps=6_000,
                batch_size=16,
                eval_interval=250,
                eval_batches=20,
                checkpoint_interval=1_000,
                device=device,
                seed=self.seed,
                dtype="float32",
            ),
            "colab": TrainingConfig(
                max_steps=10_000,
                batch_size=16,
                gradient_accumulation_steps=2,
                eval_interval=250,
                eval_batches=20,
                checkpoint_interval=1_000,
                device=device,
                seed=self.seed,
                dtype="float16",
            ),
        }
        if self.preset not in presets:
            raise ValueError(f"unknown preset {self.preset!r}")
        return presets[self.preset]


def with_step_override(config: TrainingConfig, steps: int | None) -> TrainingConfig:
    return config if steps is None else replace(config, max_steps=steps)
