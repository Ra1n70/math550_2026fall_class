"""Hardware-aware, deliberately modest reasoning-lab presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from math550.topics.reasoning import GRPOConfig, OpenModelSpec, SFTConfig, model_spec


@dataclass(frozen=True)
class ReasoningExperimentConfig:
    preset: str = "mps"
    model_profile: str = "qwen-0.5b"
    data_root: Path = Path("data/reasoning")
    output_root: Path = Path("output/reasoning")
    checkpoint_root: Path = Path("checkpoints/reasoning")
    seed: int = 550

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "analysis"

    @property
    def figure_dir(self) -> Path:
        return self.output_root / "figures"

    def model(self) -> OpenModelSpec:
        ranks = {"smoke": 4, "quick": 8, "mps": 16, "colab": 16}
        if self.preset not in ranks:
            raise ValueError(f"unknown preset {self.preset!r}")
        rank = ranks[self.preset]
        return model_spec(self.model_profile, lora_rank=rank, lora_alpha=2 * rank)

    def max_length(self) -> int:
        lengths = {"smoke": 160, "quick": 256, "mps": 384, "colab": 512}
        if self.preset not in lengths:
            raise ValueError(f"unknown preset {self.preset!r}")
        return lengths[self.preset]

    def training_limit(self) -> int | None:
        return {"smoke": 24, "quick": 512, "mps": None, "colab": None}[self.preset]

    def sft(self) -> SFTConfig:
        presets = {
            "smoke": SFTConfig(
                epochs=1,
                batch_size=1,
                gradient_accumulation_steps=2,
                max_steps=2,
                log_interval=1,
                seed=self.seed,
            ),
            "quick": SFTConfig(
                epochs=2,
                batch_size=1,
                gradient_accumulation_steps=4,
                max_steps=50,
                log_interval=10,
                seed=self.seed,
            ),
            "mps": SFTConfig(
                epochs=3,
                batch_size=1,
                gradient_accumulation_steps=8,
                max_steps=800,
                log_interval=20,
                seed=self.seed,
            ),
            "colab": SFTConfig(
                epochs=3,
                batch_size=4,
                gradient_accumulation_steps=8,
                max_steps=1_200,
                log_interval=25,
                seed=self.seed,
            ),
        }
        if self.preset not in presets:
            raise ValueError(f"unknown preset {self.preset!r}")
        return presets[self.preset]

    def grpo(self, variant: str = "grpo", *, prompt_style: str = "cot") -> GRPOConfig:
        presets = {
            "smoke": GRPOConfig(group_size=2, max_new_tokens=32, max_length=192, seed=self.seed),
            "quick": GRPOConfig(group_size=4, max_new_tokens=96, max_length=384, seed=self.seed),
            "mps": GRPOConfig(group_size=4, max_new_tokens=128, max_length=512, seed=self.seed),
            "colab": GRPOConfig(group_size=8, max_new_tokens=192, max_length=640, seed=self.seed),
        }
        if self.preset not in presets:
            raise ValueError(f"unknown preset {self.preset!r}")
        base = presets[self.preset]
        variants = {
            "grpo": {},
            "grpo-constant": {"loss_normalization": "constant"},
            "dr-grpo": {
                "advantage_normalizer": "none",
                "loss_normalization": "constant",
            },
            "rft": {
                "baseline": "none",
                "advantage_normalizer": "none",
                "loss_normalization": "constant",
                "importance_method": "none",
            },
            "maxrl": {
                "advantage_normalizer": "mean",
                "loss_normalization": "constant",
            },
            "gspo": {"importance_method": "gspo"},
        }
        if variant not in variants:
            raise ValueError(f"unknown RLVR variant {variant!r}")
        return replace(base, prompt_style=prompt_style, **variants[variant])
