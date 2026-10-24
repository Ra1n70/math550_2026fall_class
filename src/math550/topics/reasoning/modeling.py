"""Load a compact open LLM and attach trainable LoRA reasoning adapters."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SUPPORTED_DEVICES = ("auto", "cuda", "mps", "cpu")


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(SUPPORTED_DEVICES)}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


@dataclass(frozen=True)
class OpenModelSpec:
    profile: str = "qwen-0.5b"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    license: str = "Apache-2.0"
    approximate_weight_gb: float = 0.99
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


MODEL_PROFILES: dict[str, dict[str, str | float]] = {
    "qwen-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "license": "Apache-2.0",
        "approximate_weight_gb": 0.99,
    },
    "qwen-math-1.5b": {
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "revision": "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35",
        "license": "Apache-2.0",
        "approximate_weight_gb": 3.09,
    },
    "olmo2-1b-instruct": {
        "model_id": "allenai/OLMo-2-0425-1B-Instruct",
        "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
        "license": "Apache-2.0",
        "approximate_weight_gb": 2.98,
    },
}


def model_spec(
    profile: str = "qwen-0.5b",
    *,
    lora_rank: int = 16,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
) -> OpenModelSpec:
    """Build an immutable, revision-pinned model specification by course profile."""

    if profile not in MODEL_PROFILES:
        raise ValueError(f"unknown model profile {profile!r}; choose from {sorted(MODEL_PROFILES)}")
    if lora_rank < 1 or (lora_alpha is not None and lora_alpha < 1):
        raise ValueError("LoRA rank and alpha must be positive")
    record = MODEL_PROFILES[profile]
    return OpenModelSpec(
        profile=profile,
        model_id=str(record["model_id"]),
        revision=str(record["revision"]),
        license=str(record["license"]),
        approximate_weight_gb=float(record["approximate_weight_gb"]),
        lora_rank=lora_rank,
        lora_alpha=2 * lora_rank if lora_alpha is None else lora_alpha,
        lora_dropout=lora_dropout,
    )


def _ensure_padding_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs either a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def load_open_model(
    spec: OpenModelSpec,
    *,
    device: str = "auto",
    trainable_lora: bool = True,
) -> tuple[Any, Any, torch.device]:
    """Load revision-pinned open weights and optionally add LoRA parameters."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised by installation docs
        raise RuntimeError("install the reasoning extra: uv sync --extra reasoning") from error

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, revision=spec.revision, trust_remote_code=False
    )
    _ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        dtype=preferred_dtype(resolved),
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = not trainable_lora
    if trainable_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("PEFT is required for the consumer-scale training track") from error
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            bias="none",
            target_modules=list(spec.lora_target_modules),
        )
        model = get_peft_model(model, lora)
    model.to(resolved)
    return model, tokenizer, resolved


def load_reasoning_adapter(
    adapter_path: Path,
    spec: OpenModelSpec,
    *,
    device: str = "auto",
    trainable: bool = False,
) -> tuple[Any, Any, torch.device]:
    """Load the pinned base model and a saved course LoRA adapter for inference."""

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the reasoning extra: uv sync --extra reasoning") from error
    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    _ensure_padding_token(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        dtype=preferred_dtype(resolved),
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
    model.config.use_cache = not trainable
    model.to(resolved)
    model.train(mode=trainable)
    return model, tokenizer, resolved


def adapter_disabled(model: Any):
    """Return PEFT's base-model context, or a no-op for a plain causal LM."""

    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def parameter_summary(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }
