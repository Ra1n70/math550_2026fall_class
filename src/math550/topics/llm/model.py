"""GPT-style decoder implemented from elementary PyTorch modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    tie_embeddings: bool = True

    def validate(self) -> None:
        positive = (
            self.vocab_size,
            self.block_size,
            self.n_layer,
            self.n_head,
            self.n_embd,
        )
        if min(positive) < 1:
            raise ValueError("GPT dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention with explicit score computation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, width = inputs.shape
        qkv = self.qkv(inputs)
        query, key, value = qkv.split(width, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, width)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = nn.GELU(approximate="tanh")
        self.contract = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.contract(self.activation(self.expand(inputs))))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))


class GPT(nn.Module):
    """Decoder-only Transformer with learned token and position embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.language_model_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("projection.weight") or name.endswith("contract.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> GPTOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError("sequence length exceeds the configured block size")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.language_model_head(hidden)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must match input_ids shape")
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(batch_size, sequence_length)
            valid = (targets != -100).to(token_loss.dtype)
            if loss_mask is not None:
                if loss_mask.shape != targets.shape:
                    raise ValueError("loss_mask must match targets shape")
                valid = valid * loss_mask.to(token_loss.dtype)
            denominator = valid.sum().clamp_min(1.0)
            loss = (token_loss * valid).sum() / denominator
        return GPTOutput(logits=logits, loss=loss, hidden_states=hidden if return_hidden else None)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        valid_vocab_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        sampling_vocab_size = (
            self.config.vocab_size
            if valid_vocab_size is None
            else int(valid_vocab_size)
        )
        if not 1 <= sampling_vocab_size <= self.config.vocab_size:
            raise ValueError(
                "valid_vocab_size must lie between 1 and the model vocabulary size"
            )
        if eos_id is not None and not 0 <= eos_id < sampling_vocab_size:
            raise ValueError("eos_id must belong to the valid sampling vocabulary")
        generated = input_ids
        for _ in range(max_new_tokens):
            context = generated[:, -self.config.block_size :]
            # A model head may be padded for hardware efficiency.  Padded rows
            # are parameters, but they are not tokens and must never be sampled.
            logits = (
                self(context).logits[:, -1, :sampling_vocab_size] / temperature
            )
            if top_k is not None:
                k = min(max(int(top_k), 1), logits.size(-1))
                threshold = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_id is not None and torch.all(next_token.squeeze(1) == eos_id):
                break
        return generated


class BigramLanguageModel(nn.Module):
    """Capacity baseline: the next-token distribution sees only the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> GPTOutput:
        logits = self.table(input_ids)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return GPTOutput(logits=logits, loss=loss)
