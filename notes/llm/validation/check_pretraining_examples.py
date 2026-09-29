"""Numerically check the Python code extracted verbatim from the lecture notes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import torch
from torch.nn import functional as F

from math550.topics.llm import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path,
    default=ROOT / "notes/llm/validation/results.json",
    help="save the machine-readable numerical check record (default: alongside this script)",
)
args = parser.parse_args()
source = ROOT / "notes/llm/pretraining.tex"
source_bytes = source.read_bytes()
blocks = re.findall(
    r"\\begin\{lstlisting\}\[language=Python\]\s*\n(.*?)\\end\{lstlisting\}",
    source_bytes.decode("utf-8"),
    flags=re.S,
)
namespace = {}
exec(compile("\n\n".join(blocks), str(source), "exec"), namespace)
loss_batch, evaluate, train_basic, sample_next = (
    namespace[name] for name in ("loss_batch", "evaluate", "train_basic", "sample_next")
)
assert len(blocks) == 4, f"Unexpected number of listings: {len(blocks)}"
torch.set_num_threads(1)
torch.manual_seed(2026)
device = torch.device("cpu")
config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2,
                   n_embd=16, dropout=0.1)
model = GPT(config).eval()
windows = torch.tensor([
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [20, 21, 22, 23, 24, 25, 26],
], dtype=torch.long)
x, y = windows[:, :-1], windows[:, 1:]
checks = []

# The exact listing accepts raw model logits and returns mean token NLL.
logits = model(x).logits
ce = loss_batch(model, x, y)
manual_nll = -logits.log_softmax(-1).gather(-1, y[..., None]).mean()
torch.testing.assert_close(ce, manual_nll)
checks.append("loss_batch equals manually gathered log-softmax NLL")

# Include an unequal final validation batch: 12 targets then 6 targets.
loader = [(x[:2], y[:2]), (x[2:], y[2:])]
full_loss = float(manual_nll.detach())
weighted_loss = evaluate(model, loader, device)
assert abs(weighted_loss - full_loss) < 1e-6
batch_mean = sum(float(loss_batch(model, bx, by).detach())
                 for bx, by in loader) / 2
assert abs(batch_mean - full_loss) > 1e-5, "Test must distinguish weighting"
first_batch = evaluate(model, loader, device, max_batches=1)
assert abs(first_batch - float(loss_batch(model, *loader[0]).detach())) < 1e-6
for empty_args in (([], None), (loader, 0)):
    try:
        evaluate(model, empty_args[0], device, max_batches=empty_args[1])
    except ValueError as error:
        assert "No validation targets" in str(error)
    else:
        raise AssertionError("Empty evaluation should fail")
checks.append("evaluate weights 12+6 targets correctly and respects max_batches")

# Hooks inspect actual network calls under the listing decorators.
observed = []
hook = model.register_forward_hook(
    lambda mod, args, output: observed.append(
        (torch.is_grad_enabled(), output.logits.requires_grad, mod.training)
    )
)
for was_training in (True, False):
    model.train(was_training)
    observed.clear()
    evaluate(model, loader, device)
    assert model.training is was_training
    assert observed and all(row == (False, False, False) for row in observed)
    assert all(parameter.grad is None for parameter in model.parameters())
    observed.clear()
    sampled = sample_next(model, x[:1], 6, 29, top_k=4)
    assert sampled.shape == (1, 1) and not sampled.requires_grad
    assert model.training is was_training
    assert observed == [(False, False, False)]
    assert all(parameter.grad is None for parameter in model.parameters())
hook.remove()
checks.append("evaluate/sample_next disable dropout and autograd and restore both modes")

# Exceptions inside the try/finally must also restore the prior mode.
model.train()
try:
    evaluate(model, [(x, y[:, :-1])], device)
except ValueError:
    assert model.training
else:
    raise AssertionError("Mismatched target count should fail")
try:
    sample_next(model, torch.full((1, 2), 100, dtype=torch.long), 2, 29, top_k=4)
except IndexError:
    assert model.training
else:
    raise AssertionError("Out-of-range input IDs should fail")
checks.append("exception paths also restore the prior model mode")

# Check a cropped context, exact top-k membership, bounds, and unique-max greedy.
model.eval()
prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
with torch.no_grad():
    permitted = set(model(prompt[:, -6:]).logits[0, -1, :29].topk(4).indices.tolist())
    maximum = int(model(prompt[:, -6:]).logits[0, -1, :29].argmax())
draws = [int(sample_next(model, prompt, 6, 29, temperature=0.8, top_k=4))
         for _ in range(64)]
assert set(draws) <= permitted and 0 <= min(draws) <= max(draws) < 29
assert int(sample_next(model, prompt, 6, 29, top_k=1)) == maximum
for kwargs in ({"temperature": 0.0}, {"top_k": 0}, {"top_k": 30}):
    try:
        sample_next(model, prompt, 6, 29, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid setting accepted: {kwargs}")
try:
    sample_next(model, prompt, 6, 33, top_k=4)
except ValueError:
    pass
else:
    raise AssertionError("Tokenizer larger than model accepted")
checks.append("sample_next crops context, restricts valid IDs/top-k, validates settings")

# Causality is a model invariant supporting the notes' training argument.
changed = x.clone()
changed[:, 4:] = torch.tensor([[13, 14], [15, 16], [17, 18]])
with torch.no_grad():
    original_logits = model(x).logits
    changed_logits = model(changed).logits
torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4],
                           atol=1e-6, rtol=1e-6)
checks.append("changing future tokens leaves earlier logits unchanged")

# Exercise the ACTUAL train_basic defaults on a repeated shifted batch.
torch.manual_seed(550)
trained = GPT(config)
repeat_loader = [(x[:2], y[:2])] * 6
validation_loader = [(x[:2], y[:2])]
before = evaluate(trained, validation_loader, device)
history, optimizer = train_basic(trained, repeat_loader, validation_loader,
                                 device, epochs=20, eval_every=5)
after = evaluate(trained, validation_loader, device)
assert after < before - 0.5, (before, after)
assert trained.training
assert all(math.isfinite(value) for row in history for value in row)
assert history[0][0:2] == (1, 12)
assert history[-1][0:2] == (120, 1440)
assert optimizer.param_groups[0]["lr"] == 4e-4
assert optimizer.param_groups[0]["weight_decay"] == 0.1
checks.append("train_basic reduces repeated-batch loss with finite records and exact token counts")

results = {
    "source": str(source.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    },
    "seeds": {"checks_and_sampling": 2026, "repeated_batch_training": 550},
    "extracted_python_listings": len(blocks),
    "device": str(device),
    "model_config": config.to_dict(),
    "parameter_count": model.parameter_count,
    "manual_nll": full_loss,
    "unequal_batch_token_weighted_nll": weighted_loss,
    "incorrect_unweighted_batch_nll_for_comparison": batch_mean,
    "sampling_draws": 64,
    "sampling_distinct_ids": sorted(set(draws)),
    "sampling_allowed_ids": sorted(permitted),
    "repeated_batch_updates": 120,
    "repeated_batch_tokens_seen": 1440,
    "repeated_batch_before_nll": before,
    "repeated_batch_after_nll": after,
    "checks": checks,
    "scope": "Numerical tests of exact lecture listings on synthetic CPU tensors; no public-data or language-quality claim.",
}
serialized = json.dumps(results, indent=2)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(serialized + "\n", encoding="utf-8")
print(serialized)
