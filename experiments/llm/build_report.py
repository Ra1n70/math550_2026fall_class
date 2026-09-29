"""Build measured figures and the concise technical report for the LLM module."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt

from math550.topics.llm import GPT

from .config import ExperimentConfig


def _plot_training_evidence(synthetic: dict[str, object], public_history: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(
        [row["step"] for row in public_history],
        [row["validation_loss"] for row in public_history],
        marker="o",
        color="#1F4E78",
        label="held-out books",
    )
    axes[0].plot(
        [row["step"] for row in public_history],
        [row["train_loss"] for row in public_history],
        marker="s",
        color="#E07A5F",
        label="training batch",
    )
    axes[0].set(title="Public-book pretraining smoke run", xlabel="optimizer update", ylabel="cross-entropy")
    axes[0].legend(frameon=False)
    labels = ["GPT pretrain", "SFT", "reward"]
    initial = [
        synthetic["gpt"]["initial_validation_loss"],
        synthetic["sft"]["assistant_only_initial_loss"],
        synthetic["reward_model"]["initial_preference_loss"],
    ]
    final = [
        synthetic["gpt"]["final_validation_loss"],
        synthetic["sft"]["assistant_only_final_loss"],
        synthetic["reward_model"]["final_preference_loss"],
    ]
    positions = range(len(labels))
    axes[1].bar([value - 0.18 for value in positions], initial, 0.36, label="initial", color="#9FBAD0")
    axes[1].bar([value + 0.18 for value in positions], final, 0.36, label="final", color="#1F4E78")
    axes[1].set_xticks(list(positions), labels)
    axes[1].set(title="Synthetic stage-level correctness checks", ylabel="stage-specific loss")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_corpus(metadata: dict[str, object], path: Path) -> None:
    books = metadata["books"]
    labels = [f"PG{book['id']}" for book in books]
    tokens = [book["tokens"] / 1_000_000 for book in books]
    colors = ["#1F4E78" if book["split"] == "train" else "#E07A5F" for book in books]
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.bar(labels, tokens, color=colors)
    axis.set(title="Token count by whole-book split", ylabel="million BPE tokens", xlabel="Project Gutenberg eBook id")
    axis.tick_params(axis="x", rotation=45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.text(0.99, 0.94, "blue = train; coral = validation", transform=axis.transAxes, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_presets(vocab_size: int, path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for preset in ("smoke", "cpu", "mps", "colab"):
        model = GPT(ExperimentConfig(preset=preset).model(vocab_size))
        counts[preset] = model.parameter_count
        del model
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    labels = list(counts)
    values = [counts[label] / 1_000_000 for label in labels]
    bars = axis.bar(labels, values, color=["#9FBAD0", "#6B8EAD", "#1F4E78", "#E07A5F"])
    axis.bar_label(bars, labels=[f"{value:.1f}M" for value in values], padding=3)
    axis.set(title="Course model presets", ylabel="trainable parameters (millions)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return counts


def build(output_root: Path = Path("output")) -> Path:
    analysis_dir = output_root / "llm" / "analysis"
    figure_dir = output_root / "llm" / "figures"
    pdf_dir = output_root / "pdf"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    synthetic = json.loads((analysis_dir / "smoke_results.json").read_text(encoding="utf-8"))
    public_history_path = Path("checkpoints/llm/pretrain-smoke/training_history.json")
    public_history = json.loads(public_history_path.read_text(encoding="utf-8"))
    shutil.copyfile(public_history_path, analysis_dir / "public_book_smoke_history.json")
    corpus = json.loads(Path("data/llm/processed/metadata.json").read_text(encoding="utf-8"))
    _plot_training_evidence(synthetic, public_history, figure_dir / "training_evidence.png")
    _plot_corpus(corpus, figure_dir / "corpus_tokens.png")
    preset_counts = _plot_presets(corpus["vocab_size"], figure_dir / "preset_parameters.png")
    (analysis_dir / "preset_parameter_counts.json").write_text(
        json.dumps(preset_counts, indent=2), encoding="utf-8"
    )
    train_tokens = corpus["streams"]["train"]["tokens"]
    validation_tokens = corpus["streams"]["validation"]["tokens"]
    first_public = public_history[0]
    last_public = public_history[-1]
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,microtype,tabularx,xcolor}}
\usepackage[colorlinks=true,urlcolor=blue!55!black,linkcolor=blue!55!black]{{hyperref}}
\graphicspath{{{{../llm/figures/}}}}
\definecolor{{navy}}{{HTML}}{{1F4E78}}
\definecolor{{coral}}{{HTML}}{{E07A5F}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\newcommand{{\E}}{{\mathbb{{E}}}}
\begin{{document}}
\begin{{center}}
{{\LARGE\bfseries\color{{navy}} Large Language Models from Scratch}}\\[3pt]
{{\large Pretraining, instruction following, reward modeling, and PPO-style RLHF}}\\[7pt]
MATH 550 technical report -- measured build evidence
\end{{center}}

\colorbox{{blue!8}}{{\parbox{{0.96\linewidth}}{{
\textbf{{Outcome.}} The module implements a decoder-only GPT, byte-level BPE,
next-token pretraining, response-masked SFT, a pairwise reward model, GAE, and
clipped PPO using elementary PyTorch modules and optimizers. No pretrained model,
\texttt{{torch.nn.Transformer}}, Hugging Face model, or tokenizer API is used.
The downloaded corpus contains {train_tokens:,} training tokens and
{validation_tokens:,} whole-book validation tokens. A 30-update public-data
smoke run reduced held-out cross-entropy from {first_public['validation_loss']:.3f}
to {last_public['validation_loss']:.3f}; this verifies the pipeline, not useful language quality.
}}}}

\section*{{1. Learning contract and sequence}}
The nine-unit module moves from representation to optimization to behavioral
post-training. Students (1) train a byte-pair tokenizer, (2) derive the causal
language-model likelihood, (3) implement attention and a GPT block, (4) pretrain
and compare with a bigram baseline, (5) diagnose sampling and held-out loss,
(6) perform assistant-only SFT, (7) learn a Bradley--Terry reward model, (8) derive
and run PPO-style RLHF, and (9) audit data rights, reward hacking, memorization,
and evidence limits. The durable derivations are in the lecture notes; labs call
the importable \texttt{{math550.topics.llm}} package.

\section*{{2. Model and objectives}}
For tokens $x_1,\ldots,x_T$, pretraining minimizes
\[
\mathcal L_{{\rm LM}}(\theta)=-\frac1T\sum_{{t=1}}^T
\log p_\theta(x_t\mid x_{{<t}}).
\]
Each attention head computes
\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
A=\operatorname{{softmax}}\left(\frac{{QK^\top}}{{\sqrt{{d_h}}}}+M\right)V,
\]
where $M_{{ij}}=-\infty$ for $j>i$. The code explicitly creates $Q,K,V$,
reshapes heads, applies the triangular mask, normalizes the scores, concatenates
heads, and projects the result. Residual pre-normalization and a four-times-wider
GELU feed-forward network complete each block.

SFT uses the same next-token loss but masks every prompt and padding label. For a
chosen response $y^+$ and rejected response $y^-$, reward training minimizes
\[
\mathcal L_{{\rm RM}}=-\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).
\]
PPO uses sampled token rewards $r_t=-\beta(\log\pi_\theta-\log\pi_{{\rm ref}})$
plus the learned terminal reward, GAE advantages, and the clipped surrogate
$\min(\rho_t\hat A_t,\operatorname{{clip}}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)$.

\section*{{3. Public data and split discipline}}
Fourteen Project Gutenberg novels train the 512-entry BPE and GPT. \emph{{War
and Peace}} (PG2600) and \emph{{Ulysses}} (PG4300) are held out as whole books,
preventing adjacent-window leakage. Raw downloads occupy about 42 MB; uint16
token streams occupy about 21 MB. Every resolved URL and observed hash is stored
in \texttt{{data/llm/download\_manifest.json}}. Project Gutenberg's terms warn
non-U.S. users to check local copyright law.

Databricks Dolly 15k supplies 15,011 SFT records under CC BY-SA 3.0. Anthropic
HH-RLHF \texttt{{harmless-base}} supplies chosen/rejected conversations under the
MIT license. Both downloads are revision-pinned and SHA-256 checked. HH-RLHF
contains harmful prompts and unsafe rejected responses; raw examples are not
appropriate for projection in class.

\begin{{figure}}[H]\centering
\includegraphics[width=0.94\linewidth]{{corpus_tokens.png}}
\caption{{Whole-book data partition. The tokenizer is also fit on blue books only.}}
\end{{figure}}

\section*{{4. Training presets and expected use}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrrrl}}
\toprule
Preset & Layers & Heads & Width & Parameters & Intended use \\
\midrule
smoke & 2 & 2 & 64 & {preset_counts['smoke']:,} & unit and pipeline checks \\
cpu & 4 & 4 & 128 & {preset_counts['cpu']:,} & short laptop lab \\
mps & 6 & 6 & 384 & {preset_counts['mps']:,} & M2/M3/M4 or consumer GPU \\
colab & 8 & 8 & 512 & {preset_counts['colab']:,} & Colab CUDA extension \\
\bottomrule
\end{{tabular}}
\end{{table}}
The default MPS run uses a 256-token context, batch 16, 6,000 optimizer updates,
AdamW, gradient clipping, warmup, and cosine decay. This processes about 24.6
million token positions. Runtime is hardware- and PyTorch-version-dependent;
students first time 100 updates, then extrapolate. The phrase ``a few hours'' is
a target for accelerator presets, not a guaranteed benchmark.

\begin{{figure}}[H]\centering
\includegraphics[width=0.73\linewidth]{{preset_parameters.png}}
\caption{{Exact trainable parameter counts with the course's 512-entry vocabulary.}}
\end{{figure}}

\section*{{5. Verified results}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{lrrr}}
\toprule
Check & Initial & Final & Interpretation \\
\midrule
Public books, held-out LM loss & {first_public['validation_loss']:.3f} & {last_public['validation_loss']:.3f} & real-data path optimizes \\
Synthetic GPT held-out loss & {synthetic['gpt']['initial_validation_loss']:.3f} & {synthetic['gpt']['final_validation_loss']:.3f} & causal training loop works \\
Synthetic SFT assistant loss & {synthetic['sft']['assistant_only_initial_loss']:.3f} & {synthetic['sft']['assistant_only_final_loss']:.3f} & response mask works \\
Synthetic reward preference loss & {synthetic['reward_model']['initial_preference_loss']:.3f} & {synthetic['reward_model']['final_preference_loss']:.3f} & pair ranking is learnable \\
\bottomrule
\end{{tabular}}
\end{{table}}
The public smoke checkpoint generates mostly character-level noise after only 30
updates. That negative result is retained: decreasing loss demonstrates neither
instruction following nor a useful language model. The synthetic reward accuracy
reaches 1.0 on four training pairs and therefore demonstrates overfitting, not
general reward validity. PPO gradients and KL calculations are finite in the
smoke test, but no behavioral improvement claim is made.

\begin{{figure}}[H]\centering
\includegraphics[width=0.96\linewidth]{{training_evidence.png}}
\caption{{Measured optimization evidence. Losses on the right have different
semantics and should not be compared across stages.}}
\end{{figure}}

\section*{{6. Verification and limitations}}
Automated tests check Unicode round trips, deterministic BPE, serialization,
causal invariance under future-token changes, tied weights, shapes, autograd,
prompt/padding masks, token shifts, reward placement, pairwise gradients, GAE,
and PPO clipping. The integrated CPU run spans every objective.

The course model is intentionally small and trained on old English-language
fiction. It inherits narrow genre, historical stereotypes, and limited factual
coverage. Dolly contains annotator and factual errors. HH preferences encode a
particular safety policy and include disturbing content. Perplexity depends on
the tokenizer; reward scores are not truth; PPO can reward-hack. Responsible use
requires held-out prompts, human review, memorization checks, subgroup analysis,
data-rights review, and explicit refusal to call a classroom checkpoint safe or
general-purpose.

\section*{{References and exact sources}}
\small
\begin{{itemize}}
\item Sennrich, Haddow, and Birch (2016), ``Neural Machine Translation of Rare Words with Subword Units,'' \url{{https://aclanthology.org/P16-1162/}}.
\item Vaswani et al. (2017), ``Attention Is All You Need,'' \url{{https://arxiv.org/abs/1706.03762}}.
\item Radford et al. (2019), \href{{https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf}}{{``Language Models are Unsupervised Multitask Learners'' (OpenAI PDF)}}.
\item Ouyang et al. (2022), ``Training language models to follow instructions with human feedback,'' \url{{https://arxiv.org/abs/2203.02155}}.
\item Schulman et al. (2017), ``Proximal Policy Optimization Algorithms,'' \url{{https://arxiv.org/abs/1707.06347}}.
\item Bai et al. (2022), ``Training a Helpful and Harmless Assistant with RLHF,'' \url{{https://arxiv.org/abs/2204.05862}}.
\item Raschka, \emph{{Build a Large Language Model (From Scratch)}} code, \url{{https://github.com/rasbt/LLMs-from-scratch}}.
\item Data records and precise downloads: \texttt{{data/llm/sources.json}}.
\end{{itemize}}
\end{{document}}
"""
    tex_path = pdf_dir / "llm_from_scratch_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    latex_environment = os.environ.copy()
    latex_environment.update({"LC_ALL": "C", "LANG": "C"})
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        env=latex_environment,
    )
    return tex_path.with_suffix(".pdf")


if __name__ == "__main__":
    print(build())
