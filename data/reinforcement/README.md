# Reinforcement-learning data

Reinforcement learning usually creates its training transitions by interacting
with an environment. The default course run therefore does **not** rely on a
hidden pretrained model or a fixed supervised dataset.

## Included compact replay dataset

Generate 50,000 uniformly random MinAtar Breakout transitions with:

```bash
uv run --extra rl python -m experiments.reinforcement.collect_replay
```

This writes `minatar_breakout_3actions_random_seed550_50000steps.npz` plus a JSON provenance
record. The observations are 10 x 10 x 4 binary arrays. The exact environment
is MinAtar 1.0.15 from <https://github.com/kenjyoung/MinAtar>, licensed GPL-3.0;
the associated paper is <https://arxiv.org/abs/1903.03176>. The generator uses
seed 550 and uniform random actions over `0=left`, `1=right`, `2=no-op`.
The provenance records the mapping to raw MinAtar IDs `1, 3, 0`. The compact binary
is included for classroom convenience; the transition arrays can be regenerated
from the pinned environment, seed, and collection command above.

The original `minatar_breakout_random_seed550_50k.npz` and its provenance are
preserved as a historical six-action dataset. Do not feed its actions directly
to a three-output network: old IDs `1, 3` mean left/right and old IDs `0, 2, 4, 5`
all hold the paddle still. Even after mapping those IDs, its uniform six-action
collection policy is not uniform over the three distinct actions. New DQN/PPO
training collects fresh online transitions and does not load that old archive.
The Asterix homework archive retains the original six-action mapping.

## Optional full Atari data

For an offline-RL or imitation-learning extension, download Farama Minari's
`atari/breakout/expert-v0` exactly as follows:

```bash
uv sync --locked --extra rl-data
uv run --extra rl-data minari download atari/breakout/expert-v0
```

Authoritative dataset page:
<https://minari.farama.org/main/datasets/atari/breakout/expert-v0/>

As documented there, version `expert-v0` contains 23,075 transitions in 10
episodes, raw RGB observations of shape 210 x 160 x 3, four discrete actions,
and trajectories collected by a CleanRL PPO-Impala expert policy. The raw
observations are intentionally not copied into this repository: they are much
larger than the compact MinAtar replay and remain governed by the Minari/ALE
source terms. Minari itself is MIT-licensed.
