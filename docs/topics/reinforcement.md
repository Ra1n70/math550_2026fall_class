# Reinforcement learning

This topic builds the algorithmic path from a one-step Bellman update to modern
value-based, policy-gradient, and search-based systems without a high-level RL
training API:

1. deterministic GridWorld and tabular Q-learning;
2. Double-DQN with explicit replay, targets, and target synchronization on
   MinAtar Breakout;
3. REINFORCE, actor-critic baselines, generalized advantage estimation (GAE),
   and PPO on MinAtar Breakout;
4. PPO as limited off-policy reuse: per-action/token likelihood ratios, signed
   clipping regions, exact trajectory ratios, and the uncorrected old-policy
   state distribution;
5. language models as token policies, including a full score-function and
   baseline-variance derivation;
6. GRPO decomposed into group baseline, advantage scaling, sequence/constant
   normalization, stale-rollout correction, and clipping, with Dr. GRPO, RFT,
   MaxRL, CISPO, and GSPO as explicit configurations;
7. AlphaZero-style policy/value learning, PUCT search, and self-play on Connect
   Four;
8. a conceptual and mathematical bridge to MuZero for learned-model planning.

PyTorch is used for neural layers, autograd, losses, and optimizers. Course code
owns the reinforcement-learning mechanisms.

## Public data and environments

The default online benchmark is MinAtar 1.0.15 Breakout from
<https://github.com/kenjyoung/MinAtar> (GPL-3.0), described in
<https://arxiv.org/abs/1903.03176>. Observations are 10 x 10 x 4 binary tensors.
The repository includes a reproducibly generated 50,000-transition replay file
and provenance JSON in `data/reinforcement/`.
Breakout uses three action IDs throughout new training and playback:
`0 = left`, `1 = right`, `2 = no-op`, mapped to raw MinAtar IDs `1, 3, 0`.
The new replay is `minatar_breakout_3actions_random_seed550_50000steps.npz`.
The original six-action archive is retained separately as historical data;
its action IDs must not be interpreted as the new three-action IDs.
Other games, including the Asterix homework, retain their six-action mapping.
The checked-in lecture notes, slides, report, and their measured results describe
the earlier six-action experiments. They are historical teaching artifacts, not
measurements of the revised three-action policies; this code change does not
rerun those benchmarks.

The optional full Atari dataset is Farama Minari
`atari/breakout/expert-v0`: 23,075 transitions, 10 episodes, raw
210 x 160 x 3 RGB observations, and four actions. Authoritative page and exact
download command:

```bash
uv sync --locked --extra rl-data
uv run --extra rl-data minari download atari/breakout/expert-v0
```

<https://minari.farama.org/main/datasets/atari/breakout/expert-v0/>

Homework transfers to MinAtar Asterix and includes a different 20,000-transition
random replay generated with seed 551.

## Public API

- `GridWorld`, `q_learning`, `greedy_rollout`
- `MinAtarAdapter`, `ReplayBuffer`
- `MinAtarQNetwork`, `DQNConfig`, `train_dqn`, `evaluate_dqn`
- `MinAtarActorCritic`, `PPOConfig`, `train_ppo`, `evaluate_ppo`
- `MinAtarPolicyRollout`, `rollout_minatar_policy`
- `MinAtarPolicyPlayer` (live step, reset, and current-state snapshot)
- `render_minatar_policy_frame`, `save_minatar_policy_animation`
- `discounted_returns`, `generalized_advantage_estimate`, `ppo_clipped_loss`,
  `ppo_token_surrogate`
- `group_relative_advantages`, `grpo_clipped_loss`
- shared `compute_group_advantages`, `policy_gradient_loss`,
  `aggregate_token_losses`, and `reference_kl_estimate` in
  `math550.core.alignment`
- `ConnectFourState`, `ConnectFourPolicyValueNetwork`
- `run_mcts`, `self_play_game`, `train_policy_value_network`

## Commands

```bash
uv sync --locked --extra rl
make test-reinforcement
make quick-reinforcement
make play-reinforcement
make demo-reinforcement
make audit-ppo
make audit-rlvr
make benchmark-reinforcement
make notes-reinforcement
make homework-reinforcement
make validate-homework-reinforcement
```

Direct track commands:

```bash
uv run --extra rl python -m experiments.reinforcement.run_benchmark --quick --device auto
uv run --extra rl python -m experiments.reinforcement.run_benchmark --device auto --track dqn
uv run --extra rl python -m experiments.reinforcement.run_benchmark --device auto --track ppo
uv run --extra rl python -m experiments.reinforcement.run_benchmark --device auto --track selfplay
```

## Classroom policy-gradient notebook

Open [`policy_gradient_ppo_demo.ipynb`](../../experiments/reinforcement/policy_gradient_ppo_demo.ipynb)
and select the course `.venv/bin/python` kernel, then restart and **Run All**.
The notebook includes its own kernel-registration command if needed.

The cells show the score-gradient calculation, a compact actor–critic, rollout
collection, terminal-masked GAE, and PPO minibatch updates directly. The notebook
uses the current three-action Breakout contract and shares only environment,
device, and animation helpers with the package. Its GAE and policy/value losses
are checked against the existing reusable implementation during validation.

The default run collects 32,768 interactions (`NUM_UPDATES = 32`), compares
uniform random and before/after greedy policies on 20 fixed evaluation seeds,
and embeds training curves and an annotated GIF. Change `NUM_UPDATES` to adjust
the classroom runtime. CUDA, MPS, and CPU are supported. Evaluation seeds are
disjoint from training, capped episodes are reported, and error bars show
episode-return standard errors. A short run is behavior evidence from one
training seed, not a general performance claim.

Notebook outputs go to `output/reinforcement/notebook_demo/`: training and
evaluation CSVs, settings/results JSON, two figure PNGs, and the rollout GIF.
The existing benchmark outputs remain separate.

## Learned-policy playback

### Live browser player

To watch the game on a single board with playback controls, run from the repository root:

```bash
make play-reinforcement
```

This opens `http://127.0.0.1:8765/` in your browser, or an automatically selected
free port if 8765 is already occupied. The terminal prints the actual URL and
the browser opens that address. Existing servers keep running. You can also
choose a preferred port with `make play-reinforcement RL_UI_PORT=8766`, or use
`RL_UI_PORT=0` to let the OS choose one immediately. Press **Play** to watch the
learned policy move the paddle. Use **Pause**, **Step**, the speed slider,
**Replay** (same seed), or **New episode** (next seed). Choose DQN or PPO if both
checkpoints are available. The page starts paused and keeps the final board at
game over. Enable automatic next episodes to keep watching.

The server loads the saved PyTorch checkpoint and computes decisions against the
real MinAtar environment on demand. It writes no frames or plots. The board and
action values always refer to the current state; selected and executed actions
are shown separately when MinAtar repeats a previous action. Breakout exposes
only **left**, **right**, and **no-op**. No-op holds the paddle still while the
ball and game continue. New DQN and PPO training creates three-output policies;
uniform random exploration now gives each of these three actions equal weight.

Existing six-output checkpoints are preserved and adapted for playback. Left
and right retain their outputs; the four old hold-paddle controls (no-op, up,
down, fire) become one no-op choice. DQN uses their maximum Q-value, while PPO
sums their probabilities (log-sum-exp of logits before softmax). PPO's greedy
choice can therefore change after grouping; this does not retrain the policy
or establish improved performance. A warning and a note in the UI identify
legacy checkpoints. Retrain with the commands above for a native three-action
policy. Restart a running server with Ctrl+C and `make play-reinforcement` to
load the revised code.

The existing export command also accepts `--ui`:

```bash
uv run --extra rl python -m experiments.reinforcement.play_policy --ui --algorithm ppo

# Use your own trained weights, or choose another local port.
uv run --extra rl python -m experiments.reinforcement.play_ui \
  --algorithm dqn --checkpoint output/reinforcement/checkpoints/minatar_breakout_double_dqn.pt \
  --device auto --port 8766

# Print the URL without launching the system browser.
uv run --extra rl python -m experiments.reinforcement.play_ui --no-browser
```

If a checkpoint is missing, first run `make quick-reinforcement` for the short
training demonstration or the full training track above. A custom `--checkpoint`
exposes that policy only; choose the matching `--algorithm` and `--game`. The UI
uses Python's standard-library HTTP server and browser canvas, with no additional
dependencies. It binds to this computer's loopback address and has one shared
episode per server, so use one controlling browser tab. Stop it with Ctrl+C.
On a remote notebook, forward the selected port to your browser; the default
URL is intended for a local Python process.

### Export an animation and decision trace

To save an annotated episode for a report, the original export commands remain:

```bash
# Double-DQN checkpoint; also available as `make demo-reinforcement`
uv run --extra rl python -m experiments.reinforcement.play_policy \
  --algorithm dqn --device auto --seed 10550

# PPO checkpoint
uv run --extra rl python -m experiments.reinforcement.play_policy \
  --algorithm ppo --device auto --seed 30550
```

The driver loads the PyTorch weights with `weights_only=True`, runs a greedy
episode, and writes four linked artifacts below `output/reinforcement/demos/`:

- an animated GIF of the learned policy;
- a PNG contact sheet for reports or static viewing;
- a CSV decision trace with actions, rewards, cumulative return, and every
  action's Q-value or PPO probability;
- provenance JSON containing the checkpoint SHA-256, environment version,
  seed, device, episode return, and output paths.

Each frame labels the selected action, reward, cumulative return, object-channel
colors, and the complete action-preference vector. For PPO it also shows the
critic value. The fixed-seed episode explains behavior but does not replace the
multi-episode evaluation table.

`auto` prefers CUDA, then MPS, then CPU. Explicit unavailable accelerator
requests fail clearly.

## Measured quick-run interpretation

On the local Apple M2 MPS backend, the verified combined quick run trained DQN
for 5,000 steps in 35.40 seconds and evaluated at 3.33 mean return over six
episodes, versus 0.67 for uniform random. Its 8,192-step PPO track took 3.55
seconds and evaluated at $0.83\pm0.17$ versus $0.50\pm0.34$ for random over six episodes;
its final approximate KL was 0.0011 and clip fraction was 0.0. This checks the
training path and diagnostics but is too small for a learning claim. The GRPO
lab is a deterministic grouped-objective calculation, not an LLM training run.
The tiny self-play arena had only four games and scored 1.0
both before and after one iteration; it validates execution but cannot establish
self-improvement. Full results should use several seeds and larger evaluation
sets.

Generated numerical truth lives under `output/reinforcement/`; the report,
lecture notes, slides, and homework reuse those results and state their limits.

## PPO's on/off-policy status

PPO first collects a rollout from a frozen behavior policy `pi_old`. The first
minibatch update is on-policy, but later minibatch epochs reuse the same actions
after the live policy has changed. For each sampled action—or each generated LLM
token—the ratio

```text
r_t = exp(log pi_theta(a_t | s_t) - log pi_old(a_t | s_t))
```

multiplies the score gradient by `r_t * advantage` while the signed clipped
surrogate is active. Positive advantages saturate above `1 + epsilon`; negative
advantages saturate below `1 - epsilon`. The other two outside-window cases
remain live.

This is limited, near-on-policy reuse rather than arbitrary replay. The sampled
states or LLM prefixes still follow `pi_old`, and token clipping is not the exact
sequence correction, whose ratio is the product of all token ratios. Run
`make audit-ppo` to reproduce the token table, JSON/CSV audit, and explanatory
figure under `output/reinforcement/`.
