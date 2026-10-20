"""Generate a compact, source-traceable MinAtar replay dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from math550.topics.reinforcement import (
    MINATAR_BREAKOUT,
    MinAtarAdapter,
    collect_random_transitions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="breakout")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output archive; default name records the game, action count, seed, and steps",
    )
    args = parser.parse_args()

    environment = MinAtarAdapter(game=args.game, seed=args.seed)
    if args.output is None:
        args.output = Path("data/reinforcement") / (
            f"minatar_{args.game}_{environment.action_count}actions_random_"
            f"seed{args.seed}_{args.steps}steps.npz"
        )
    data = collect_random_transitions(environment, args.steps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    metadata = {
        "environment": f"MinAtar {args.game.title()}",
        "environment_version": MINATAR_BREAKOUT.version,
        "source_url": MINATAR_BREAKOUT.source_url,
        "paper_url": MINATAR_BREAKOUT.paper_url,
        "license": MINATAR_BREAKOUT.license,
        "game": args.game,
        "steps": args.steps,
        "seed": args.seed,
        "policy": f"uniform random over {environment.action_count} actions",
        "sticky_action_probability": 0.1,
        "difficulty_ramping": True,
        "observation_shape": list(environment.observation_shape),
        "action_count": environment.action_count,
        "action_names": list(environment.action_names),
        "raw_minatar_action_ids": list(environment.raw_action_ids),
        "arrays": {name: list(values.shape) for name, values in data.items()},
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
