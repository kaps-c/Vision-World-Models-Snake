"""``snek`` command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from snekwm.config import CollectConfig, load_config


def _collect(args: argparse.Namespace) -> None:
    from snekwm.data.collect import collect

    cfg = load_config(args.config, "collect", CollectConfig)
    paths = collect(
        cfg.out_dir,
        size=cfg.size,
        episodes=cfg.episodes,
        max_steps=cfg.max_steps,
        epsilon=cfg.epsilon,
        seed=cfg.seed,
    )
    print(f"wrote {len(paths)} episodes to {cfg.out_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run the chosen subcommand."""
    parser = argparse.ArgumentParser(prog="snek")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="roll out the data policy")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_collect)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
