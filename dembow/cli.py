"""Command line interface for Dembow.

    dembow train     # train the RBM on the reggaeton corpus
    dembow generate  # sample new reggaeton out of a trained model
"""

from __future__ import annotations

import argparse

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dembow",
        description="The first A.I. that generates reggaeton hits.",
    )
    parser.add_argument("--version", action="version", version=f"dembow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the RBM on a folder of MIDI files")
    t.add_argument("--data-dir", default="reggaeton_samples")
    t.add_argument("--checkpoint", default="dembow.pt")
    t.add_argument("--num-timesteps", type=int, default=15)
    t.add_argument("--n-hidden", type=int, default=50)
    t.add_argument("--num-epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=100)
    t.add_argument("--lr", type=float, default=0.005)
    t.add_argument("--k", type=int, default=1, help="Gibbs steps for contrastive divergence")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    g = sub.add_parser("generate", help="sample new reggaeton from a trained model")
    g.add_argument("--checkpoint", default="dembow.pt")
    g.add_argument("--output-dir", default="generated")
    g.add_argument("--num-samples", type=int, default=10)
    g.add_argument("--k", type=int, default=1)
    g.add_argument("--seed-dir", default="reggaeton_samples", help="seed chains from real songs ('none' to start silent)")
    g.add_argument("--tempo-bpm", type=float, default=95.0)
    g.add_argument("--random-seed", type=int, default=0)
    g.add_argument("--device", default=None)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from .train import train

        train(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            num_timesteps=args.num_timesteps,
            n_hidden=args.n_hidden,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            k=args.k,
            seed=args.seed,
            device=args.device,
        )
    elif args.command == "generate":
        from .generate import generate

        seed_dir = None if str(args.seed_dir).lower() == "none" else args.seed_dir
        generate(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            k=args.k,
            seed_dir=seed_dir,
            tempo_bpm=args.tempo_bpm,
            random_seed=args.random_seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
