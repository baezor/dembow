"""Command line interface for Dembow.

    dembow train     # train the Transformer on the reggaeton corpus
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

    t = sub.add_parser("train", help="train the Transformer on a folder of MIDI files")
    t.add_argument("--data-dir", default="reggaeton_samples")
    t.add_argument("--checkpoint", default="dembow.pt")
    t.add_argument("--seq-len", type=int, default=384)
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--n-layers", type=int, default=4)
    t.add_argument("--n-heads", type=int, default=4)
    t.add_argument("--num-epochs", type=int, default=80)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--augment", type=int, default=3, help="pitch-shift augmentation range in semitones (0 disables)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)

    g = sub.add_parser("generate", help="sample new reggaeton from a trained model")
    g.add_argument("--checkpoint", default="dembow.pt")
    g.add_argument("--output-dir", default="generated")
    g.add_argument("--num-samples", type=int, default=5)
    g.add_argument("--max-new-tokens", type=int, default=800)
    g.add_argument("--temperature", type=float, default=1.0, help="<1 tighter, >1 wilder")
    g.add_argument("--top-p", type=float, default=0.92, help="nucleus sampling threshold")
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--prime-bars", type=int, default=2, help="real bars used to prime each song ('--seed-dir none' for cold start)")
    g.add_argument("--seed-dir", default="reggaeton_samples")
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
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            augment=args.augment,
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
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            prime_bars=args.prime_bars,
            seed_dir=seed_dir,
            tempo_bpm=args.tempo_bpm,
            random_seed=args.random_seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
