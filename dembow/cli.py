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

    t = sub.add_parser("train", help="train a model on a folder of MIDI files")
    t.add_argument("--model", choices=["lstm", "rbm"], default="lstm", help="lstm (default) or classic rbm")
    t.add_argument("--data-dir", default="reggaeton_samples")
    t.add_argument("--checkpoint", default="dembow.pt")
    t.add_argument("--num-epochs", type=int, default=None, help="defaults: 60 (lstm) / 200 (rbm)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    # LSTM knobs
    t.add_argument("--seq-len", type=int, default=64)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--lr", type=float, default=None, help="defaults: 1e-3 (lstm) / 0.005 (rbm)")
    t.add_argument("--no-key-norm", action="store_true", help="skip transposing songs to a common key")
    # RBM knobs (classic mode)
    t.add_argument("--num-timesteps", type=int, default=15)
    t.add_argument("--n-hidden", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=None, help="defaults: 64 (lstm) / 100 (rbm)")
    t.add_argument("--k", type=int, default=1, help="Gibbs steps for contrastive divergence (rbm)")

    g = sub.add_parser("generate", help="sample new reggaeton from a trained model")
    g.add_argument("--checkpoint", default="dembow.pt")
    g.add_argument("--output-dir", default="generated")
    g.add_argument("--num-samples", type=int, default=5)
    g.add_argument("--num-steps", type=int, default=64, help="steps to generate after the prime (lstm)")
    g.add_argument("--prime-steps", type=int, default=16, help="seed steps taken from a real song (lstm)")
    g.add_argument("--max-pitched", type=int, default=5, help="max simultaneous pitched notes (lstm)")
    g.add_argument("--temperature", type=float, default=1.0, help="<1 tighter, >1 wilder (lstm)")
    g.add_argument("--k", type=int, default=1, help="Gibbs steps (rbm)")
    g.add_argument("--seed-dir", default="reggaeton_samples", help="seed from real songs ('none' to start silent)")
    g.add_argument("--tempo-bpm", type=float, default=95.0)
    g.add_argument("--random-seed", type=int, default=0)
    g.add_argument("--device", default=None)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        if args.model == "lstm":
            from .train import train_lstm

            train_lstm(
                data_dir=args.data_dir,
                checkpoint=args.checkpoint,
                seq_len=args.seq_len,
                hidden=args.hidden,
                layers=args.layers,
                num_epochs=args.num_epochs if args.num_epochs is not None else 60,
                batch_size=args.batch_size if args.batch_size is not None else 64,
                lr=args.lr if args.lr is not None else 1e-3,
                normalize_key=not args.no_key_norm,
                seed=args.seed,
                device=args.device,
            )
        else:
            from .train import train

            train(
                data_dir=args.data_dir,
                checkpoint=args.checkpoint,
                num_timesteps=args.num_timesteps,
                n_hidden=args.n_hidden,
                num_epochs=args.num_epochs if args.num_epochs is not None else 200,
                batch_size=args.batch_size if args.batch_size is not None else 100,
                lr=args.lr if args.lr is not None else 0.005,
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
            num_steps=args.num_steps,
            prime_steps=args.prime_steps,
            max_pitched=args.max_pitched,
            temperature=args.temperature,
            k=args.k,
            seed_dir=seed_dir,
            tempo_bpm=args.tempo_bpm,
            random_seed=args.random_seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
