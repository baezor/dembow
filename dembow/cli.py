"""Command line interface for Dembow.

    dembow train     # train the Transformer on the reggaeton corpus
    dembow generate  # sample new reggaeton out of a trained model
"""

from __future__ import annotations

import argparse

from . import __version__

# Hardware presets. CPU training of a Transformer is slow, so the CPU preset is
# deliberately small and leans on early stopping; the GPU preset is more
# ambitious. Any value passed explicitly on the command line overrides these.
PRESETS = {
    "cpu": dict(d_model=128, n_layers=2, n_heads=4, seq_len=192, num_epochs=40, batch_size=24, augment=1),
    "gpu": dict(d_model=256, n_layers=4, n_heads=4, seq_len=384, num_epochs=120, batch_size=32, augment=3),
}


def _resolve_preset(name: str) -> dict:
    if name == "auto":
        try:
            import torch

            name = "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            name = "cpu"
    return PRESETS[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dembow",
        description="The first A.I. that generates reggaeton hits.",
    )
    parser.add_argument("--version", action="version", version=f"dembow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the Transformer on a folder of MIDI files")
    t.add_argument("--preset", choices=["auto", "cpu", "gpu"], default="auto",
                   help="hardware preset for model size / epochs (default auto); explicit flags override it")
    t.add_argument("--data-dir", default="reggaeton_samples")
    t.add_argument("--checkpoint", default="dembow.pt")
    # These default to None so the preset can fill them in unless set explicitly.
    t.add_argument("--seq-len", type=int, default=None)
    t.add_argument("--d-model", type=int, default=None)
    t.add_argument("--n-layers", type=int, default=None)
    t.add_argument("--n-heads", type=int, default=None)
    t.add_argument("--num-epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--augment", type=int, default=None, help="pitch-shift augmentation range in semitones (0 disables)")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val-frac", type=float, default=0.1, help="fraction of songs held out for validation")
    t.add_argument("--patience", type=int, default=8, help="early-stop after N epochs without val improvement")
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
    g.add_argument("--repetition-penalty", type=float, default=1.15, help="discourage looping (1.0 = off)")
    g.add_argument("--no-repeat-ngram", type=int, default=0, help="hard-ban repeated token n-grams of this size (0 = off)")
    g.add_argument("--prime-bars", type=int, default=2, help="real bars used to prime each song ('--seed-dir none' for cold start)")
    g.add_argument("--seed-dir", default="reggaeton_samples")
    g.add_argument("--tempo-bpm", type=float, default=95.0)
    g.add_argument("--render", action="store_true", help="also render each song to .wav audio")
    g.add_argument("--soundfont", default=None, help="SoundFont (.sf2) for FluidSynth rendering; omit to use the builtin synth")
    g.add_argument("--random-seed", type=int, default=0)
    g.add_argument("--device", default=None)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from .train import train

        preset = _resolve_preset(args.preset)
        pick = lambda name: getattr(args, name) if getattr(args, name) is not None else preset[name]
        train(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            seq_len=pick("seq_len"),
            d_model=pick("d_model"),
            n_layers=pick("n_layers"),
            n_heads=pick("n_heads"),
            num_epochs=pick("num_epochs"),
            batch_size=pick("batch_size"),
            augment=pick("augment"),
            lr=args.lr,
            val_frac=args.val_frac,
            patience=args.patience,
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
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
            prime_bars=args.prime_bars,
            seed_dir=seed_dir,
            tempo_bpm=args.tempo_bpm,
            render=args.render,
            soundfont=args.soundfont,
            random_seed=args.random_seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
