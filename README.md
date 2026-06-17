# Dembow
## The first A.I. that generates reggaeton hits. 🔥

![Denbow.jpg](denbow.jpg)

Dembow learns from a corpus of reggaeton MIDI and writes new tracks of its own.
It began life in 2016 as a Restricted Boltzmann Machine over a binary piano roll;
it is now a **decoder-only Transformer over an event-based music language** — the
same recipe behind modern symbolic-music models.

## How it works

Dembow treats music the way a language model treats text. Every song is
tokenized into a stream of musical **events** (REMI-style):

```
BOS  BAR  POS_0  INST_drums DRUM_kick  DUR_1 VEL_5
              POS_0  INST_bass  PITCH_36   DUR_4 VEL_6
              POS_4  INST_drums DRUM_snare DUR_1 VEL_5  ...
     BAR  ...  EOS
```

Each note carries its **instrument group** (drums / bass / mid / high), **pitch**,
**duration**, and **velocity** — so the model can write expressive,
multi-instrument arrangements, not a flat on/off grid. A small Transformer then
learns to predict the next event from everything before it, using masked
self-attention to capture phrasing, repetition, and the way the drums and bass
lock into the dembow groove.

Generation is autoregressive with **temperature + nucleus (top-p) sampling**, the
standard modern decoding strategy.

## What changed from the original

| Then (2016) | Now |
| --- | --- |
| Python 2, TensorFlow 1.x | Python 3, **PyTorch** |
| Restricted Boltzmann Machine | **Decoder-only Transformer** |
| Binary piano roll (on/off only) | **Event tokens**: pitch + duration + velocity |
| All tracks flattened into one roll | **Multi-instrument** (drums / bass / mid / high) |
| No sense of time | **Self-attention** over the whole sequence |
| Trained on ~76 raw files | **Pitch-augmented** corpus (×7) for generalization |
| `python-midi` (Py2, dead) | [`mido`](https://mido.readthedocs.io) |
| Threw the weights away | Saves & loads checkpoints |
| One-shot script | A real CLI + installable package + CI |

## Getting started

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # numpy, mido, torch
# or install the package + `dembow` command:
pip install -e .
```

## Make magic happen

```sh
dembow train                 # -> dembow.pt
dembow generate              # -> generated/dembow_*.mid
```

Without installing:

```sh
python -m dembow.cli train
python -m dembow.cli generate
```

Or light the fire (train + generate in one go):

```sh
python fire.py
```

## Tuning

Training:

```sh
dembow train \
  --num-epochs 120 \       # train longer for tighter structure
  --d-model 320 --n-layers 6 \   # a bigger model
  --augment 4              # ±4 semitones of pitch augmentation (0 disables)
```

Generation:

```sh
dembow generate \
  --num-samples 8 \
  --max-new-tokens 1200 \  # longer songs
  --temperature 0.9 \      # <1 tighter & more repetitive, >1 wilder
  --top-p 0.92 \           # nucleus sampling threshold
  --prime-bars 2 \         # real bars used to kick off each song
  --seed-dir none          # cold start instead of priming from a real song
```

**Honest note on quality.** The corpus is only ~76 short MIDI files, so even a
Transformer is data-limited — it captures the *feel* (groove, instrumentation,
key) more than polished, hook-worthy songwriting. The single biggest lever is
**more clean MIDI** in `reggaeton_samples/`. Pitch augmentation and priming from
real songs help it stay in the pocket meanwhile.

## Project layout

```
dembow/
  tokenizer.py   MIDI <-> event tokens (the REMI-style music language)
  model.py       the decoder-only Transformer
  data.py        load the corpus, augment, build training windows
  train.py       training loop + checkpointing
  generate.py    sample new songs (temperature / top-p) and write MIDI
  cli.py         the `dembow` command
fire.py          one-shot entry point
reggaeton_samples/   the MIDI corpus
tests/           a fast end-to-end smoke test
```

## Contribute

We still need your help feeding the model. If you have reggaeton MIDI, drop it in
`reggaeton_samples/` and open a pull request — more data is the single best way
to make Dembow sound like a hit.
