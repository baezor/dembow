# Dembow
## The first A.I. that generates reggaeton hits. 🔥

![Denbow.jpg](denbow.jpg)

Dembow learns the *dembow* groove from a corpus of reggaeton MIDI files and
dreams up new patterns of its own. It started life in 2016 as a Restricted
Boltzmann Machine; it has since grown a sequence model that actually understands
the beat — but it kept its soul.

## Two engines, same soul

| Engine | What it is | Use it for |
| --- | --- | --- |
| **LSTM** (default) | A recurrent net that reads the song one 16th-note at a time and predicts what comes next. It learns *sequence* — the dembow kick/snare, where the bass lands, how a phrase moves. | Output that actually grooves. |
| **RBM** (classic) | The original 2016 model: a Restricted Boltzmann Machine sampling a static "bag of notes" with contrastive divergence + Gibbs sampling. | Nostalgia, and seeing where it all started. |

### Why the early version sounded like noise

The original threw every one of a song's ~6–7 tracks into a single piano roll,
which blended the **dembow drums** (channel 9 — almost half of all the notes in
the corpus, and the signature of the genre) in with bass and melody, scrambling
the groove. It also modelled a window of time as an unordered set, so it had no
way to learn "boom — boom-chick," and it sampled every pitch independently, so
each output was a wall of 300+ simultaneous notes.

The revival fixes the **representation** for both engines:

- **Drums are separated** from pitched content and bucketed into musical classes
  (kick, snare, clap, hats, …), so the model can learn the beat as a beat.
- **Songs are transposed to a common key**, so it learns relative harmony
  instead of smearing every key together.
- **Output is kept sparse** — a handful of notes per step — so it's music.

## What changed in the revival

| Then (2016) | Now |
| --- | --- |
| Python 2 (`print "..."`) | Python 3 |
| TensorFlow 1.x graph + `Session` | **PyTorch** (CPU/GPU) |
| `python-midi` (unmaintained, Py2) | [`mido`](https://mido.readthedocs.io) |
| All tracks flattened into one roll | **Drums separated**, key-normalized |
| RBM only (no sense of time) | **LSTM** sequence model (+ RBM classic) |
| 300+ notes of noise per window | Sparse, groove-aware output |
| Threw the trained weights away | Saves & loads checkpoints |
| One-shot script | A real CLI + installable package |
| `glob('*.mid*')` skipped `.MID` | Case-insensitive corpus loading |

## Getting started

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt      # numpy, mido, torch
# or install the package + `dembow` command:
pip install -e .
```

## Make magic happen

Train the default LSTM on the included reggaeton corpus, then generate:

```sh
dembow train                 # -> dembow.pt   (LSTM, ~1 min on CPU)
dembow generate              # -> generated/dembow_*.mid
```

Want the original model? Just ask for it:

```sh
dembow train --model rbm
dembow generate              # generation auto-detects the model in the checkpoint
```

Without installing, the same things work through the module:

```sh
python -m dembow.cli train
python -m dembow.cli generate
```

Or just light the fire (train + generate in one go):

```sh
python fire.py
```

## Making it sound more like reggaeton

Generation **primes each track with the opening bars of a real reggaeton song**,
so it starts in the pocket — including the dembow drums — then improvises from
there. A few knobs to push it further:

```sh
# train longer / bigger for a tighter groove
dembow train --num-epochs 150 --hidden 384 --seq-len 96

# generation controls
dembow generate \
  --num-samples 8 \
  --num-steps 128 \        # length after the priming bars
  --max-pitched 4 \        # fewer simultaneous notes = cleaner melody
  --temperature 0.8 \      # <1 = tighter & more repetitive, >1 = wilder
  --tempo-bpm 95           # reggaeton pocket
```

**Honest note on quality.** The corpus is only ~76 short MIDI files, so the
model is data-limited — it captures the *feel* (the beat, the density, the key)
more than polished, hook-worthy songwriting. The two biggest levers are **more
clean MIDI** in `reggaeton_samples/` and longer training. If you want the
dembow beat to be rock-solid every time, the natural next step is a
*groove-template* mode that locks in the canonical dembow drum pattern and lets
the model improvise bass and melody over it.

## Project layout

```
dembow/
  representation.py  drum+pitched, key-normalized features (the genre-aware input)
  lstm.py            the sequence model (default engine)
  rbm.py             the Restricted Boltzmann Machine (classic engine)
  midi_io.py         basic MIDI <-> piano roll, via mido
  dataset.py         load the corpus, build training windows/sequences
  train.py           training loops + checkpointing
  generate.py        sample new patterns and write MIDI
  cli.py             the `dembow` command
fire.py              nostalgic one-shot entry point
reggaeton_samples/   the MIDI corpus
tests/               a fast end-to-end smoke test
```

## Contribute

We still need your help feeding and training the model. If you have reggaeton
MIDI, drop it in `reggaeton_samples/` and open a pull request — more data is the
single best way to make Dembow sound like a hit.
