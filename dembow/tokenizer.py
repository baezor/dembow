"""Event-based (REMI-style) tokenization of MIDI.

The old project represented music as a binary piano roll -- a grid of on/off
bits with no duration, velocity, or instrument identity. This replaces it with a
sequence of musical *events*, the representation modern symbolic-music models use
(Music Transformer, REMI, MMM). A song becomes a flat sequence of tokens::

    BOS  BAR  POS_0  INST_drums PITCH_? DUR_1 VEL_5  INST_bass PITCH_36 DUR_4 VEL_6 ...
         BAR  POS_4  ...  EOS

Each note carries its instrument group, pitch, duration, and velocity, so the
Transformer can learn expressive, multi-instrument arrangements -- not just a
flat grid. Timing is quantized to a 16th-note grid (16 positions per 4/4 bar).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import mido

# -- musical vocabulary ----------------------------------------------------

STEPS_PER_BAR = 16          # 16th-note grid, 4/4
MEL_LOW, MEL_HIGH = 24, 96  # melodic pitch range (C1..C7)
N_MEL = MEL_HIGH - MEL_LOW + 1
N_VEL_BINS = 8
DUR_BUCKETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
INST_GROUPS = ["drums", "bass", "mid", "high"]
DRUM_CHANNEL = 9

# General MIDI percussion -> a small set of drum classes (the dembow toolkit).
DRUM_CLASSES = ["kick", "snare", "clap", "closed_hat", "open_hat", "tom", "crash", "ride", "perc"]
_DRUM_MAP = {
    35: 0, 36: 0, 38: 1, 40: 1, 37: 2, 39: 2, 42: 3, 44: 3, 46: 4,
    41: 5, 43: 5, 45: 5, 47: 5, 48: 5, 50: 5, 49: 6, 57: 6, 55: 6, 51: 7, 59: 7, 53: 7,
}
_DRUM_REPR = [36, 38, 39, 42, 46, 45, 49, 51, 54]  # play each class back as this GM note
# Instrument group -> GM program for playback (drums use channel 9).
_INST_PROGRAM = {"bass": 38, "mid": 0, "high": 81}


def _drum_class(note: int) -> int:
    return _DRUM_MAP.get(note, 8)


def _vel_bin(vel: int) -> int:
    return min(N_VEL_BINS - 1, max(0, vel // (128 // N_VEL_BINS)))


def _vel_value(bin_idx: int) -> int:
    return min(127, bin_idx * (128 // N_VEL_BINS) + 24)


def _dur_bucket(steps: int) -> int:
    best = 0
    for i, b in enumerate(DUR_BUCKETS):
        if abs(b - steps) < abs(DUR_BUCKETS[best] - steps):
            best = i
    return best


def _pitch_group(pitch: int) -> str:
    if pitch < 48:
        return "bass"
    if pitch < 72:
        return "mid"
    return "high"


class Vocab:
    """Builds and holds the token <-> id mapping."""

    def __init__(self):
        tokens: List[str] = ["PAD", "BOS", "EOS", "BAR"]
        tokens += [f"POS_{p}" for p in range(STEPS_PER_BAR)]
        tokens += [f"INST_{g}" for g in INST_GROUPS]
        tokens += [f"PITCH_{n}" for n in range(MEL_LOW, MEL_HIGH + 1)]
        tokens += [f"DRUM_{c}" for c in range(len(DRUM_CLASSES))]
        tokens += [f"DUR_{i}" for i in range(len(DUR_BUCKETS))]
        tokens += [f"VEL_{v}" for v in range(N_VEL_BINS)]
        self.itos: List[str] = tokens
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi[token]


VOCAB = Vocab()
PAD, BOS, EOS, BAR = VOCAB["PAD"], VOCAB["BOS"], VOCAB["EOS"], VOCAB["BAR"]


# -- parsing ---------------------------------------------------------------

def _read_notes(path: str) -> List[Tuple[int, str, int, int, int]]:
    """Return notes as ``(start_step, group, sound_id, dur_steps, vel)``."""
    midi_file = mido.MidiFile(path)
    ticks_per_step = midi_file.ticks_per_beat / 4.0
    notes = []
    open_notes: Dict[Tuple[int, int], Tuple[int, int]] = {}

    abs_tick = 0
    for msg in mido.merge_tracks(midi_file.tracks):
        abs_tick += msg.time
        step = int(round(abs_tick / ticks_per_step))
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.channel == DRUM_CHANNEL:
                # Percussion: instantaneous hit, no sustain.
                notes.append((step, "drums", _drum_class(msg.note), 1, msg.velocity))
            else:
                open_notes[(msg.channel, msg.note)] = (step, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in open_notes and msg.channel != DRUM_CHANNEL:
                start, vel = open_notes.pop(key)
                dur = max(1, step - start)
                notes.append((start, _pitch_group(msg.note), msg.note, dur, vel))
    return notes


def encode(path: str, transpose: int = 0) -> List[int]:
    """Encode a MIDI file into a list of token ids (optionally transposed)."""
    notes = _read_notes(path)
    # Apply transposition to pitched notes only.
    shifted = []
    for start, group, sound, dur, vel in notes:
        if group != "drums":
            sound += transpose
            if sound < MEL_LOW or sound > MEL_HIGH:
                continue
        shifted.append((start, group, sound, dur, vel))
    shifted.sort(key=lambda n: (n[0], INST_GROUPS.index(n[1]), n[2]))

    ids = [BOS]
    cur_bar, cur_pos = -1, -1
    for start, group, sound, dur, vel in shifted:
        bar, pos = divmod(start, STEPS_PER_BAR)
        while cur_bar < bar:
            ids.append(BAR)
            cur_bar += 1
            cur_pos = -1
        if pos != cur_pos:
            ids.append(VOCAB[f"POS_{pos}"])
            cur_pos = pos
        ids.append(VOCAB[f"INST_{group}"])
        if group == "drums":
            ids.append(VOCAB[f"DRUM_{sound}"])
        else:
            ids.append(VOCAB[f"PITCH_{sound}"])
        ids.append(VOCAB[f"DUR_{_dur_bucket(dur)}"])
        ids.append(VOCAB[f"VEL_{_vel_bin(vel)}"])
    ids.append(EOS)
    return ids


# -- decoding --------------------------------------------------------------

def decode(ids: List[int], tempo_bpm: float = 95.0, ticks_per_beat: int = 480) -> mido.MidiFile:
    """Turn a token-id sequence back into a multi-track MIDI file."""
    ticks_per_step = ticks_per_beat // 4
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    midi_file.tracks.append(meta)

    # Collect note events per instrument group as (tick, kind, note, vel).
    events: Dict[str, List[Tuple[int, str, int, int]]] = {g: [] for g in INST_GROUPS}

    bar, pos = -1, 0
    group = sound = dur = vel = None
    for tid in ids:
        tok = VOCAB.itos[tid]
        if tok in ("BOS", "PAD"):
            continue
        if tok == "EOS":
            break
        if tok == "BAR":
            bar += 1
            pos = 0
        elif tok.startswith("POS_"):
            pos = int(tok[4:])
        elif tok.startswith("INST_"):
            group, sound, dur, vel = tok[5:], None, None, None
        elif tok.startswith("PITCH_"):
            sound = int(tok[6:])
        elif tok.startswith("DRUM_"):
            sound = int(tok[5:])
        elif tok.startswith("DUR_"):
            dur = DUR_BUCKETS[int(tok[4:])]
        elif tok.startswith("VEL_"):
            vel = _vel_value(int(tok[4:]))
            if group is not None and sound is not None:
                start_step = max(0, bar) * STEPS_PER_BAR + pos
                tick = start_step * ticks_per_step
                if group == "drums":
                    note = _DRUM_REPR[sound % len(_DRUM_REPR)]
                    events["drums"].append((tick, "note_on", note, vel))
                    events["drums"].append((tick + ticks_per_step // 2, "note_off", note, 0))
                else:
                    length = (dur or 2) * ticks_per_step
                    events[group].append((tick, "note_on", sound, vel))
                    events[group].append((tick + length, "note_off", sound, 0))
            group = sound = dur = vel = None

    for g in INST_GROUPS:
        evs = events[g]
        if not evs:
            continue
        track = mido.MidiTrack()
        channel = DRUM_CHANNEL if g == "drums" else INST_GROUPS.index(g)
        if g != "drums":
            track.append(mido.Message("program_change", program=_INST_PROGRAM[g], channel=channel, time=0))
        evs.sort(key=lambda e: e[0])
        last = 0
        for tick, kind, note, v in evs:
            track.append(mido.Message(kind, note=note, velocity=v, channel=channel, time=tick - last))
            last = tick
        midi_file.tracks.append(track)

    return midi_file
