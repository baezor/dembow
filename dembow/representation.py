"""A genre-aware representation: drums separated from pitched content.

The original piano roll flattened all ~6-7 tracks of a reggaeton MIDI into one
blob, which mixed the dembow kick/snare (the signature of the genre, ~45% of all
notes, living on MIDI channel 9) in with bass and melody. This module pulls the
two apart so a model can actually learn the groove:

A song becomes a ``[num_steps, N_FEATURES]`` matrix whose columns are::

    [0 : N_DRUMS]                       drum onsets, bucketed into a few classes
    [N_DRUMS : N_DRUMS + SPAN]          pitched "play"       (a note is sounding)
    [N_DRUMS + SPAN : N_DRUMS + 2*SPAN] pitched "articulate" (a note was struck)

Pitched content is also transposed to a common key (C) so the model learns
relative harmony instead of smearing every key together. Drums are left alone.
"""

from __future__ import annotations

from typing import List

import mido
import numpy as np

from .midi_io import LOWER_BOUND, UPPER_BOUND, SPAN

DRUM_CHANNEL = 9  # General MIDI percussion channel (0-indexed)

# Bucket the dozens of GM percussion notes into a handful of musically distinct
# classes. This keeps the drum vocabulary small and learnable.
DRUM_CLASSES = [
    "kick",
    "snare",
    "clap",
    "closed_hat",
    "open_hat",
    "tom",
    "crash",
    "ride",
    "perc",
]
N_DRUMS = len(DRUM_CLASSES)

_DRUM_MAP = {
    35: 0, 36: 0,                      # kick
    38: 1, 40: 1,                      # snare
    37: 2, 39: 2,                      # rim / clap
    42: 3, 44: 3,                      # closed hat / pedal hat
    46: 4,                             # open hat
    41: 5, 43: 5, 45: 5, 47: 5, 48: 5, 50: 5,  # toms
    49: 6, 57: 6, 55: 6,              # crash / splash
    51: 7, 59: 7, 53: 7,             # ride
}
# A representative GM note used to play each class back.
_DRUM_REPR = [36, 38, 39, 42, 46, 45, 49, 51, 54]


def _drum_class(note: int) -> int:
    return _DRUM_MAP.get(note, 8)  # default: generic percussion


N_FEATURES = N_DRUMS + 2 * SPAN
DRUM_SLICE = slice(0, N_DRUMS)
PLAY_SLICE = slice(N_DRUMS, N_DRUMS + SPAN)
ARTIC_SLICE = slice(N_DRUMS + SPAN, N_DRUMS + 2 * SPAN)

# Krumhansl-Schmuckler minor-key profile, used to guess each song's tonic so we
# can transpose everything to a common key. Reggaeton lives mostly in minor.
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _estimate_transpose(pitch_class_hist: np.ndarray) -> int:
    """Return the semitone shift that moves the song's tonic to C (pitch class 0)."""
    if pitch_class_hist.sum() == 0:
        return 0
    best_root, best_score = 0, -np.inf
    for root in range(12):
        profile = np.roll(_MINOR_PROFILE, root)
        score = np.corrcoef(pitch_class_hist, profile)[0, 1]
        if score > best_score:
            best_score, best_root = score, root
    return (-best_root) % 12


def _iter_events(midi_file: mido.MidiFile, steps_per_quarter: int):
    """Yield ``(step, channel, note, is_on)`` for every note event."""
    ticks_per_step = midi_file.ticks_per_beat / steps_per_quarter
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            step = int(abs_tick // ticks_per_step)
            if msg.type == "note_on" and msg.velocity > 0:
                yield step, msg.channel, msg.note, True
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                yield step, msg.channel, msg.note, False


def song_to_features(path: str, steps_per_quarter: int = 4, normalize_key: bool = True) -> np.ndarray:
    """Parse ``path`` into a ``[num_steps, N_FEATURES]`` drum+pitched matrix."""
    midi_file = mido.MidiFile(path)
    events = sorted(_iter_events(midi_file, steps_per_quarter), key=lambda e: e[0])
    if not events:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    # Decide the transpose from the pitched (non-drum) content.
    shift = 0
    if normalize_key:
        hist = np.zeros(12)
        for _, channel, note, is_on in events:
            if is_on and channel != DRUM_CHANNEL:
                hist[note % 12] += 1
        shift = _estimate_transpose(hist)

    num_steps = events[-1][0] + 1
    features = np.zeros((num_steps, N_FEATURES), dtype=np.float32)
    play = features[:, PLAY_SLICE]
    artic = features[:, ARTIC_SLICE]
    drums = features[:, DRUM_SLICE]

    note_start = {}
    for step, channel, note, is_on in events:
        if channel == DRUM_CHANNEL:
            if is_on:
                drums[step, _drum_class(note)] = 1.0
            continue
        pitch = note + shift
        if pitch < LOWER_BOUND or pitch >= UPPER_BOUND:
            continue
        idx = pitch - LOWER_BOUND
        if is_on:
            note_start[pitch] = step
            artic[step, idx] = 1.0
            play[step, idx] = 1.0
        else:
            start = note_start.pop(pitch, step)
            play[start : step + 1, idx] = 1.0

    return features


def _assemble_track(timed_messages, channel: int) -> mido.MidiTrack:
    """Turn a list of ``(tick, type, note, velocity)`` into a delta-timed track."""
    track = mido.MidiTrack()
    timed_messages = sorted(timed_messages, key=lambda m: m[0])
    last = 0
    for tick, kind, note, velocity in timed_messages:
        track.append(mido.Message(kind, note=note, velocity=velocity, channel=channel, time=tick - last))
        last = tick
    return track


def features_to_midi(
    features,
    path: str,
    steps_per_quarter: int = 4,
    tempo_bpm: float = 95.0,
    velocity: int = 96,
    ticks_per_beat: int = 480,
) -> str:
    """Write a ``[num_steps, N_FEATURES]`` matrix to a 2-track MIDI (drums + pitched)."""
    features = np.asarray(features)
    drums = features[:, DRUM_SLICE]
    play = features[:, PLAY_SLICE]
    artic = features[:, ARTIC_SLICE]
    ticks_per_step = int(round(ticks_per_beat / steps_per_quarter))
    num_steps = features.shape[0]

    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    midi_file.tracks.append(meta)

    # Drums: each onset is a short hit on the percussion channel.
    drum_msgs = []
    for step in range(num_steps):
        for cls in range(N_DRUMS):
            if drums[step, cls] > 0.5:
                tick = step * ticks_per_step
                drum_msgs.append((tick, "note_on", _DRUM_REPR[cls], velocity))
                drum_msgs.append((tick + ticks_per_step // 2, "note_off", _DRUM_REPR[cls], 0))
    midi_file.tracks.append(_assemble_track(drum_msgs, channel=DRUM_CHANNEL))

    # Pitched: reconstruct sustained notes from play + articulate bits.
    pitched_msgs = []
    sounding = np.zeros(SPAN, dtype=bool)
    for step in range(num_steps):
        tick = step * ticks_per_step
        for i in range(SPAN):
            is_playing = play[step, i] > 0.5
            is_struck = artic[step, i] > 0.5
            if sounding[i] and (not is_playing or is_struck):
                pitched_msgs.append((tick, "note_off", i + LOWER_BOUND, 0))
                sounding[i] = False
            if is_playing and not sounding[i]:
                pitched_msgs.append((tick, "note_on", i + LOWER_BOUND, velocity))
                sounding[i] = True
    final_tick = num_steps * ticks_per_step
    for i in range(SPAN):
        if sounding[i]:
            pitched_msgs.append((final_tick, "note_off", i + LOWER_BOUND, 0))
    midi_file.tracks.append(_assemble_track(pitched_msgs, channel=0))

    midi_file.save(path)
    return path
