"""Convert MIDI files to and from the piano-roll representation the RBM uses.

This is the modern replacement for the original ``midi_manipulation.py``, which
depended on the Python-2-only, unmaintained ``python-midi`` library. It uses
``mido`` instead and keeps the same representation the model was designed around:

A song becomes a ``[num_steps, 2 * SPAN]`` matrix. Time is quantized onto a grid
(16th notes by default). For each step and each pitch we store two bits::

    columns [0 : SPAN]        -> "play"      : the note is sounding this step
    columns [SPAN : 2 * SPAN] -> "articulate": the note was (re)struck this step

The "articulate" bit is what lets the model tell a freshly struck note apart
from one that is simply being held -- essential for capturing the percussive,
on-the-grid feel of the dembow rhythm.
"""

from __future__ import annotations

import math
from typing import List

import mido
import numpy as np

# The slice of the piano we care about. Identical to the original project so
# that the representation -- and the model's notion of "note range" -- is
# preserved. 24 == C1, 102 == F#7.
LOWER_BOUND = 24
UPPER_BOUND = 102
SPAN = UPPER_BOUND - LOWER_BOUND  # 78 pitches


def _iter_note_events(midi_file: mido.MidiFile):
    """Yield ``(abs_tick, pitch, is_on)`` for every note event in the file.

    All tracks are merged onto a single absolute-tick timeline. A note_on with
    zero velocity is treated as a note_off, per the MIDI spec.
    """
    for track in midi_file.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                yield abs_tick, msg.note, True
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                yield abs_tick, msg.note, False


def midi_to_note_state_matrix(path: str, steps_per_quarter: int = 4) -> np.ndarray:
    """Parse ``path`` into a ``[num_steps, 2 * SPAN]`` piano-roll matrix.

    ``steps_per_quarter`` controls the time grid: 4 means a 16th-note grid,
    which is what the original code effectively used.
    """
    midi_file = mido.MidiFile(path)
    ticks_per_step = midi_file.ticks_per_beat / steps_per_quarter

    events = sorted(_iter_note_events(midi_file), key=lambda e: e[0])
    if not events:
        return np.zeros((0, 2 * SPAN), dtype=np.float32)

    last_tick = events[-1][0]
    num_steps = int(math.floor(last_tick / ticks_per_step)) + 1

    play = np.zeros((num_steps, SPAN), dtype=np.float32)
    articulate = np.zeros((num_steps, SPAN), dtype=np.float32)

    # Track when each pitch was turned on so we can fill in the held steps.
    note_start_step = {}
    for abs_tick, pitch, is_on in events:
        if pitch < LOWER_BOUND or pitch >= UPPER_BOUND:
            continue
        idx = pitch - LOWER_BOUND
        step = int(math.floor(abs_tick / ticks_per_step))
        if is_on:
            note_start_step[pitch] = step
            articulate[step, idx] = 1.0
            play[step, idx] = 1.0
        else:
            start = note_start_step.pop(pitch, step)
            play[start : step + 1, idx] = 1.0

    return np.concatenate([play, articulate], axis=1)


def note_state_matrix_to_midi(
    statematrix,
    path: str,
    steps_per_quarter: int = 4,
    tempo_bpm: float = 95.0,
    velocity: int = 90,
    ticks_per_beat: int = 480,
) -> str:
    """Write a ``[num_steps, 2 * SPAN]`` matrix back out to a MIDI file.

    ``tempo_bpm`` defaults to 95 -- right in the reggaeton pocket.
    """
    statematrix = np.asarray(statematrix)
    play = statematrix[:, :SPAN]
    articulate = statematrix[:, SPAN:]

    ticks_per_step = int(round(ticks_per_beat / steps_per_quarter))

    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    num_steps = play.shape[0]
    sounding = np.zeros(SPAN, dtype=bool)
    pending: List[mido.Message] = []
    last_event_tick = 0

    def flush(messages, current_tick):
        nonlocal last_event_tick
        for msg in messages:
            track.append(msg.copy(time=current_tick - last_event_tick))
            last_event_tick = current_tick

    for step in range(num_steps):
        tick = step * ticks_per_step
        ons, offs = [], []
        for i in range(SPAN):
            is_playing = play[step, i] > 0.5
            is_struck = articulate[step, i] > 0.5
            if sounding[i] and (not is_playing or is_struck):
                offs.append(i)
                sounding[i] = False
            if is_playing and not sounding[i]:
                ons.append(i)
                sounding[i] = True
        msgs = [mido.Message("note_off", note=i + LOWER_BOUND, velocity=0) for i in offs]
        msgs += [mido.Message("note_on", note=i + LOWER_BOUND, velocity=velocity) for i in ons]
        if msgs:
            flush(msgs, tick)

    # Turn off anything still ringing at the end.
    final_tick = num_steps * ticks_per_step
    tail = [
        mido.Message("note_off", note=i + LOWER_BOUND, velocity=0)
        for i in range(SPAN)
        if sounding[i]
    ]
    if tail:
        flush(tail, final_tick)

    midi_file.save(path)
    return path
