"""Render generated MIDI to audio so you can actually hear it.

Two backends, tried in order:

1. **FluidSynth** -- if the ``fluidsynth`` binary and a SoundFont are available,
   use them for the best, most realistic sound.
2. **Built-in NumPy synth** -- a tiny, dependency-free oscillator/noise synth so
   rendering works everywhere even without FluidSynth installed. It won't sound
   like a studio, but you'll hear the groove.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import wave
from typing import List, Optional

import mido
import numpy as np

_SOUNDFONT_PATHS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
]


def _find_soundfont(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    return next((p for p in _SOUNDFONT_PATHS if os.path.exists(p)), None)


def _render_fluidsynth(midi_path: str, wav_path: str, soundfont: str, sample_rate: int) -> bool:
    try:
        subprocess.run(
            ["fluidsynth", "-ni", "-F", wav_path, "-r", str(sample_rate), soundfont, midi_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return os.path.exists(wav_path)
    except Exception:
        return False


def _adsr(n: int, sample_rate: int, release: float = 0.05) -> np.ndarray:
    """A simple attack/release envelope to avoid clicks."""
    env = np.ones(n)
    attack = min(n, int(0.005 * sample_rate))
    rel = min(n, int(release * sample_rate))
    if attack:
        env[:attack] = np.linspace(0, 1, attack)
    if rel:
        env[-rel:] *= np.linspace(1, 0, rel)
    return env


def _tone(freq: float, dur: float, sample_rate: int, harmonics=(1.0, 0.4, 0.2)) -> np.ndarray:
    n = max(1, int(dur * sample_rate))
    t = np.arange(n) / sample_rate
    wave_out = sum(amp * np.sin(2 * np.pi * freq * h * t) for h, amp in enumerate(harmonics, start=1))
    return wave_out * _adsr(n, sample_rate)


def _drum(note: int, sample_rate: int) -> np.ndarray:
    """Percussion: pitched sine for the kick, shaped noise for snare/hats."""
    if note <= 37:  # kick
        n = int(0.18 * sample_rate)
        t = np.arange(n) / sample_rate
        freq = 110 * np.exp(-30 * t) + 45  # pitch drop
        return np.sin(2 * np.pi * freq * t) * np.exp(-12 * t)
    if note in (38, 39, 40):  # snare / clap
        n = int(0.12 * sample_rate)
        return np.random.randn(n) * np.exp(-22 * np.arange(n) / sample_rate)
    # hats / cymbals: short bright noise
    n = int(0.05 * sample_rate)
    return np.random.randn(n) * np.exp(-60 * np.arange(n) / sample_rate)


def _builtin_synth(midi_file: mido.MidiFile, sample_rate: int) -> np.ndarray:
    """Synthesize a MIDI file to a mono float waveform with no external deps."""
    # Collect notes as (start_sec, dur_sec, channel, note, velocity).
    notes = []
    open_notes = {}
    abs_time = 0.0
    for msg in midi_file:  # iterating a MidiFile yields delta time in seconds
        abs_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.channel == 9:
                notes.append((abs_time, 0.2, 9, msg.note, msg.velocity))
            else:
                open_notes[(msg.channel, msg.note)] = (abs_time, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in open_notes:
                start, vel = open_notes.pop(key)
                notes.append((start, max(0.05, abs_time - start), msg.channel, msg.note, vel))

    if not notes:
        return np.zeros(sample_rate, dtype=np.float32)

    total = max(s + d for s, d, *_ in notes) + 0.3
    buf = np.zeros(int(total * sample_rate) + sample_rate, dtype=np.float32)
    for start, dur, channel, note, vel in notes:
        idx = int(start * sample_rate)
        amp = (vel / 127.0) * 0.25
        if channel == 9:
            seg = _drum(note, sample_rate) * amp
        else:
            freq = 440.0 * 2 ** ((note - 69) / 12.0)
            gain = 1.4 if channel == 1 else 1.0  # bass a touch louder
            seg = _tone(freq, dur, sample_rate) * amp * gain
        end = min(len(buf), idx + len(seg))
        buf[idx:end] += seg[: end - idx]

    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = buf / peak * 0.95
    return buf[: int(total * sample_rate)]


def _write_wav(samples: np.ndarray, path: str, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{len(pcm)}h", *pcm.tolist()))


def render_to_wav(
    midi_path: str,
    wav_path: str,
    soundfont: Optional[str] = None,
    sample_rate: int = 22050,
) -> str:
    """Render a MIDI file to WAV. Uses FluidSynth if available, else a builtin synth."""
    sf = _find_soundfont(soundfont)
    if shutil.which("fluidsynth") and sf and _render_fluidsynth(midi_path, wav_path, sf, sample_rate):
        return wav_path
    midi_file = mido.MidiFile(midi_path)
    _write_wav(_builtin_synth(midi_file, sample_rate), wav_path, sample_rate)
    return wav_path
