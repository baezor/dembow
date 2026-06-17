"""A small LSTM that learns the reggaeton groove over time.

Unlike the RBM -- which models a static "bag of notes" with no sense of order --
this network reads the song one grid-step at a time and predicts the next step.
That is what lets it learn sequential structure: the dembow kick/snare pattern,
where the bass lands, how a phrase moves. It is the default engine; the RBM
remains available as the original "classic mode".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .representation import (
    ARTIC_SLICE,
    DRUM_SLICE,
    N_DRUMS,
    N_FEATURES,
    PLAY_SLICE,
    SPAN,
)


@dataclass
class LSTMConfig:
    n_features: int = N_FEATURES
    hidden: int = 256
    layers: int = 2
    steps_per_quarter: int = 4


class DembowLSTM(nn.Module):
    """Predicts the next grid-step's multi-hot feature vector."""

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.n_features, config.hidden)
        self.lstm = nn.LSTM(config.hidden, config.hidden, num_layers=config.layers, batch_first=True)
        self.output = nn.Linear(config.hidden, config.n_features)

    def forward(self, x, state=None):
        h = torch.relu(self.input(x))
        h, state = self.lstm(h, state)
        return self.output(h), state

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prime: np.ndarray,
        num_steps: int = 64,
        max_pitched: int = 5,
        drum_threshold: float = 0.4,
        pitch_threshold: float = 0.5,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Autoregressively roll out ``num_steps`` from a priming sequence.

        ``prime`` is a ``[P, N_FEATURES]`` seed (e.g. the opening of a real song).
        Pitched output is kept sparse -- at most ``max_pitched`` notes per step,
        chosen by probability -- so the result is music, not a wall of noise.
        """
        self.eval()
        device = torch.device(device)
        self.to(device)

        prime_t = torch.from_numpy(prime.astype(np.float32)).to(device).unsqueeze(0)
        _, state = self.forward(prime_t)
        current = prime_t[:, -1:, :]

        generated = []
        prev_play = prime[-1, PLAY_SLICE] > 0.5
        for _ in range(num_steps):
            logits, state = self.forward(current, state)
            probs = torch.sigmoid(logits.squeeze(0).squeeze(0) / temperature).cpu().numpy()

            step = np.zeros(N_FEATURES, dtype=np.float32)
            # Drums: independent threshold per class.
            step[DRUM_SLICE] = (probs[DRUM_SLICE] > drum_threshold).astype(np.float32)

            # Pitched: keep only the few most likely notes above threshold.
            play_probs = probs[PLAY_SLICE].copy()
            candidates = np.where(play_probs > pitch_threshold)[0]
            if len(candidates) > max_pitched:
                candidates = candidates[np.argsort(play_probs[candidates])[-max_pitched:]]
            play = np.zeros(SPAN, dtype=np.float32)
            play[candidates] = 1.0
            step[PLAY_SLICE] = play
            # Articulate deterministically: a note is struck when it starts.
            step[ARTIC_SLICE] = ((play > 0.5) & ~prev_play).astype(np.float32)
            prev_play = play > 0.5

            generated.append(step)
            current = torch.from_numpy(step).to(device).view(1, 1, -1)

        return np.array(generated, dtype=np.float32)

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {"model_type": "lstm", "config": asdict(self.config), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "DembowLSTM":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(LSTMConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
