"""A small decoder-only Transformer language model over music tokens.

This is the new engine: instead of an RBM's static "bag of notes" or an LSTM's
single recurrent state, it uses masked self-attention to condition every token on
the entire sequence so far. That long-range view is what lets it learn musical
structure -- phrases, repetition, how the drums and bass lock together -- one
event at a time, exactly as a GPT-style model generates text.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _banned_ngram_tokens(ids: list[int], n: int) -> list[int]:
    """Tokens that would complete a previously seen n-gram given the current tail."""
    if len(ids) < n:
        return []
    prefix = tuple(ids[-(n - 1):]) if n > 1 else ()
    banned = []
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == prefix:
            banned.append(ids[i + n - 1])
    return banned


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_len: int = 512
    dropout: float = 0.1


class MusicTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: input embedding and output projection share weights.
        self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.drop(self.token_emb(x) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 600,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = 0.92,
        repetition_penalty: float = 1.15,
        recent_window: int = 96,
        no_repeat_ngram_size: int = 0,
        eos_id: int | None = None,
        device: str | torch.device = "cpu",
    ) -> list[int]:
        """Autoregressively sample a continuation.

        Uses temperature + nucleus (top-p) sampling, plus repetition control to
        keep the model from collapsing into a degenerate loop -- a real risk when
        a Transformer is trained on so little data. ``repetition_penalty`` gently
        down-weights tokens seen in the last ``recent_window`` steps (musical
        repetition is still allowed); ``no_repeat_ngram_size`` hard-bans exact
        n-gram repeats when > 0.
        """
        self.eval()
        ids = prompt.to(device).tolist()
        for _ in range(max_new_tokens):
            context = torch.tensor(ids[-self.config.max_len:], device=device).unsqueeze(0)
            logits = self.forward(context)[0, -1]

            if repetition_penalty and repetition_penalty != 1.0:
                for t in set(ids[-recent_window:]):
                    logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty
            if no_repeat_ngram_size > 0:
                for t in _banned_ngram_tokens(ids, no_repeat_ngram_size):
                    logits[t] = -float("inf")

            logits = logits / max(temperature, 1e-6)

            if top_k is not None:
                kth = torch.topk(logits, top_k).values[-1]
                logits[logits < kth] = -float("inf")
            if top_p is not None:
                ordered, idx = torch.sort(logits, descending=True)
                probs = torch.softmax(ordered, dim=-1)
                cutoff = torch.cumsum(probs, dim=-1) > top_p
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                ordered[cutoff] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(0, idx, ordered)

            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
        return ids

    def save(self, path: str) -> None:
        torch.save({"model_type": "transformer", "config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "MusicTransformer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
