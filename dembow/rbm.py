"""A Restricted Boltzmann Machine in PyTorch.

This is the heart of Dembow and the part whose essence we are most careful to
preserve. The original project trained an RBM with single-step contrastive
divergence (CD-1) using a hand-written TensorFlow 1.x graph. Here we keep the
exact same model -- visible/hidden units, Gibbs sampling, contrastive
divergence -- but express it in modern PyTorch so it is readable, runs on
today's machines, and can use a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class RBMConfig:
    """Everything needed to reconstruct the model and decode its output."""

    n_visible: int
    n_hidden: int
    num_timesteps: int
    span: int
    steps_per_quarter: int = 4


class RBM:
    """Bernoulli-Bernoulli Restricted Boltzmann Machine trained with CD-k."""

    def __init__(self, config: RBMConfig, device: str | torch.device = "cpu", seed: int | None = None):
        self.config = config
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        # Small random weights, zero biases -- same initialization scheme as the
        # original (random_normal weights with 0.01 std, zero biases).
        self.W = (0.01 * torch.randn(config.n_visible, config.n_hidden, device=self.device))
        self.bv = torch.zeros(config.n_visible, device=self.device)
        self.bh = torch.zeros(config.n_hidden, device=self.device)

    # -- sampling helpers ---------------------------------------------------

    @staticmethod
    def _bernoulli(probs: torch.Tensor) -> torch.Tensor:
        """Sample a 0/1 tensor from a tensor of probabilities."""
        return torch.bernoulli(probs)

    def sample_hidden(self, v: torch.Tensor):
        """Propagate visible -> hidden. Returns (probabilities, samples)."""
        prob_h = torch.sigmoid(v @ self.W + self.bh)
        return prob_h, self._bernoulli(prob_h)

    def sample_visible(self, h: torch.Tensor):
        """Propagate hidden -> visible. Returns (probabilities, samples)."""
        prob_v = torch.sigmoid(h @ self.W.t() + self.bv)
        return prob_v, self._bernoulli(prob_v)

    def gibbs(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Run a k-step Gibbs chain starting from visible state ``v``."""
        for _ in range(k):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)
        return v

    # -- training -----------------------------------------------------------

    def contrastive_divergence(self, v0: torch.Tensor, lr: float = 0.005, k: int = 1) -> float:
        """One CD-k weight update on a batch. Returns reconstruction error."""
        prob_h0, h0 = self.sample_hidden(v0)

        vk = v0
        hk = h0
        for _ in range(k):
            _, vk = self.sample_visible(hk)
            prob_hk, hk = self.sample_hidden(vk)

        batch_size = v0.shape[0]
        # Positive and negative associations, averaged over the batch.
        pos = v0.t() @ prob_h0
        neg = vk.t() @ prob_hk
        self.W += lr * (pos - neg) / batch_size
        self.bv += lr * torch.sum(v0 - vk, dim=0) / batch_size
        self.bh += lr * torch.sum(prob_h0 - prob_hk, dim=0) / batch_size

        return float(torch.mean((v0 - vk) ** 2))

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, num_samples: int, k: int = 1, init: torch.Tensor | None = None) -> torch.Tensor:
        """Gibbs-sample ``num_samples`` visible vectors from the model."""
        if init is None:
            init = torch.zeros(num_samples, self.config.n_visible, device=self.device)
        else:
            init = init.to(self.device)
        return self.gibbs(init, k=k)

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "W": self.W.cpu(),
                "bv": self.bv.cpu(),
                "bh": self.bh.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "RBM":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(RBMConfig(**checkpoint["config"]), device=device)
        model.W = checkpoint["W"].to(model.device)
        model.bv = checkpoint["bv"].to(model.device)
        model.bh = checkpoint["bh"].to(model.device)
        return model
