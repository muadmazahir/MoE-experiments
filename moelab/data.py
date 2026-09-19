"""Text corpus loading and batching.

Every strategy must see byte-identical data in the same order, so batches are
drawn from an explicitly seeded generator rather than global RNG state.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def _ensure_corpus(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


@dataclass
class Corpus:
    """A tokenised corpus split into train/val, plus its vocabulary."""

    train: Tensor
    val: Tensor
    vocab_size: int
    itos: list[str]
    stoi: dict[str, int]


def load_corpus(
    path: str | os.PathLike = "data/tinyshakespeare.txt", val_frac: float = 0.1
) -> Corpus:
    """Load the corpus and tokenise it at character level."""
    tokens = list(_ensure_corpus(Path(path)))
    vocab = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([stoi[t] for t in tokens], dtype=torch.long)

    n_val = int(len(ids) * val_frac)
    return Corpus(
        train=ids[:-n_val].clone(),
        val=ids[-n_val:].clone(),
        vocab_size=len(vocab),
        itos=vocab,
        stoi=stoi,
    )


class BatchSampler:
    """Random contiguous windows, reproducible for a given seed."""

    def __init__(self, data: Tensor, block_size: int, batch_size: int, seed: int,
                 device: torch.device):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self) -> tuple[Tensor, Tensor]:
        hi = len(self.data) - self.block_size - 1
        starts = torch.randint(0, hi, (self.batch_size,), generator=self.gen)
        x = torch.stack([self.data[s : s + self.block_size] for s in starts])
        y = torch.stack([self.data[s + 1 : s + 1 + self.block_size] for s in starts])
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def fixed_eval_batches(
    data: Tensor, block_size: int, batch_size: int, n_batches: int,
    device: torch.device, seed: int = 1234,
) -> list[tuple[Tensor, Tensor]]:
    """A frozen evaluation set, identical across every strategy and seed."""
    sampler = BatchSampler(data, block_size, batch_size, seed, device)
    return [sampler() for _ in range(n_batches)]
