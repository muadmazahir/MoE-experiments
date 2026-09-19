"""A small decoder-only transformer whose feed-forward blocks are MoE layers.

Testing a bare MoE layer on a synthetic regression target does not exercise the
thing we actually care about: whether the router learns a useful, non-degenerate
partition of real token distributions. Language modelling does, so the MoE layer
is dropped into the FFN slot of a GPT-style block and every strategy is compared
on next-token prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from moelab.layers import MoELayer, RouterStats


@dataclass
class MoEConfig:
    vocab_size: int = 256
    block_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    num_experts: int = 8
    k: int = 2
    d_hidden: int = 256  # derived in __post_init__ as 2 * d_model
    dropout: float = 0.0
    aux_alpha: float = 0.0
    # Per-expert token budget as a multiple of the fair share N*k/E; None means
    # no cap, so an imbalanced router wastes parameters but never drops a token.
    capacity_factor: float | None = None
    # "auto" (raw gate for k=1, renormalised otherwise), "renorm" or "raw".
    gate_norm: str = "auto"
    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.d_hidden = 2 * self.d_model


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MoELayer(
            cfg.d_model,
            cfg.d_hidden,
            cfg.num_experts,
            cfg.k,
            aux_alpha=cfg.aux_alpha,
            dropout=cfg.dropout,
            capacity_factor=cfg.capacity_factor,
            gate_norm=cfg.gate_norm,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, RouterStats]:
        x = x + self.attn(self.ln1(x))
        out, aux, stats = self.ffn(self.ln2(x))
        return x + out, aux, stats


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None
    aux_loss: Tensor
    router_stats: list[RouterStats] = field(default_factory=list)


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # -- routing controls -------------------------------------------------

    @property
    def moe_layers(self) -> list[MoELayer]:
        return [b.ffn for b in self.blocks]

    def set_routing(self, mode: str | None = None, epsilon: float | None = None,
                    gate_temp: float | None = None, aux_alpha: float | None = None,
                    surrogate_alpha: float | None = None,
                    conf_threshold: float | None = None,
                    tau_mult: float | None = None) -> None:
        """Rewrite the routing policy of every MoE layer in place."""
        for layer in self.moe_layers:
            if mode is not None:
                layer.mode = mode
            if epsilon is not None:
                layer.epsilon = epsilon
            if gate_temp is not None:
                layer.gate_temperature = gate_temp
            if aux_alpha is not None:
                layer.aux_alpha = aux_alpha
            if surrogate_alpha is not None:
                layer.surrogate_alpha = surrogate_alpha
            if conf_threshold is not None:
                layer.conf_threshold = conf_threshold
            if tau_mult is not None:
                layer.tau_mult = tau_mult

    # -- forward ----------------------------------------------------------

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> ModelOutput:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        aux_total = x.new_zeros(())
        stats: list[RouterStats] = []
        for block in self.blocks:
            x, aux, s = block(x)
            aux_total = aux_total + aux
            stats.append(s)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return ModelOutput(logits, loss, aux_total, stats)
