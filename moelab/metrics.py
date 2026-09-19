"""Router-collapse diagnostics and an analytic FLOP model.

Collapse is not visible in the loss curve -- a collapsed MoE trains fine, it
just wastes most of its parameters -- so it has to be measured directly from the
dispatch histogram. FLOPs are counted analytically from the actual number of
(token, expert) evaluations rather than timed, so the comparison is hardware- and
implementation-independent.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass
class CollapseMetrics:
    """Summary of how evenly a router spreads tokens across its experts."""

    load: list[float]  # fraction of dispatched token-slots per expert
    entropy_norm: float  # H(load) / ln(E): 1.0 = perfectly balanced, 0 = collapsed
    max_load: float  # busiest expert's share
    dead_experts: int  # experts below dead_threshold of a uniform share
    # Reported in the JSON only. They are alternative summaries of the same
    # histogram as entropy_norm, so printing them alongside it says nothing new.
    effective_experts: float  # exp(H(load)): how many experts are "really" in use
    cv: float  # coefficient of variation of the load; 0 = balanced
    gini: float  # 0 = perfectly even, ->1 = all mass on one expert


def collapse_metrics(counts: Tensor, dead_threshold: float = 0.1) -> CollapseMetrics:
    """Build collapse metrics from accumulated dispatch counts.

    Args:
        counts: (E,) total token-slots dispatched to each expert.
        dead_threshold: an expert is "dead" below this multiple of a uniform
            share (default: under 10% of its fair share).
    """
    # Via the host: the histogram is tiny, and MPS has no float64.
    c = counts.cpu().to(torch.float64)
    total = c.sum().clamp_min(1.0)
    load = (c / total).tolist()
    e = len(load)
    ln_e = math.log(e)

    h = -sum(p * math.log(p) for p in load if p > 0.0)
    mean = 1.0 / e
    var = sum((p - mean) ** 2 for p in load) / e

    order = sorted(load)
    cum = 0.0
    for i, p in enumerate(order, start=1):
        cum += i * p
    gini = (2.0 * cum) / (e * sum(order)) - (e + 1.0) / e if sum(order) > 0 else 0.0

    return CollapseMetrics(
        load=[round(p, 6) for p in load],
        entropy_norm=h / ln_e,
        max_load=max(load),
        dead_experts=sum(1 for p in load if p < dead_threshold * mean),
        effective_experts=math.exp(h),
        cv=math.sqrt(var) / mean,
        gini=gini,
    )


def metrics_to_dict(m: CollapseMetrics) -> dict:
    return asdict(m)


class FlopModel:
    """Analytic forward-FLOP accounting for the MoE transformer.

    FLOPs are 2 x MACs (one multiply, one add). Backward is charged at 2x
    forward, the standard approximation, so a training step costs ~3x forward.
    Expert cost is driven by the measured number of (token, expert) evaluations,
    which is exactly what differs between a dense pass (E per token) and a
    top-k pass (k per token).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        d, t, v = cfg.d_model, cfg.block_size, cfg.vocab_size
        n_moe = cfg.n_layer  # every block's feed-forward is an MoE layer

        # Per-token MACs for everything that does not depend on routing.
        attn_proj = 4 * d * d  # qkv + output projection
        attn_matmul = d * (t + 1)  # causal qk^T and attn@v, averaged over positions
        router = cfg.d_model * cfg.num_experts
        self.backbone_macs_per_token = (
            cfg.n_layer * (attn_proj + attn_matmul)
            + n_moe * router
            + d * v  # tied LM head
        )
        self.expert_macs_per_pair = 2 * d * cfg.d_hidden
        self.n_moe_layers = n_moe

    def forward_flops(self, n_tokens: int, expert_pairs: int) -> int:
        """Args: expert_pairs summed over all MoE layers and all tokens."""
        macs = n_tokens * self.backbone_macs_per_token
        macs += expert_pairs * self.expert_macs_per_pair
        return 2 * macs

    def train_flops(self, n_tokens: int, expert_pairs: int) -> int:
        return 3 * self.forward_flops(n_tokens, expert_pairs)

    def inference_flops_per_token(self) -> int:
        """Steady-state sparse inference: k experts per token per MoE layer."""
        pairs = self.n_moe_layers * self.cfg.k
        return self.forward_flops(1, pairs)

    def dense_equivalent_flops_per_token(self) -> int:
        """What the same model would cost with no sparsity at all."""
        pairs = self.n_moe_layers * self.cfg.num_experts
        return self.forward_flops(1, pairs)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    expert = sum(
        p.numel() for layer in model.moe_layers for p in layer.experts.parameters()
    )
    per_expert = expert // max(
        sum(layer.num_experts for layer in model.moe_layers), 1
    )
    active = total - expert + per_expert * model.cfg.k * len(model.moe_layers)
    return {
        "total_params": total,
        "expert_params": expert,
        "active_params_per_token": active,
        "sparsity": 1.0 - active / total if total else 0.0,
    }
