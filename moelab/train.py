"""Training and evaluation harness shared by every strategy.

Fairness rules baked in here:

  * identical model init (same seed), identical data order, identical eval set;
  * the budget that is held constant is *tokens seen*, not optimiser steps, so
    the gradient-accumulation strategy does not silently get 8x the data --
    it gets the same data and fewer, larger updates, which is the real trade;
  * the learning-rate schedule is indexed by token progress, not step count,
    so accumulation does not change the shape of the LR curve;
  * every strategy is evaluated in the sparse top-k regime with exploration off,
    because that is how any of them would actually be deployed.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from moelab.data import BatchSampler
from moelab.metrics import CollapseMetrics, FlopModel, collapse_metrics, count_params
from moelab.strategies import Strategy
from moelab.transformer import MoEConfig, MoETransformer


@dataclass
class TrainConfig:
    steps: int = 3000  # micro steps == the shared token budget
    batch_size: int = 32
    block_size: int = 128
    lr: float = 3e-4
    lr_warmup_frac: float = 0.05
    lr_min_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    seed: int = 0
    device: str = "cpu"
    # "tokens": every strategy sees the same number of tokens (steps micro-steps),
    #   so gradient accumulation trades update count for update quality.
    # "steps":  every strategy takes the same number of optimiser steps, so the
    #   accumulating strategy consumes accum_steps x more tokens and FLOPs.
    equalize: str = "tokens"


class RouterAccumulator:
    """Accumulates per-layer routing telemetry on-device across many steps."""

    def __init__(self, n_layers: int, n_experts: int, device: torch.device):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.counts = [
            torch.zeros(self.n_experts, dtype=torch.long, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.n_tokens = 0
        self.dense_tokens = 0
        self.expert_pairs = 0
        self.dropped_pairs = 0

    def update(self, stats_list) -> None:
        for i, s in enumerate(stats_list):
            self.counts[i] += s.counts
            self.n_tokens += s.n_tokens
            self.dense_tokens += s.dense_tokens
            self.expert_pairs += s.expert_token_pairs
            self.dropped_pairs += s.dropped_pairs

    def summarize(self) -> tuple[list[CollapseMetrics], CollapseMetrics]:
        """Per-layer metrics plus a pooled 'whole model' view."""
        per_layer = [collapse_metrics(self.counts[i]) for i in range(self.n_layers)]
        pooled = collapse_metrics(torch.stack(self.counts).sum(0))
        return per_layer, pooled


def lr_at(cfg: TrainConfig, progress: float) -> float:
    """Cosine schedule with warm-up, indexed by fraction of the token budget."""
    if progress < cfg.lr_warmup_frac:
        return cfg.lr * progress / max(cfg.lr_warmup_frac, 1e-9)
    t = (progress - cfg.lr_warmup_frac) / max(1.0 - cfg.lr_warmup_frac, 1e-9)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return cfg.lr * (cfg.lr_min_frac + (1.0 - cfg.lr_min_frac) * cos)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(
    model: MoETransformer, batches: list[tuple[Tensor, Tensor]], device: torch.device
) -> dict:
    """Val loss plus collapse metrics, always under greedy sparse routing."""
    # mode/epsilon/temperature are training aids and are switched off; tau_mult
    # is deliberately left alone, because threshold selection is the strategy's
    # routing rule and the model has to be scored under the rule it trained on.
    saved = [(l.mode, l.epsilon, l.gate_temperature) for l in model.moe_layers]
    model.eval()
    model.set_routing(mode="topk", epsilon=0.0, gate_temp=1.0)

    acc = RouterAccumulator(len(model.moe_layers), model.cfg.num_experts, device)
    total, n = torch.zeros((), device=device), 0
    for x, y in batches:
        out = model(x, y)
        total += out.loss * y.numel()
        n += y.numel()
        acc.update(out.router_stats)
    loss = (total / n).item()

    per_layer, pooled = acc.summarize()
    for layer, (mode, eps, temp) in zip(model.moe_layers, saved):
        layer.mode, layer.epsilon, layer.gate_temperature = mode, eps, temp
    model.train()
    return {
        "val_loss": loss,
        "val_ppl": math.exp(min(loss, 20.0)),
        "pooled": pooled,
        "per_layer": per_layer,
        "eval_expert_pairs": acc.expert_pairs,
        "eval_drop_rate": acc.dropped_pairs / max(acc.expert_pairs + acc.dropped_pairs, 1),
        "eval_tokens": acc.n_tokens // max(len(model.moe_layers), 1),
        "eval_experts_per_token": acc.expert_pairs / max(acc.n_tokens, 1),
    }


def train_one(
    strategy: Strategy,
    model_cfg: MoEConfig,
    train_cfg: TrainConfig,
    train_data: Tensor,
    eval_batches: list[tuple[Tensor, Tensor]],
    verbose: bool = True,
) -> dict:
    """Train a single strategy end to end and return its full result record."""
    device = torch.device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)  # identical init across strategies

    if strategy.eps_start > 0.0:
        # Exploration draws k *distinct* experts from outside the top k, so the
        # complement has to be big enough to supply them. k = 1 is also excluded
        # deliberately: an exploring token would then hold nothing but the random
        # expert, throwing its prediction away rather than adding a trial to it.
        if model_cfg.k < 2:
            raise ValueError(
                f"the '{strategy.name}' strategy needs k >= 2 so that an exploring "
                f"token keeps a real expert alongside the one it is trying; got k=1"
            )
        if model_cfg.num_experts < 2 * model_cfg.k:
            raise ValueError(
                f"the '{strategy.name}' strategy needs num_experts >= 2k to draw k "
                f"distinct experts from outside the top-k; got "
                f"num_experts={model_cfg.num_experts}, k={model_cfg.k}"
            )

    cfg = MoEConfig(**{**asdict(model_cfg), "aux_alpha": strategy.aux_alpha})
    model = MoETransformer(cfg).to(device)
    model.train()

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )

    # Static for the whole run; the per-step schedule below never touches it.
    model.set_routing(surrogate_alpha=strategy.surrogate_alpha)

    sampler = BatchSampler(
        train_data, cfg.block_size, train_cfg.batch_size, train_cfg.seed + 777, device
    )
    flops = FlopModel(cfg)
    train_acc = RouterAccumulator(len(model.moe_layers), cfg.num_experts, device)

    history: list[dict] = []
    cum_expert_pairs = 0
    cum_dropped_pairs = 0
    cum_tokens = 0
    dense_micro_steps = 0
    opt_steps = 0
    accum = max(1, strategy.accum_steps)
    # Under "steps" equalisation the accumulating strategy runs accum x more
    # micro-batches so that every strategy performs the same number of updates.
    total_micro = train_cfg.steps * (accum if train_cfg.equalize == "steps" else 1)
    eval_every = train_cfg.eval_every * (accum if train_cfg.equalize == "steps" else 1)

    if verbose:
        print(f"\n=== {strategy.label} ===")
        print(f"    {strategy.description}")
        print(
            f"    {'step':>6} {'e/tok':>6} {'eps':>5} {'train':>7} "
            f"{'val':>7} {'ppl':>8} {'H_norm':>7} {'dead':>4} {'drop%':>6} "
            f"{'TFLOPs':>8}"
        )

    _sync(device)
    t0 = time.perf_counter()

    for micro in range(total_micro):
        ph = strategy.phase(micro, total_micro)
        model.set_routing(**asdict(ph))
        if ph.mode == "dense":
            dense_micro_steps += 1

        x, y = sampler()
        out = model(x, y)
        loss = out.loss + out.aux_loss
        (loss / accum).backward()

        train_acc.update(out.router_stats)
        cum_expert_pairs += sum(s.expert_token_pairs for s in out.router_stats)
        cum_dropped_pairs += sum(s.dropped_pairs for s in out.router_stats)
        cum_tokens += y.numel()

        if (micro + 1) % accum == 0 or micro == total_micro - 1:
            lr = lr_at(train_cfg, micro / total_micro)
            for group in opt.param_groups:
                group["lr"] = lr
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_steps += 1

        last = micro == total_micro - 1
        if (micro + 1) % eval_every == 0 or last:
            _sync(device)
            elapsed = time.perf_counter() - t0
            ev = evaluate(model, eval_batches, device)
            _, train_pooled = train_acc.summarize()
            dense_frac = train_acc.dense_tokens / max(train_acc.n_tokens, 1)
            # Experts actually evaluated per token per layer: exactly k for
            # plain top-k, and variable for the dense-fallback and threshold
            # strategies -- which is the whole story for both of them.
            experts_per_token = train_acc.expert_pairs / max(train_acc.n_tokens, 1)
            train_acc.reset()
            rec = {
                "micro_step": micro + 1,
                "opt_step": opt_steps,
                "mode": ph.mode,
                "epsilon": round(ph.epsilon, 4),
                "gate_temp": round(ph.gate_temp, 3),
                "conf_threshold": round(ph.conf_threshold, 4),
                "tau_mult": round(ph.tau_mult, 4),
                "train_dense_frac": dense_frac,
                "train_experts_per_token": experts_per_token,
                "train_loss": out.loss.item(),
                "aux_loss": out.aux_loss.item(),
                "val_loss": ev["val_loss"],
                "val_ppl": ev["val_ppl"],
                "tokens_seen": cum_tokens,
                "train_flops": flops.train_flops(cum_tokens, cum_expert_pairs),
                "expert_pairs": cum_expert_pairs,
                "wall_s": elapsed,
                "eval_entropy_norm": ev["pooled"].entropy_norm,
                "eval_effective_experts": ev["pooled"].effective_experts,
                "eval_max_load": ev["pooled"].max_load,
                "eval_dead_experts": ev["pooled"].dead_experts,
                "eval_drop_rate": ev["eval_drop_rate"],
                "eval_cv": ev["pooled"].cv,
                "train_entropy_norm": train_pooled.entropy_norm,
                "per_layer_entropy_norm": [m.entropy_norm for m in ev["per_layer"]],
                "per_layer_dead": [m.dead_experts for m in ev["per_layer"]],
            }
            history.append(rec)
            if verbose:
                print(
                    f"    {micro + 1:6d} {experts_per_token:6.2f} {ph.epsilon:5.2f} "
                    f"{rec['train_loss']:7.3f} "
                    f"{rec['val_loss']:7.3f} {rec['val_ppl']:8.2f} "
                    f"{rec['eval_entropy_norm']:7.3f} {rec['eval_dead_experts']:4d} "
                    f"{100 * rec['eval_drop_rate']:6.2f} "
                    f"{rec['train_flops'] / 1e12:8.1f}"
                )

    _sync(device)
    wall = time.perf_counter() - t0
    final = evaluate(model, eval_batches, device)

    return {
        "strategy": strategy.name,
        "label": strategy.label,
        "description": strategy.description,
        "config": {"model": asdict(cfg), "train": asdict(train_cfg),
                   "strategy": asdict(strategy)},
        "params": count_params(model),
        "final": {
            "val_loss": final["val_loss"],
            "val_ppl": final["val_ppl"],
            "drop_rate": final["eval_drop_rate"],
            "experts_per_token": final["eval_experts_per_token"],
            "collapse": asdict(final["pooled"]),
            "per_layer_collapse": [asdict(m) for m in final["per_layer"]],
        },
        "efficiency": {
            "wall_s": wall,
            "opt_steps": opt_steps,
            "micro_steps": total_micro,
            "tokens_seen": cum_tokens,
            "dense_micro_steps": dense_micro_steps,
            "train_expert_pairs": cum_expert_pairs,
            "train_dropped_pairs": cum_dropped_pairs,
            "train_drop_rate": cum_dropped_pairs
            / max(cum_expert_pairs + cum_dropped_pairs, 1),
            "train_flops": flops.train_flops(cum_tokens, cum_expert_pairs),
            "sparse_baseline_train_flops": flops.train_flops(
                cum_tokens,
                cum_tokens * len(model.moe_layers) * cfg.k,
            ),
            "inference_flops_per_token": flops.inference_flops_per_token(),
            "dense_equivalent_flops_per_token": flops.dense_equivalent_flops_per_token(),
        },
        "history": history,
    }
