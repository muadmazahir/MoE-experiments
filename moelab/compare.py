"""Run every strategy under identical conditions and report the comparison."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from moelab.data import Corpus, fixed_eval_batches
from moelab.strategies import Strategy
from moelab.train import TrainConfig, train_one
from moelab.transformer import MoEConfig


def pick_device(requested: str = "auto") -> str:
    """Auto-select a device: CUDA, else Apple Metal, else CPU.

    One caveat, and it is about wall-clock only. Sparse dispatch is a handful of
    tiny index_select/index_add kernels per expert per layer, which on Metal is
    launch-latency bound -- measured ~6x slower than CPU at this model size --
    while the dense path is ~4x faster. So on MPS a strategy that routes tokens
    densely looks cheaper in elapsed time than it is. Every figure the comparison
    reports is analytic and unaffected; wall_s in comparison.json is not.
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_comparison(
    strategies: list[Strategy],
    model_cfg: MoEConfig,
    train_cfg: TrainConfig,
    corpus: Corpus,
    seeds: list[int],
    verbose: bool = True,
) -> dict:
    device = torch.device(train_cfg.device)
    eval_batches = fixed_eval_batches(
        corpus.val, model_cfg.block_size, train_cfg.batch_size,
        train_cfg.eval_batches, device,
    )

    runs: list[dict] = []
    for seed in seeds:
        for strategy in strategies:
            cfg = TrainConfig(**{**asdict(train_cfg), "seed": seed})
            if verbose and len(seeds) > 1:
                print(f"\n--- seed {seed} ---")
            result = train_one(strategy, model_cfg, cfg, corpus.train,
                               eval_batches, verbose=verbose)
            result["seed"] = seed
            runs.append(result)

    return {
        "runs": runs,
        "corpus": {"vocab_size": corpus.vocab_size,
                   "train_tokens": len(corpus.train), "val_tokens": len(corpus.val)},
        "device": train_cfg.device,
        "seeds": seeds,
    }


# -- reporting -----------------------------------------------------------


def _agg(runs: list[dict], name: str, path: list[str]) -> tuple[float, float]:
    vals = []
    for r in runs:
        if r["strategy"] != name:
            continue
        node = r
        for key in path:
            node = node[key]
        vals.append(float(node))
    if not vals:
        return float("nan"), 0.0
    return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def _fmt(mean: float, std: float, spec: str = ".3f") -> str:
    return f"{mean:{spec}}" if std == 0 else f"{mean:{spec}}±{std:{spec}}"


def _bar(load: list[float], width: int = 28) -> str:
    """ASCII load histogram; a flat bar is a healthy router."""
    blocks = " ▁▂▃▄▅▆▇█"
    hi = max(load) or 1.0
    per = max(1, width // len(load))
    return "".join(blocks[min(8, int(p / hi * 8 + 0.5))] * per for p in load)


def format_report(result: dict) -> str:
    runs = result["runs"]
    names = list(dict.fromkeys(r["strategy"] for r in runs))
    by_name = {n: next(r for r in runs if r["strategy"] == n) for n in names}
    lines: list[str] = []
    w = 96

    c = result["corpus"]
    ex = by_name[names[0]]
    mc = ex["config"]["model"]
    tc = ex["config"]["train"]
    lines += [
        "=" * w,
        "MoE ROUTER-COLLAPSE STRATEGY COMPARISON",
        "=" * w,
        f"corpus      : tiny-shakespeare char-level, vocab {c['vocab_size']}, "
        f"{c['train_tokens']:,} train / {c['val_tokens']:,} val tokens",
        f"model       : {mc['n_layer']}L x {mc['d_model']}d x {mc['n_head']}h, "
        f"{mc['num_experts']} experts top-{mc['k']}, block {mc['block_size']}",
        f"routing     : gate_norm={mc['gate_norm']}, capacity_factor="
        f"{mc['capacity_factor'] if mc['capacity_factor'] is not None else 'none'}, "
        f"lr={tc['lr']:g}, batch {tc['batch_size']}",
        f"params      : {ex['params']['total_params']:,} total, "
        f"{ex['params']['active_params_per_token']:,} active/token "
        f"({ex['params']['sparsity']:.1%} sparse)",
        f"budget      : equalize={tc['equalize']} -- "
        + (
            f"{tc['steps']:,} micro-steps x batch {tc['batch_size']} x "
            f"{mc['block_size']} = {ex['efficiency']['tokens_seen']:,} tokens for every "
            "strategy; accumulation buys update quality with update count"
            if tc["equalize"] == "tokens"
            else f"{tc['steps']:,} optimiser steps for every strategy; the "
            "accumulating strategy therefore consumes proportionally more tokens"
        ),
        f"device      : {result['device']}   seeds: {result['seeds']}",
        "",
        "-" * w,
        "QUALITY  (lower is better; evaluated with greedy top-k routing, exploration off)",
        "-" * w,
        f"{'strategy':<10} {'val loss':>14} {'val ppl':>14} {'tokens dropped':>16}",
    ]
    capped = mc["capacity_factor"] is not None
    for n in names:
        drop, dstd = _agg(runs, n, ["final", "drop_rate"])
        lines.append(
            f"{n:<10} {_fmt(*_agg(runs, n, ['final', 'val_loss'])):>14} "
            f"{_fmt(*_agg(runs, n, ['final', 'val_ppl']), spec='.2f'):>14} "
            + (f"{_fmt(100 * drop, 100 * dstd, '.2f') + '%':>16}" if capped
               else f"{'n/a (no cap)':>16}")
        )
    lines += [
        "",
        "-" * w,
        "ROUTER COLLAPSE  (pooled over MoE layers, greedy routing on the held-out set)",
        "-" * w,
        f"{'strategy':<10} {'H_norm↑':>12} {'max load↓':>11} {'dead↓':>7}",
    ]
    n_exp = mc["num_experts"]
    for n in names:
        lines.append(
            f"{n:<10} {_fmt(*_agg(runs, n, ['final', 'collapse', 'entropy_norm'])):>12} "
            f"{_fmt(*_agg(runs, n, ['final', 'collapse', 'max_load'])):>11} "
            f"{_fmt(*_agg(runs, n, ['final', 'collapse', 'dead_experts']), spec='.1f'):>7}"
        )
    lines += [
        f"           (perfect balance is H_norm=1.0 and max load=1/{n_exp}="
        f"{1 / n_exp:.3f}; 'dead' is pooled, which hides per-layer collapse)",
        "",
        "-" * w,
        "EFFICIENCY  (analytic FLOPs = 2xMACs, backward charged at 2x forward)",
        "-" * w,
        f"{'strategy':<10} {'train TFLOPs':>14} {'vs top-k':>10} {'opt steps':>11} "
        f"{'tokens':>13} {'experts/tok':>13}",
    ]
    for n in names:
        tf, tfs = _agg(runs, n, ["efficiency", "train_flops"])
        base, _ = _agg(runs, n, ["efficiency", "sparse_baseline_train_flops"])
        steps, _ = _agg(runs, n, ["efficiency", "opt_steps"])
        toks, _ = _agg(runs, n, ["efficiency", "tokens_seen"])
        ept, epts = _agg(runs, n, ["final", "experts_per_token"])
        lines.append(
            f"{n:<10} {_fmt(tf / 1e12, tfs / 1e12, '.1f'):>14} "
            f"{tf / base:9.2f}x {steps:11.0f} {toks:13,.0f} "
            f"{_fmt(ept, epts, '.2f'):>13}"
        )
    lines += [
        "           ('experts/tok' is expert evaluations per token at eval, so it "
        "sets inference cost. It is k minus whatever",
        "            capacity dropped -- so below k for any run with a capacity "
        "factor. Only a strategy that changes the routing",
        "            rule itself moves it on its own, and that strategy's trained "
        "artefact is then not interchangeable with the",
        "            others. 'vs top-k' >1.0 prices dense passes, <1.0 means "
        "capacity dropped work rather than served it.)",
        "",
        "-" * w,
        "PER-LAYER EXPERT LOAD  (each glyph column is one expert; flat = balanced)"
        + (f"  [seed {runs[0]['seed']} only]" if len(result["seeds"]) > 1 else ""),
        "-" * w,
    ]
    for n in names:
        run = by_name[n]
        lines.append(f"{n}  ({run['label']})")
        for i, m in enumerate(run["final"]["per_layer_collapse"]):
            lines.append(
                f"   L{i}  |{_bar(m['load'])}|  H={m['entropy_norm']:.3f} "
                f"dead={m['dead_experts']}  max={m['max_load']:.3f}"
            )
        lines.append("")

    n_evals = len(by_name[names[0]]["history"])
    pct = [f"{100 * (i + 1) / n_evals:.0f}%" for i in range(n_evals)]
    lines += [
        "-" * w,
        "TRAJECTORIES  (columns are % of each strategy's own training budget)"
        + (f"  [seed {runs[0]['seed']} only]" if len(result["seeds"]) > 1 else ""),
        "-" * w,
        "router balance H_norm (1.0 = perfectly balanced, low = collapsing)",
        "progress  " + "".join(f"{v:>8}" for v in pct),
    ]
    for n in names:
        h = by_name[n]["history"]
        lines.append(f"{n:<10}" + "".join(f"{r['eval_entropy_norm']:>8.3f}" for r in h))
    lines += ["", "val loss", "progress  " + "".join(f"{v:>8}" for v in pct)]
    for n in names:
        h = by_name[n]["history"]
        lines.append(f"{n:<10}" + "".join(f"{r['val_loss']:>8.3f}" for r in h))
    lines.append("=" * w)
    return "\n".join(lines)


def save_results(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = result
    path = out / "comparison.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
