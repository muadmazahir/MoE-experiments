#!/usr/bin/env python
"""Compare three anti-router-collapse strategies for sparse MoE transformers.

  1. aux      -- auxiliary load-balancing loss (the classical fix)
  2. explore  -- no aux loss; sparse top-k with epsilon-greedy exploration
                 outside the top-k, annealed to ~0
  4. surrogate-- no aux loss, no exploration; estimates the experts top-k skipped
                 and feeds them back straight-through for router gradient
  5. conf     -- no aux loss; tokens the router is not confident about go through
                 every expert, with the confidence threshold annealed to 0
  6. threshold-- no aux loss; selects every expert above a probability threshold
                 instead of a fixed top-k, with the threshold rising over time
  3. accum    -- no aux loss; gradients accumulated over many micro-batches
  0. none     -- control with no mitigation, so the others have a reference

Examples:
  python run_experiment.py --smoke                  # ~1 min sanity run
  python run_experiment.py                          # default comparison
  python run_experiment.py --steps 6000 --seeds 0,1,2
  python run_experiment.py --level word --methods aux,explore,accum
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from moelab.compare import format_report, pick_device, run_comparison, save_results
from moelab.data import load_corpus
from moelab.strategies import build_strategies
from moelab.train import TrainConfig
from moelab.transformer import MoEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    m = p.add_argument_group("model")
    m.add_argument("--d-model", type=int, default=128)
    m.add_argument("--n-layer", type=int, default=4)
    m.add_argument("--num-experts", type=int, default=8)
    m.add_argument("--k", type=int, default=2)
    m.add_argument("--gate-norm", default="auto", choices=["auto", "renorm", "raw"],
                   help="how the gate is formed from router logits. 'renorm' (GShard/"
                        "Mixtral) softmaxes the selected logits; 'raw' (Switch) uses "
                        "the router probability itself, so confidence scales the "
                        "expert output. 'auto' = raw for k=1, renorm otherwise.")
    m.add_argument("--capacity-factor", type=float, default=None,
                   help="per-expert token budget as a multiple of the fair share "
                        "N*k/E; overflow tokens are dropped. Unset means no cap, in "
                        "which case an imbalanced router costs parameters but not "
                        "accuracy. 1.25 is a typical production value.")

    t = p.add_argument_group("training")
    t.add_argument("--steps", type=int, default=3000,
                   help="micro-steps; the shared token budget for every strategy")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--seeds", default="0")
    t.add_argument("--equalize", default="tokens", choices=["tokens", "steps"],
                   help="hold the token budget constant (default) or the optimiser-"
                        "step count constant across strategies")
    t.add_argument("--device", default="auto")

    s = p.add_argument_group("strategy hyper-parameters")
    s.add_argument("--methods", default="none,aux,explore,accum,surrogate,conf,threshold")
    s.add_argument("--surrogate-alpha", type=float, default=1.0,
                   help="weight on the straight-through estimate of the experts "
                        "top-k skipped; 0 disables it")
    s.add_argument("--dense-warmup-frac", type=float, default=0.0,
                   help="share of training the explore strategy spends in dense "
                        "routing before going sparse; 0 (the default) starts sparse")
    s.add_argument("--eps-start", type=float, default=0.60,
                   help="initial share of tokens routed away from their top-k "
                        "experts; needs num_experts >= 2 * k")
    s.add_argument("--conf-start", type=float, default=0.80,
                   help="initial top-k probability mass a token must hold to be "
                        "routed sparsely; below it the token goes through every "
                        "expert. Annealed to 0. Useful range is (k/E, 1]")
    s.add_argument("--tau-start", type=float, default=1.0,
                   help="selection threshold at the start of training, in units "
                        "of the uniform share 1/E. Every expert above it is "
                        "selected, so the count varies per token")
    s.add_argument("--tau-end", type=float, default=1.6,
                   help="selection threshold at the end, same units. Above ~4 "
                        "nothing clears it on merit, since at most one expert "
                        "can exceed one half")
    s.add_argument("--gate-temp", type=float, default=4.0,
                   help="gate softmax temperature for explored tokens; held "
                        "constant, and never applied to an exploited token, so the "
                        "greedy path always matches deployment. 1.0 disables it")

    o = p.add_argument_group("output")
    o.add_argument("--out", default="results")
    o.add_argument("--smoke", action="store_true",
                   help="tiny fast configuration for sanity checking the harness")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    block_size = 128
    if args.smoke:
        args.steps, args.eval_every = 240, 60
        args.d_model, args.n_layer, args.batch_size = 64, 2, 16
        block_size = 64

    device = pick_device(args.device)
    torch.manual_seed(0)

    corpus = load_corpus()
    model_cfg = MoEConfig(
        vocab_size=corpus.vocab_size,
        block_size=block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        num_experts=args.num_experts,
        k=args.k,
        capacity_factor=args.capacity_factor,
        gate_norm=args.gate_norm,
    )
    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=block_size,
        lr=args.lr,
        eval_every=args.eval_every,
        device=device,
        equalize=args.equalize,
    )

    registry = build_strategies(
        dense_warmup_frac=args.dense_warmup_frac,
        eps_start=args.eps_start,
        gate_temp=args.gate_temp,
        surrogate_alpha=args.surrogate_alpha,
        conf_start=args.conf_start,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
    )
    wanted = [n.strip() for n in args.methods.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in registry]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; choose from {list(registry)}")
    strategies = [registry[n] for n in wanted]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    result = run_comparison(strategies, model_cfg, train_cfg, corpus, seeds)

    report = format_report(result)
    print("\n" + report)

    out_dir = Path(args.out)
    path = save_results(result, out_dir)
    (out_dir / "report.txt").write_text(report)
    print(f"\nwrote {path} and {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
