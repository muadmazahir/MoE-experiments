# MoE-experiments

Experiments with the MoE architecture. Currently: **six ways to stop a router from
collapsing**, compared head-to-head.

A sparse mixture-of-experts layer, dropped into a small GPT-style transformer, used to
compare six strategies for keeping the router from collapsing onto a handful of
experts — measured on **quality**, **router collapse** and **compute**.

A work-in-progress write-up of these experiments is in [paper/main.pdf](paper/main.pdf).


## The problem

A top-k router is a positive feedback loop. An expert that gets picked slightly more
often gets more gradient, becomes slightly better, and gets picked more often still.
Left alone it ends with a few experts doing all the work and the rest dead — you pay
for the parameters and get none of the capacity. The loss curve does not show it;
you have to measure the dispatch histogram directly.

## The six strategies (plus a control)

| key | strategy | how it avoids collapse |
|---|---|---|
| `none` | **control** | nothing |
| `aux` | **1. auxiliary load-balancing loss** | adds `alpha * E * sum_i f_i * P_i` to the task loss, which is minimised when dispatch load `f` and router mass `P` are both uniform |
| `explore` | **2. epsilon-greedy exploration outside the top-k** | sparse from step 0; a sampled token abandons its whole top-k and is routed to `k` distinct experts drawn from *outside* it, so written-off experts keep getting tried and keep receiving router gradient |
| `accum` | **3. gradient accumulation** | accumulates gradients over many micro-batches before each optimiser step, so each router update is averaged over a large, diverse token population instead of reinforcing whatever one small batch preferred |
| `surrogate` | **4. straight-through estimate of the unevaluated experts** | estimates what the `E - k` skipped experts would have output by fitting "output is proportional to router probability" on the k that ran, and feeds it back straight-through — no forward change, but every router logit gets gradient |
| `conf` | **5. dense fallback where the router is unsure** | a token whose top-k experts hold less than a threshold of the probability mass goes through *every* expert instead, so an undecided token is never forced into a guess the feedback loop would reinforce. The threshold anneals to 0 |
| `threshold` | **6. threshold selection instead of top-k** | selects *every* expert above a probability threshold rather than a fixed k, so the count varies per token. The threshold rises over training, pruning experts the router has earned the right to dismiss |

Each strategy is written up in **[STRATEGIES.md](STRATEGIES.md)** — the maths,
and the concerns and open questions for each one. Measured outcomes across three
regimes are in **[results/RESULTS.md](results/RESULTS.md)**. Two knobs there decide whether
collapse is even possible in a given configuration (`--gate-norm` and
`--capacity-factor`); they are worth reading before interpreting any result.

## What is measured

**Quality** — validation cross-entropy and perplexity on a held-out slice, plus
the token drop rate when a capacity factor is set.

**Router collapse** — from the dispatch histogram, pooled and per layer:

| metric | reading |
|---|---|
| `H_norm` | load entropy / ln(E). `1.0` = perfectly balanced, `0` = fully collapsed |
| `effective_experts` | `exp(H)` — how many experts are *really* carrying load *(JSON only)* |
| `max_load` | busiest expert's share (uniform would be `1/E`) |
| `CV` | coefficient of variation of the load *(JSON only)* |
| `dead_experts` | experts below 10% of a fair share. Pooling over the model hides per-layer collapse, so the report gives the pooled count next to a per-layer breakdown. |
| `gini` | inequality of the load distribution *(JSON only)* |
| `drop_rate` | share of assignments refused for capacity (only with `--capacity-factor`) |

`effective_experts`, `CV` and `gini` are alternative summaries of the same
histogram as `H_norm`, so the report prints `H_norm` alone; all three are in
`comparison.json` for plotting.

**Compute** — analytic, so it is hardware- and implementation-independent. FLOPs are
`2 x MACs`, backward charged at `2x` forward. Expert cost is driven by the *measured*
number of `(token, expert)` evaluations, which is exactly what differs between a dense
pass (`E` per token) and a top-k pass (`k` per token) — so a strategy that routes some
or all tokens densely is priced as a real number rather than a hand-wave. Wall-clock is
recorded in `comparison.json` but kept out of the report, because it depends heavily
on the device and is the least trustworthy number here.

## Usage

```bash
python run_experiment.py --smoke                    # ~1 min sanity check
python run_experiment.py                            # the default comparison
python run_experiment.py --steps 6000 --seeds 0,1,2 # longer, averaged over seeds
python run_experiment.py --equalize steps           # match optimiser steps instead of tokens
python run_experiment.py --methods aux,explore      # a subset
python run_experiment.py --k 1 --capacity-factor 1.25 --methods none,aux,accum
                                                    # the regime where collapse
                                                    # bites; explore needs k>=2
python tests/test_moelab.py                         # correctness tests (no pytest needed)
```

Results land in `results/report.txt` (the human-readable table) and
`results/comparison.json` (every metric at every eval, for plotting).

Useful knobs: `--num-experts`, `--k`, `--gate-norm`, `--capacity-factor`,
`--d-model`, `--n-layer`, `--batch-size`, `--lr`, `--dense-warmup-frac` (0 =
sparse from step 0), `--eps-start`, `--gate-temp`, `--surrogate-alpha`, `--conf-start`,
`--tau-start/--tau-end`. `--help` lists all 23.

`d_hidden` is derived as `2 * d_model`; the optimiser settings, `aux_alpha`,
`accum_steps` and the epsilon anneal endpoints are constants in the source.
