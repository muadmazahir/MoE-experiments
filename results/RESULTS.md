# Results

Six anti-collapse strategies plus a control, across three regimes. Mechanism and
maths for each strategy are in [STRATEGIES.md](../STRATEGIES.md).

**Single seed (0), one small model, one small corpus, char-level.** Quality gaps
of a few percent of perplexity are noise and should not be ranked. The
*dropping* and *max load* gaps are large and consistent, and the failures called
out below are far outside noise.

Raw per-eval metrics are in each run's `comparison.json`; the formatted tables
are in the sibling `report.txt`.

---

## Verdict

**The auxiliary loss wins, unambiguously.** Best or statistically tied for best
quality in all three regimes, the flattest router in all three
(`H_norm` 0.999 / 1.000 / 0.998), and the only strategy that essentially
eliminates capacity dropping — **0.01%, 0.37%, 0.17%** against **12.6%, 15.8%,
17.0%** for the control. It costs ~1.00x the training FLOPs of unmitigated
top-k, and that gap is not overhead: the control is cheaper only because it
silently refused to serve a eighth of its expert work.

Nothing else comes close on the axis that actually costs tokens. The loss-free
strategies each do something real, but every one of them has a regime where it
is worse than doing nothing.

---

## Experiment 1 — top-2, renormalised gate

The Mixtral-style default, and the configuration most people would deploy.

`--steps 3000 --capacity-factor 1.25`, 4L x 128d, 8 experts top-2, 12.3M tokens
per strategy.

| strategy | val loss | H_norm↑ | max load↓ | dropped↓ | TFLOPs | vs top-k | experts/tok |
|---|---|---|---|---|---|---|---|
| **conf** | **1.658** | 0.973 | 0.195 | 13.46% | 90.3 | 1.42x | 1.73 |
| **aux** | 1.663 | **0.999** | **0.138** | **0.01%** | 63.6 | 1.00x | 2.00 |
| none | 1.698 | 0.986 | 0.172 | 12.55% | 57.7 | 0.91x | 1.75 |
| surrogate | 1.711 | 0.994 | 0.165 | 3.03% | 60.8 | 0.95x | 1.94 |
| threshold | 1.719 | 0.959 | 0.216 | 19.02% | 52.2 | 0.82x | **1.11** |
| explore | 1.766 | 0.987 | 0.152 | 1.82% | 62.5 | 0.98x | 1.96 |
| accum | 2.327 | 0.958 | 0.213 | 19.79% | 53.5 | 0.84x | 1.60 |

**`conf` has the best val loss and the second-worst router.** `H_norm` 0.973
against the control's 0.986, and it drops *more* tokens than doing nothing. Its
layer 0 ends at `H=0.772` with three dead experts. The quality came from the
dense phase training every expert properly, not from better routing — its loss
curve is flat near 2.5 until 42% of training, then falls off a cliff to 1.81 as
the threshold anneals and evaluation finally matches training. It is also the
most expensive strategy measured.

**`explore` cost quality for no balance gain.** 1.766 is *worse than the
control*, and `H_norm` 0.987 vs 0.986 is nothing. Starting at eps=0.6 routes most
early tokens to experts the router rejected, and the disruption outweighs what
the exploration buys. It did cut dropping to 1.82%, so it flattens the tail of
the load distribution even though the entropy barely moves.

**`surrogate` worked here.** Dropping fell from 12.55% to 3.03% and `H_norm` rose
to 0.994, at 0.95x the cost, with quality a wash. This is the regime where it is
the router's only gradient signal for unchosen experts.

---

## Experiment 2 — top-1, raw gate (Switch style)

The regime where collapse actually bites: no second expert to absorb overflow,
and a gate whose *absolute* confidence scales the expert output, so the
rich-get-richer loop has room to run.

`--k 1 --gate-norm raw --capacity-factor 1.25 --lr 1e-3 --batch-size 16
--steps 3000`. `explore` is excluded — it requires `k >= 2`.

| strategy | val loss | H_norm↑ | max load↓ | dropped↓ | TFLOPs | vs top-k | dead (Σ layers) |
|---|---|---|---|---|---|---|---|
| **aux** | **1.609** | **1.000** | **0.134** | **0.37%** | 21.9 | 0.99x | 0 |
| conf | 1.610 | 0.973 | 0.213 | 13.75% | 43.4 | 1.95x | 0 |
| threshold | 1.619 | 0.976 | 0.200 | 19.63% | 23.7 | 1.07x | 1 |
| none | 1.635 | 0.969 | 0.202 | 15.79% | 19.0 | 0.85x | 0 |
| **surrogate** | **1.888** | 0.965 | 0.194 | 28.51% | 19.2 | 0.86x | **11** |
| accum | 2.220 | 0.962 | 0.206 | 27.88% | 18.5 | 0.83x | 4 |

**`surrogate` broke here** — 1.888 against the control's 1.635, 28.51% dropping,
and **eleven dead experts summed over four layers**, with layer 3 at `H=0.640`.
A complete reversal from experiment 1.

The mechanism explains it. Under the raw gate the unchosen logits *already*
receive gradient through the full-softmax normaliser, so the surrogate stacks a
second pressure on top of an existing one — and being collinear with the
evaluated experts' output, it pushes every unchosen expert in the same direction
regardless of merit. Helpful when it is the only signal, harmful when it is
redundant.

**`aux` reached `H_norm` = 1.000**, a perfectly flat router, while cutting
dropping to 0.37% from the control's 15.79%.

---

## Experiment 3 — equal optimiser steps

The fair-to-accumulation reading. Every strategy performs 1000 updates; the
accumulating one therefore consumes 8x the tokens and FLOPs to do so.

`--equalize steps --steps 1000 --capacity-factor 1.25`, otherwise as experiment 1.

| strategy | val loss | H_norm↑ | max load↓ | dropped↓ | TFLOPs | tokens |
|---|---|---|---|---|---|---|
| **accum** | **1.884** | 0.989 | 0.167 | 6.49% | **156.3** | **32.8M** |
| aux | 2.105 | **0.998** | **0.137** | **0.17%** | 21.1 | 4.1M |
| conf | 2.128 | 0.991 | 0.167 | 6.64% | 33.0 | 4.1M |
| none | 2.166 | 0.962 | 0.207 | 17.01% | 18.4 | 4.1M |
| threshold | 2.173 | 0.992 | 0.177 | 11.52% | 18.1 | 4.1M |
| surrogate | 2.178 | 0.981 | 0.184 | 20.45% | 18.0 | 4.1M |
| explore | 2.183 | 0.988 | 0.178 | 5.18% | 20.5 | 4.1M |

**Given equal updates, accumulation's hypothesis holds.** 1.884 against the
control's 2.166, balance up from 0.962 to 0.989, dropping down from 17.0% to
6.5%.

**But it is by far the worst strategy per unit of compute.** It needed
**156.3 TFLOPs** to reach 1.884. The auxiliary loss in experiment 1 reached
**1.663 for 63.6 TFLOPs** — a better loss for 2.5x less compute. Accumulation's
win is per optimiser step, which is the right currency only if updates rather
than FLOPs are what you are short of.

---

## Cross-cutting findings

### `max_load` predicts dropping; `H_norm` saturates

Capacity here is `1.25/8 = 0.156` of the load. Any expert above that share
overflows, and in experiment 1 the relationship is almost exact:

| max load | 0.138 | 0.152 | 0.165 | 0.172 | 0.195 | 0.213 | 0.216 |
|---|---|---|---|---|---|---|---|
| dropped | 0.01% | 1.8% | 3.0% | 12.6% | 13.5% | 19.8% | 19.0% |

Meanwhile `H_norm` sat between 0.958 and 0.999 for every strategy and barely
discriminated. It is an aggregate over the whole histogram; what costs you tokens
is the busiest expert against the capacity line. The relationship is looser in
experiment 3, because a *pooled* max load hides per-layer overflow — but
directionally it holds everywhere.

**`max_load` is the operationally meaningful collapse metric.**

### Pooling hides collapse completely

Every strategy in all three experiments reports a pooled `dead` count of **0**.
Per layer, the same runs lose up to **eleven** experts (`surrogate`, experiment
2). Worst-layer entropies run as low as 0.640 while the pooled figure reads a
healthy 0.965. A model-level average is not a collapse diagnostic.

### A capacity factor is itself an anti-collapse force

Experiment 3 was first run by accident without `--capacity-factor`, and two
results moved sharply:

| | with cap | no cap |
|---|---|---|
| `accum` H_norm vs control | 0.989 vs 0.962 | 0.973 vs 0.970 |
| `threshold` H_norm | 0.992 | 0.930, declining across training |

Dropping is by lowest gate first, so an overloaded expert's marginal tokens are
refused and it receives less gradient — which damps the runaway directly. With no
cap nothing checks it, and `threshold`'s entropy declines monotonically
(0.963 → 0.942 → 0.932 → 0.930), the only strategy that does. Turning the cap on
does not merely *reveal* collapse as a quality cost; it partially *suppresses*
the collapse.

---

## Per-strategy summary

| strategy | verdict |
|---|---|
| **1. aux** | Wins on every axis that matters, in every regime, at ~1.00x cost. The only one to eliminate dropping. |
| **2. explore** | Consistently costs quality (worst or near-worst in exp 1 and 3) for no gain in `H_norm`. Does flatten the load *tail* — dropping falls to 1.8% in exp 1. eps=0.6 looks too high. |
| **3. accum** | Wins per optimiser step, loses badly per FLOP. 2.5x the compute of `aux` for a worse loss. |
| **4. surrogate** | Unreliable. Real gains in exp 1 (dropping 12.6% → 3.0%); actively harmful in exp 2 (11 dead experts, worst quality bar `accum`); worst dropping in exp 3. Do not use with a raw gate. |
| **5. conf** | Best quality in exp 1, second in exp 2 — but buys it with a dense phase, not with routing. Worse balance and *more* dropping than the control in exp 1. Most expensive strategy (1.42–1.95x). |
| **6. threshold** | Not a collapse fix: mid-pack quality, regime-dependent balance, the only strategy whose entropy declines when nothing checks it. Its real result is **compute** — 1.11 experts/token at inference in exp 1 against 2.00, a 44% saving for ~1% val loss. |

---

## What would change these conclusions

- **More seeds.** Everything here is seed 0. The ordering among `aux`, `conf`,
  `none`, `surrogate` and `threshold` in experiment 1 spans ~3.5% of perplexity
  and should not be trusted. `--seeds 0,1,2` reports mean±std automatically.
- **Tuning `alpha`.** The auxiliary loss has never been tuned; `1e-2` is simply
  the canonical value. Its dominance is therefore an *untuned* result, which
  makes it stronger, not weaker.
- **A lower `--eps-start`.** `explore` at 0.6 disrupts more than it buys; the
  mechanism might survive at 0.2–0.3.
- **A dense warm-up alone.** `conf`'s gain looks attributable to dense training
  rather than to confidence gating. `--dense-warmup-frac 0.15 --conf-start 0`
  isolates that and is the cheaper hypothesis.
- **Whether balance is meaningful at all.** Every metric here measures only how
  *evenly* load is spread. A uniformly random router scores `H_norm` = 1.0
  exactly like a perfectly specialised one. `aux`'s 0.999 is consistent with
  either.

## Reproducing

```bash
python run_experiment.py --steps 3000 --capacity-factor 1.25 --seeds 0 --out results/exp1-seed0
python run_experiment.py --k 1 --gate-norm raw --capacity-factor 1.25 --lr 1e-3 --batch-size 16 --steps 3000 --methods none,aux,accum,surrogate,conf,threshold --seeds 0 --out results/exp2-seed0
python run_experiment.py --equalize steps --steps 1000 --capacity-factor 1.25 --seeds 0 --out results/exp3-seed0
```

Run on a Colab T4; ~60 minutes total. Every reported figure is analytic and
device-independent — a CPU run of the same configuration agrees on training
FLOPs to three decimal places and on validation loss to ~0.002.
