# The strategies

What each anti-collapse strategy does, the maths behind it, and what is wrong or
unresolved about it. The README covers the experimental setup and how to run
things; this file is the mechanism.

- [Notation](#notation)
- [0. Control](#0-control--no-mitigation)
- [1. Auxiliary load-balancing loss](#1-auxiliary-load-balancing-loss-aux)
- [2. Epsilon-greedy exploration outside the top-k](#2-epsilon-greedy-exploration-outside-the-top-k-explore)
- [3. Gradient accumulation](#3-gradient-accumulation-accum)
- [4. Straight-through estimate of the unevaluated experts](#4-straight-through-estimate-of-the-unevaluated-experts-surrogate)
- [5. Dense fallback where the router is unsure](#5-dense-fallback-where-the-router-is-unsure-conf)
- [6. Threshold selection instead of top-k](#6-threshold-selection-instead-of-top-k-threshold)
- [Cross-cutting: the two knobs that decide whether collapse is possible](#cross-cutting-the-two-knobs-that-decide-whether-collapse-is-possible)

---

## Notation

A token arrives at the layer as $x \in \mathbb{R}^{d}$. A linear router scores
every expert, and a softmax turns the scores into a distribution:

$$\ell = W_r\,x \in \mathbb{R}^{E}, \qquad p = \operatorname{softmax}(\ell),
\qquad p_e = \frac{e^{\ell_e}}{\sum_{j=1}^{E} e^{\ell_j}}$$

A routing rule picks a set of experts $\mathcal{S}(x) \subseteq \{1,\dots,E\}$
— normally the top $k$ — and a gate $g_e$ weights each one's contribution:

$$y = \sum_{e \in \mathcal{S}(x)} g_e \, \mathrm{E}_e(x), \qquad
\mathrm{E}_e(x) = W_2^{(e)}\,\mathrm{GELU}\!\left(W_1^{(e)} x\right)$$

The gate comes in two forms (see [cross-cutting](#cross-cutting-the-two-knobs-that-decide-whether-collapse-is-possible)):

$$g_e^{\text{renorm}} = \frac{e^{\ell_e / T}}{\sum_{j \in \mathcal{S}} e^{\ell_j / T}}
\qquad\qquad g_e^{\text{raw}} = p_e$$

Over a batch of $N$ tokens, write $\mathcal{A}$ for the multiset of
$(\text{token}, \text{expert})$ assignments actually made, $|\mathcal{A}| = Nk$
for plain top-$k$. Two quantities drive the diagnostics:

$$f_i = \frac{1}{|\mathcal{A}|}\,\bigl|\{(t,e) \in \mathcal{A} : e = i\}\bigr|
\qquad\qquad \bar p_i = \frac{1}{N}\sum_{t=1}^{N} p_{t,i}$$

$f$ is the **dispatch load** (what the router did) and $\bar p$ is the **router
mass** (what the router believed). Both sum to $1$.

### The problem all six address

Top-$k$ routing is a positive feedback loop. An expert selected slightly more
often receives more gradient, improves, and is selected more often still. The
unselected experts receive **exactly zero** gradient — not a small amount,
structurally none — so they cannot catch up. Formally, for $j \notin \mathcal{S}(x)$
under a renormalised gate:

$$\frac{\partial \mathcal{L}}{\partial W^{(j)}} = 0
\qquad\text{and}\qquad \frac{\partial \mathcal{L}}{\partial \ell_j} = 0$$

The loss curve does not reveal this: a collapsed MoE trains perfectly well, it
just wastes most of its parameters. It has to be measured from the dispatch
histogram directly.

---

## 0. Control — no mitigation

Plain top-$k$ from step 0, $\mathcal{S}(x) = \operatorname{Top-}k(p)$, no extra
loss term. Present so the other five have a reference point; without it there is
no way to tell a strategy that works from a configuration in which collapse never
happened anyway.

---

## 1. Auxiliary load-balancing loss (`aux`)

The classical fix (Switch Transformer / ST-MoE). Add a term to the task loss that
penalises correlation between where tokens went and where the router's belief was:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \underbrace{\alpha\,E \sum_{i=1}^{E} f_i\,\bar p_i}_{\mathcal{L}_{\text{aux}}}$$

Because $f$ and $\bar p$ are both distributions over $E$ experts, the sum is a
dot product between two points on the simplex. It is minimised when both are
uniform and maximised when both concentrate on one expert:

$$\mathcal{L}_{\text{aux}} = \alpha\,E \sum_i f_i \bar p_i \in [\alpha,\ \alpha E]$$

with $\alpha$ at perfect balance ($f_i = \bar p_i = 1/E$) and $\alpha E$ at total
collapse. The gradient pushes $\bar p_i$ down for experts that are already
overloaded, which is the direct counter-force to rich-get-richer.

Default $\alpha = 10^{-2}$ (`AUX_ALPHA` in `moelab/strategies.py`).

### Concerns

- **It optimises a proxy, and the proxy has a trivial solution.** The term is
  minimised by *any* uniform dispatch, including uniform-at-random. A router that
  has learned nothing and assigns tokens by coin flip scores a perfect
  $\mathcal{L}_{\text{aux}} = \alpha$, exactly like one that has learned a
  meaningful partition. None of the collapse metrics can tell those apart either
  — see the note on measurement in the README.
- **It is the only strategy that changes the objective.** $\alpha$ is a real
  trade-off against task loss, and it has never been tuned here. Too large and
  you buy balance with quality; too small and it does nothing.
- **Balance is not the same as good routing.** An $H_{\text{norm}}$ of $0.999$ is
  suspicious precisely because a term that explicitly rewards uniformity will
  happily purchase uniformity by making routing less informative.

---

## 2. Epsilon-greedy exploration outside the top-k (`explore`)

No auxiliary loss. Force under-used experts to be tried, so the router keeps
receiving evidence about them.

**Selection.** The decision is per token, all-or-nothing. With probability
$\epsilon$ the token abandons its entire top-$k$ and is routed to $k$ *distinct*
experts drawn uniformly from the complement:

$$\mathcal{S}(x) = \begin{cases}
\operatorname{Top-}k(p) & \text{with probability } 1-\epsilon \\[4pt]
\text{a uniform } k\text{-subset of } \{1,\dots,E\} \setminus \operatorname{Top-}k(p)
& \text{with probability } \epsilon
\end{cases}$$

Drawing from the complement rather than from all $E$ is what makes the sample
informative: a draw landing back on an already-chosen expert teaches the router
nothing it did not know. This requires $E \ge 2k$, and $k \ge 2$ so that a trial
is something the token does *in addition* to being served rather than instead of it.

$\epsilon$ anneals linearly over a fraction of training:

$$\epsilon(s) = \epsilon_0 + \min\!\left(1, \tfrac{s}{\rho S}\right)\left(\epsilon_1 - \epsilon_0\right),
\qquad \epsilon_0 = 0.6,\ \epsilon_1 = 0.01,\ \rho = 0.6$$

**Gate temperature.** Epsilon-greedy alone does not work, and this is the
interesting part. An expert the router has written off has $p_j \approx 0$, so its
gate is $\approx 0$: its output never reaches the residual stream, and the softmax
is saturated so the gradient it sends back is $\approx 0$ too. The token is spent
and nothing is learned.

$$\frac{\partial \mathcal{L}}{\partial \ell_e}
= \underbrace{\left\langle \frac{\partial \mathcal{L}}{\partial y},\, \mathrm{E}_e(x)\right\rangle}_{\text{how useful expert } e \text{ was}}
\cdot \underbrace{\frac{1}{T}\,g_e\,(1 - g_e)}_{\text{vanishes as } g_e \to 0}$$

Dividing the selected logits by $T > 1$ desaturates the softmax. Crucially the
temperature is applied **per token, only to tokens exploration took over**:

$$T_t = \begin{cases} T & \text{if token } t \text{ explored} \\ 1 & \text{otherwise}\end{cases}
\qquad g_e = \frac{e^{\ell_e / T_t}}{\sum_{j \in \mathcal{S}} e^{\ell_j / T_t}}$$

Because an exploring token replaces *all* $k$ slots, it has no exploited slot left
in its gate softmax, so tempered and untempered logits are never mixed inside one
normaliser. Every exploited token is gated at $T = 1$ — exactly how the model is
evaluated — so the temperature opens no train/deploy gap and therefore never has
to be annealed. Default $T = 4$, held constant.

For scale: with logits $[4, -16]$, the written-off expert's gate goes from
$2.1 \times 10^{-9}$ at $T=1$ to $7.6 \times 10^{-2}$ at $T=8$, and
$\partial g / \partial \ell$ from $2.1 \times 10^{-9}$ to $8.8 \times 10^{-3}$.

### Concerns

- **The signal decays linearly in $p_j$.** The gate of an explored expert is
  roughly $\propto e^{\ell_j/T}$, so the experts most in need of rescue are still
  the ones the mechanism says least about — the temperature widens the window, it
  does not remove the asymmetry.
- **Exploration is substitution, not addition.** At $\epsilon = 1$ the top-$k$
  receives *nothing*, so a full-exploration pass inverts the load rather than
  flattening it. Flattening comes from mixing the two populations, so $\epsilon$
  must stay meaningfully below 1.
- **Nothing holds the router open once $\epsilon$ anneals.** The mechanism has no
  effect at $\epsilon = 0.01$, so a long run has time to drift back. Raising
  `--eps-end` is the obvious thing to try, but note the schedule anneals
  $\epsilon$ only — the temperature is already constant.
- **A uniform gate *floor* would not work**, and this is why the fix is a
  temperature. A floor added after the softmax is a constant with respect to
  $\ell$, so $\partial g / \partial \ell = 0$: it revives the expert's weights but
  leaves the router's preference frozen.

---

## 3. Gradient accumulation (`accum`)

No auxiliary loss. Accumulate gradients over $M$ micro-batches before each
optimiser step:

$$\theta \leftarrow \theta - \eta\,\nabla_\theta \left(\frac{1}{M}\sum_{m=1}^{M} \mathcal{L}_m\right)$$

The claim is about variance, not bias. A router update computed from one small
batch reinforces whatever that batch happened to prefer; averaging over $M$ times
as many tokens shrinks the variance of the router gradient by $\approx 1/M$,
which damps the feedback loop that turns a small accidental asymmetry into a
permanent one. Default $M = 8$ (`ACCUM_STEPS`).

### Concerns

- **The budget question decides the answer.** On a fixed *token* budget it takes
  $1/M$ as many optimiser steps as everyone else, so its results measure
  undertraining rather than routing. On a fixed *step* budget (`--equalize steps`)
  it consumes $M\times$ the tokens and FLOPs. Neither reading is wrong, and they
  disagree — which is why both exist.
- **Its advantage is per update, which is the right currency only if updates
  rather than FLOPs are what you are short of.**
- **It is the one strategy with no MoE-specific machinery at all**, so any effect
  is a generic variance-reduction effect, not something about routing.

---

## 4. Straight-through estimate of the unevaluated experts (`surrogate`)

No auxiliary loss, no exploration. Give the router gradient for the $E - k$
experts it never evaluated, at zero extra expert cost, by *estimating* what they
would have produced.

**The model.** Assume an expert's output scales with the probability the router
gave it, $\mathrm{E}_e(x) \approx c\,p_e$, with one slope $c$ per token fitted by
least squares through the origin over the $k$ experts that actually ran:

$$c = \arg\min_{c} \sum_{e \in \mathcal{S}} \bigl(\mathrm{E}_e(x) - c\,p_e\bigr)^2
= \frac{\sum_{e \in \mathcal{S}} p_e\,\mathrm{E}_e(x)}{\sum_{e \in \mathcal{S}} p_e^{2}}$$

The term top-$k$ omitted from the full dense sum is then estimated as

$$\sum_{j \notin \mathcal{S}} p_j\,\mathrm{E}_j(x) \;\approx\; c \sum_{j \notin \mathcal{S}} p_j^{2}$$

**The shortcut.** The numerator of $c$ never has to be materialised. The
dispatched output is $y = \sum_{\mathcal{S}} g_e \mathrm{E}_e(x)$, and with no
exploration at $T = 1$ the gate is proportional to $p_e$, so
$\sum_{\mathcal{S}} p_e \mathrm{E}_e(x) = \kappa\,y$ with

$$\kappa = \begin{cases} 1 & \text{raw gate} \\ \sum_{e \in \mathcal{S}} p_e & \text{renormalised gate}\end{cases}$$

The whole correction therefore collapses to a per-token **scalar** times $y$:

$$s = \kappa\,\frac{\sum_{j \notin \mathcal{S}} p_j^{2}}{\sum_{e \in \mathcal{S}} p_e^{2}}$$

**Straight-through.** The correction is wired in so that it contributes exactly
zero to the forward value but a real gradient to every router logit:

$$\hat y = y + \alpha\,\operatorname{sg}(y)\,\bigl(s - \operatorname{sg}(s)\bigr)$$

where $\operatorname{sg}$ is stop-gradient. Forward, $s - \operatorname{sg}(s) = 0$
so $\hat y = y$ exactly and the deployed model is unchanged. Backward,
$\partial \mathcal{L}/\partial s = \langle \partial\mathcal{L}/\partial y,\, \operatorname{sg}(y)\rangle$,
and $s$ depends on the full probability vector, so $\partial s / \partial \ell_j \neq 0$
for every $j$ including the unevaluated.

### Concerns

- **The gradient decays as $p_j^{2}$.** The estimate is proportional to $p_j$ and
  is then weighted by $p_j$ again to enter the sum. Measured, $\text{grad}/p_j^2$
  is flat to within 10% across four orders of magnitude of $p_j$. This is the same
  self-defeating shape as strategy 2, **one power worse**: the experts most in
  need of rescue are quadratically the ones it says least about. Expect it to slow
  a drift toward collapse rather than reverse one.
- **It is a balance pressure, not credit assignment.** Every skipped expert is
  estimated as a *rescaling of the experts that ran*, so the surrogate is
  collinear with $y$. The router learns how much total mass belongs off its
  top-$k$; it cannot learn *which* skipped expert deserves it, because the
  estimated direction is identical for all of them.
- **It reaches the router, not the starving experts.** $\hat{\mathrm{E}}_j$ is
  built from other experts' outputs, so expert $j$'s own weights still receive
  nothing. Recovery is indirect: the router must rebalance first.
- **Two incompatibilities follow from the derivation.** It must not be combined
  with exploration or a gate temperature, both of which break $g_e \propto p_e$.
  A capacity factor biases it, because dropped assignments are missing from $y$
  while their probabilities still appear in the fit.
- **The proportionality assumption is asserted, not verified.** Nothing checks
  that $\mathrm{E}_e(x) \approx c\,p_e$ holds; the residual of the fit is never
  looked at.

---

## 5. Dense fallback where the router is unsure (`conf`)

No auxiliary loss, no exploration. Commit to the top-$k$ only where the router is
confident enough to have earned it.

Confidence is the combined mass of the chosen set. Below a threshold $\tau_c$ the
token is passed through **every** expert, weighted by the full router
distribution exactly as a dense pass would:

$$\mathcal{S}(x) = \begin{cases}
\{1,\dots,E\} & \text{if } \sum_{e \in \operatorname{Top-}k(p)} p_e < \tau_c \\[4pt]
\operatorname{Top-}k(p) & \text{otherwise}
\end{cases}$$

$$\tau_c(s) = \tau_c^{0}\left(1 - \min\!\left(1, \tfrac{s}{0.8\,S}\right)\right),
\qquad \tau_c^{0} = 0.8$$

The threshold anneals to exactly $0$, so the last fifth of training — and all of
evaluation — is ordinary top-$k$.

**Useful range $(k/E,\ 1]$.** The top $k$ of any distribution hold at least a
uniform $k/E$ share, so a threshold at or below that can never fire.

**The idea.** Collapse is driven by *forced guesses*: when the router cannot
discriminate, top-$k$ still makes it commit to whatever is marginally ahead, and
that arbitrary commitment is what the feedback loop amplifies. Routing densely
instead means no guess is reinforced, and every expert receives gradient on that
token. It is a dense warm-up in which tokens graduate **individually**, as the
router earns confidence about each, rather than all at once on a fixed schedule.

### Concerns

- **Confidence is not correctness, and this is the catch.** A *collapsed* router
  is a very confident one — that is what collapse looks like from the inside. A
  router that has already collapsed sails past the threshold and never triggers
  the fallback. The mechanism is prophylactic, not curative: it can stop early
  commitment, it cannot recover a router that committed wrongly and is sure of it.
- **It is the most expensive mechanism here**, and the cost is front-loaded
  exactly where uncertainty is highest. A dense token costs $E/k$ times a sparse
  one — $4\times$ at $E = 8$, $k = 2$.
- **It flatters the training-time load histogram.** Dense tokens add an equal
  share to every expert, so $f$ looks more uniform than the routing decision
  really is. The reported collapse metrics come from evaluation, which is pure
  top-$k$, so they are unaffected — but $f$ also feeds the aux loss, so the two
  should not be combined without thought.

---

## 6. Threshold selection instead of top-k (`threshold`)

No auxiliary loss, no exploration. Replace the fixed-$k$ rule entirely: select
*every* expert whose probability clears a bar, so the count varies per token.

$$\mathcal{S}(x) = \Bigl\{\,e : p_e > \tfrac{\tau}{E} \Bigr\} \cup \Bigl\{\arg\max_e p_e\Bigr\}$$

The threshold is expressed in **units of the uniform share $1/E$**, so it means
the same thing at any $E$: $\tau = 1$ is "beat an even split". The argmax is
always retained, so the count floors at 1 rather than 0. It **rises** over
training, pruning experts the router has earned the right to dismiss:

$$\tau(s) = \tau_0 + \min\!\left(1, \tfrac{s}{0.8\,S}\right)(\tau_1 - \tau_0),
\qquad \tau_0 = 1.0,\ \tau_1 = 1.6$$

**Calibration.** Those defaults come from measuring what the router actually does.
After training a while it places roughly $0.39$ on its best expert, $0.17$ on its
second and $0.12$ on its third, so the band that discriminates is narrow and sits
just above uniform:

| $\tau$ (units of $1/E$) | 1.0 | 1.2 | 1.6 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| mean experts selected | 2.21 | 1.69 | 1.25 | 0.98 | 0.16 |
| tokens with none above it | 0% | 0% | 1% | 9% | 84% |

An *absolute* threshold near $0.5$ cannot work: since $\sum_e p_e = 1$, at most one
expert can ever exceed one half, and the router's top probability plateaus below
$0.4$.

**The per-token gate fallback.** Renormalising over a *single* selected expert
gives the constant $1$, whose derivative with respect to $\ell$ is zero — the same
degeneracy `--gate-norm auto` exists to avoid at $k = 1$. Here the count is per
token, so the fallback must be too:

$$g_e = \begin{cases}
p_e & \text{if } |\mathcal{S}(x)| = 1 \\[4pt]
\dfrac{e^{\ell_e}}{\sum_{j \in \mathcal{S}} e^{\ell_j}} & \text{otherwise}
\end{cases}$$

Without it the router would freeze on exactly the tokens it is most confident
about, and a rising threshold makes those the majority.

### Concerns

- **This is a sharpening rule, not a balancing one — it may make collapse
  worse.** Pruning experts below a *rising* bar is the rich-get-richer loop with
  an explicit ratchet: an expert that falls behind clears the bar for fewer
  tokens, receives less gradient, and falls further behind. It prunes per token,
  but nothing stops it pruning an expert globally. Of the six this is the one
  whose mechanism could plausibly worsen the thing it is meant to fix.
- **It is the only strategy whose trained artefact differs.** The threshold is
  the routing rule, not a training aid, so it stays in force at evaluation —
  scoring it under top-$k$ would measure a rule it never trained on. Its inference
  cost is therefore variable per token rather than exactly $k$, and the
  `experts/tok` column in the report exists to make that visible. Quality and
  collapse remain comparable across strategies; the compute axis, for this row
  only, does not.
- **It answers a different question.** If what you want to know is "which
  training procedure produces the best top-$k$ model", this is not a candidate —
  it is a different architecture that happens to sit in the same table.
- **The schedule is a guess.** $\tau$ rising linearly over 80% of training is not
  derived from anything; the calibration table fixes the *range*, not the path.

---

## Cross-cutting: the two knobs that decide whether collapse is possible

Both default to the safe choice, but getting them wrong makes the whole
comparison look like a tie. They were found by probing.

### `--gate-norm`

How the gate is formed from the router logits.

$$g_e^{\text{renorm}} = \frac{e^{\ell_e}}{\sum_{j \in \mathcal{S}} e^{\ell_j}}
\qquad\qquad g_e^{\text{raw}} = p_e = \frac{e^{\ell_e}}{\sum_{j=1}^{E} e^{\ell_j}}$$

`renorm` (GShard/Mixtral) normalises over the *selected* logits, so the $k$ gates
sum to $1$ and only **relative** preference among the chosen matters. `raw`
(Switch) uses the router's own probability, so **absolute** confidence scales the
expert's contribution. This is the difference between a router that can and cannot
collapse: `renorm` removes much of the rich-get-richer pressure by construction.

At $k = 1$, `renorm` is a softmax over a *single* number — the constant $1.0$ —
so the router receives **no task gradient at all** and never trains. A frozen
router looks perfectly balanced, which is very easy to mistake for a strategy
working. `auto` (the default) uses `raw` for $k = 1$ and `renorm` above it;
`test_renormalised_gate_is_degenerate_at_k_equals_one` guards it, and strategy 6
extends the same fallback to a per-token basis.

### `--capacity-factor`

A per-expert budget, with overflow dropped (Switch-style, lowest gate first):

$$C = \left\lfloor \gamma \cdot \frac{|\mathcal{A}|}{E} \right\rfloor$$

where $\gamma$ is the capacity factor and $|\mathcal{A}|$ the assignments actually
made — $Nk$ for plain top-$k$, larger when a strategy routes some tokens densely.
$\gamma = 1.25$ is a typical production value.

Unset means no cap, and **with no cap an imbalanced router costs wasted parameters
but not accuracy** — every token still reaches its top-$k$ experts, so collapse
and quality are effectively independent axes. Setting it is what couples them: a
collapsed router overflows its favourites and drops real tokens. Dropped
assignments cost no FLOPs, and the collapse metrics keep reporting what the router
*wanted*, not what capacity allowed.
