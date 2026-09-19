"""The strategies under comparison, as schedules over the routing controls.

Every strategy shares one architecture, one initialisation and one data order;
the only thing that varies is how the router is kept from collapsing. See the
README for what each one does and what it costs.

``Strategy.phase`` is a pure function of (micro step, total micro steps) and is
the whole abstraction: a strategy is a trajectory through (routing mode,
exploration rate, gate temperature), plus the static weights applied outside the
layer. Adding a strategy means adding an entry below, not a new code path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Strategy:
    name: str
    label: str
    description: str
    aux_alpha: float = 0.0
    accum_steps: int = 1
    dense_warmup_frac: float = 0.0
    eps_start: float = 0.0
    eps_end: float = 0.0
    eps_decay_frac: float = 1.0  # share of post-warmup steps over which eps anneals
    # Weight on the straight-through estimate of the unevaluated experts. Static
    # rather than scheduled: it costs nothing and has nothing to anneal away,
    # since it never changes a forward value.
    surrogate_alpha: float = 0.0
    # Gate temperature for explored tokens. Constant: it never touches an
    # exploited token, so there is no train/deploy gap for annealing to close.
    gate_temp: float = 1.0
    # Top-k probability mass below which a token is routed densely instead.
    # Annealed to 0, so the last stretch of training is ordinary top-k.
    conf_start: float = 0.0
    # Selection threshold in units of the uniform share 1/E, rising start->end.
    # 0 leaves selection as plain top-k.
    tau_start: float = 0.0
    tau_end: float = 0.0

    def phase(self, micro_step: int, total_micro: int) -> Phase:
        """The routing controls for one micro step."""
        warmup = int(self.dense_warmup_frac * total_micro)
        if micro_step < warmup:
            return Phase(mode="dense")
        conf = 0.0
        if self.conf_start > 0.0:
            span = max(1, int(CONF_DECAY_FRAC * (total_micro - warmup)))
            t = min(1.0, (micro_step - warmup) / span)
            conf = self.conf_start * (1.0 - t)  # reaches 0: pure top-k at the end
        tau = 0.0
        if self.tau_start > 0.0:
            span = max(1, int(TAU_RISE_FRAC * (total_micro - warmup)))
            t = min(1.0, (micro_step - warmup) / span)
            tau = self.tau_start + t * (self.tau_end - self.tau_start)
        if self.eps_start <= 0.0:
            return Phase(conf_threshold=conf, tau_mult=tau)
        span = max(1, int(self.eps_decay_frac * (total_micro - warmup)))
        t = min(1.0, (micro_step - warmup) / span)
        eps = self.eps_start + t * (self.eps_end - self.eps_start)
        # The temperature is held constant. It only ever applies to tokens that
        # exploration took over, and those disappear on their own as epsilon
        # anneals, so there is nothing left over at the end of training either
        # way -- and no need to distort the exploited gate on the way there.
        return Phase(epsilon=eps, gate_temp=self.gate_temp, conf_threshold=conf,
                     tau_mult=tau)


AUX_ALPHA = 1e-2  # Switch/ST-MoE load-balancing weight
ACCUM_STEPS = 8  # micro-batches per optimiser step for the accumulation strategy
EPS_END = 0.01  # exploration rate at the end of its anneal
EPS_DECAY_FRAC = 0.60  # share of post-warmup steps the anneal spans
CONF_DECAY_FRAC = 0.80  # share of training the confidence threshold anneals over
TAU_RISE_FRAC = 0.80  # share of training the selection threshold rises over


@dataclass(frozen=True)
class Phase:
    """The routing controls for one micro step, as `set_routing` takes them."""

    mode: str = "topk"
    epsilon: float = 0.0
    gate_temp: float = 1.0
    conf_threshold: float = 0.0
    tau_mult: float = 0.0


def build_strategies(
    dense_warmup_frac: float = 0.0,
    eps_start: float = 0.60,
    gate_temp: float = 4.0,
    surrogate_alpha: float = 1.0,
    conf_start: float = 0.80,
    tau_start: float = 1.0,
    tau_end: float = 1.6,
) -> dict[str, Strategy]:
    return {
        "aux": Strategy(
            name="aux",
            label="1. Auxiliary load-balancing loss",
            description=(
                f"top-k from step 0, aux loss alpha={AUX_ALPHA} added to the task loss"
            ),
            aux_alpha=AUX_ALPHA,
        ),
        "explore": Strategy(
            name="explore",
            label="2. Epsilon-greedy exploration outside the top-k",
            description=(
                "no aux loss; "
                + (f"dense routing for the first {dense_warmup_frac:.0%} of training, "
                   "then " if dense_warmup_frac > 0 else "sparse top-k from step 0, ")
                + f"epsilon annealed {eps_start}->{EPS_END} over "
                f"{EPS_DECAY_FRAC:.0%} of the remaining steps; an exploring token is "
                f"routed to k distinct experts drawn from outside its top-k and gated "
                f"at a fixed temperature {gate_temp:g}"
            ),
            dense_warmup_frac=dense_warmup_frac,
            eps_start=eps_start,
            eps_end=EPS_END,
            eps_decay_frac=EPS_DECAY_FRAC,
            gate_temp=gate_temp,
        ),
        "surrogate": Strategy(
            name="surrogate",
            label="4. Straight-through estimate of the unevaluated experts",
            description=(
                "no aux loss, no exploration; the E-k experts top-k skipped are "
                "estimated as c*p_j from a least-squares fit of output against "
                f"probability over the k that ran, wired in straight-through "
                f"(alpha={surrogate_alpha}) so the forward pass stays exactly top-k "
                "while every router logit receives gradient"
            ),
            surrogate_alpha=surrogate_alpha,
        ),
        "conf": Strategy(
            name="conf",
            label="5. Dense fallback where the router is not confident",
            description=(
                "no aux loss, no exploration; a token whose top-k experts hold "
                "less than the threshold of the router's probability mass goes "
                f"through every expert instead, threshold {conf_start:g}->0 over "
                f"{CONF_DECAY_FRAC:.0%} of training"
            ),
            conf_start=conf_start,
        ),
        "threshold": Strategy(
            name="threshold",
            label="6. Threshold selection instead of top-k",
            description=(
                "no aux loss, no exploration; every expert above "
                f"{tau_start:g}->{tau_end:g} uniform shares (1/E) is selected, so the "
                f"count varies per token, rising over {TAU_RISE_FRAC:.0%} of training. "
                "Unlike the others this is the routing rule at evaluation too"
            ),
            tau_start=tau_start,
            tau_end=tau_end,
        ),
        "accum": Strategy(
            name="accum",
            label=f"3. Gradient accumulation x{ACCUM_STEPS}",
            description=(
                f"no aux loss; gradients accumulated over {ACCUM_STEPS} micro-batches "
                "before each optimiser step"
            ),
            accum_steps=ACCUM_STEPS,
        ),
        "none": Strategy(
            name="none",
            label="0. Control: nothing (top-k, no aux loss)",
            description="sparse top-k from step 0 with no collapse mitigation at all",
        ),
    }
