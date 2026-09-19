"""Sparse MoE feed-forward layer with pluggable routing.

One layer serves every strategy; what differs is how the trainer drives it each
step via ``MoETransformer.set_routing``. The README explains what each routing
control is for and why; this module documents only how it is implemented.

  ``mode``              "topk" or "dense" (every expert on every token).
  ``epsilon``           exploration rate -- see ``_explore``.
  ``gate_temperature``  softens the gate of explored tokens -- see ``forward``.
  ``surrogate_alpha``   straight-through estimate of the experts top-k skipped,
                        see ``_surrogate_scale``.
  ``conf_threshold``    tokens whose top-k probability mass falls below this go
                        through every expert instead -- see ``forward``.
  ``tau_mult``          replaces top-k selection entirely with "every expert
                        above a probability threshold" -- see
                        ``_threshold_assignments``. Unlike the controls above it
                        is a routing rule rather than a training aid, so it
                        stays in force at evaluation.
  ``aux_alpha``         load-balancing loss weight; 0 switches it off.
  ``gate_norm``         "renorm" (GShard/Mixtral), "raw" (Switch), or "auto",
                        which is "raw" at k=1 and "renorm" above. The k=1 case
                        matters: a renormalised gate over a single logit is the
                        constant 1.0, so the router would get no task gradient
                        at all. ``test_renormalised_gate_is_degenerate_at_k_
                        equals_one`` guards it.
  ``capacity_factor``   per-expert token budget as a multiple of the fair share
                        N*k/E; overflow is dropped, lowest gate first.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RouterStats:
    """Routing telemetry for one forward pass.

    Every tensor stays on-device and un-reduced so the trainer can accumulate
    across steps and pay for the host sync only when it actually logs.
    """

    counts: Tensor  # (E,) long -- token/slot pairs the router *chose* per expert
    n_tokens: int
    dense_tokens: int  # tokens routed to every expert instead of just their top-k
    expert_token_pairs: int  # (token, expert) evaluations -- the FLOPs driver
    dropped_pairs: int  # assignments refused because an expert hit capacity
    mode: str


class Expert(nn.Module):
    """A single feed-forward expert."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MoELayer(nn.Module):
    """Top-k mixture-of-experts feed-forward layer.

    Args:
        d_model: model width.
        d_hidden: hidden width of each expert.
        num_experts: number of experts (E).
        k: experts activated per token in sparse mode.
        aux_alpha: weight of the load-balancing auxiliary loss. 0 disables it.
        dropout: dropout inside each expert.
        capacity_factor: per-expert token budget as a multiple of the fair share
            ``N * k / E``. ``None`` means unlimited (no token is ever dropped).
        gate_norm: "auto", "renorm" or "raw" -- see the module docstring.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int,
        k: int,
        aux_alpha: float = 0.0,
        dropout: float = 0.0,
        capacity_factor: float | None = None,
        gate_norm: str = "auto",
    ):
        super().__init__()
        if not 1 <= k <= num_experts:
            raise ValueError(f"k={k} must be in [1, num_experts={num_experts}]")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.k = k
        self.aux_alpha = aux_alpha
        self.capacity_factor = capacity_factor
        if gate_norm not in ("auto", "renorm", "raw"):
            raise ValueError("gate_norm must be 'auto', 'renorm' or 'raw'")
        # With k == 1 a renormalised gate is identically 1.0 and the router gets
        # no gradient, so "auto" falls back to the raw router probability.
        self._gate_norm_auto = gate_norm == "auto"
        self.gate_norm = ("raw" if k == 1 else "renorm") if gate_norm == "auto" else gate_norm

        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden, dropout) for _ in range(num_experts)]
        )

        # Runtime routing controls; the trainer rewrites these per step via
        # MoETransformer.set_routing.
        self.mode: str = "topk"
        self.epsilon: float = 0.0
        self.gate_temperature: float = 1.0
        self.surrogate_alpha: float = 0.0
        self.conf_threshold: float = 0.0
        self.tau_mult: float = 0.0

    # -- routing ---------------------------------------------------------

    def _explore(self, idx: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
        """All-or-nothing epsilon-greedy exploration, decided per token.

        With probability ``epsilon`` a token throws away its entire top-k
        selection and is routed instead to k *distinct* experts drawn uniformly
        from the ``num_experts - k`` experts the router did **not** rank in its
        top k. Drawing from the complement rather than from all E is what makes
        the sample informative: a draw that landed back on an expert the router
        had already chosen would tell it nothing it did not already know.

        Replacing all k slots rather than one also keeps the tempered gate
        clean -- an exploring token has no exploited slot left in its gate
        softmax, so ``forward`` never has to mix tempered and untempered logits
        inside a single normaliser.

        The implementation ranks uniform noise with the top-k entries pushed to
        the back of the order, which is distributionally identical to "draw
        uniformly, resample whenever a top-k expert comes up" but costs one
        sort instead of an unbounded rejection loop.

        Returns ``(selected experts, per-token mask of which tokens explored)``.
        """
        if self.num_experts - self.k < self.k:
            raise ValueError(
                f"exploration needs num_experts >= 2k to draw k distinct experts "
                f"from outside the top-k; got num_experts={self.num_experts}, k={self.k}"
            )
        n = idx.shape[0]
        in_topk = torch.zeros(n, self.num_experts, dtype=torch.bool, device=idx.device)
        in_topk.scatter_(1, idx, True)
        # 2.0 exceeds any uniform draw, so the top-k experts sort to the back and
        # can never land in the first k -- i.e. sampling without replacement from
        # the complement.
        noise = torch.rand(n, self.num_experts, device=idx.device)
        picks = noise.masked_fill(in_topk, 2.0).argsort(dim=-1)[:, : self.k]
        explored = torch.rand(n, device=idx.device) < epsilon
        return torch.where(explored.unsqueeze(-1), picks, idx), explored

    def _surrogate_scale(self, probs: Tensor, idx: Tensor) -> Tensor:
        """Per-token scalar `s` such that `s * y` estimates the skipped experts.

        The model is: an expert's output is proportional to the probability the
        router assigned it, ``E_e(x) ~ c * p_e``, with one slope ``c`` per token
        fitted on the k experts that *were* evaluated. Least squares through the
        origin gives ``c = sum_topk p_e E_e / sum_topk p_e^2``, and the term
        top-k left out of the dense sum is then::

            sum_{j not in topk} p_j E_j(x)  ~  c * sum_{j not in topk} p_j^2

        The numerator of ``c`` never has to be materialised. The dispatched
        output is ``y = sum_topk g_e E_e``, and with no exploration at ``T = 1``
        the gate is proportional to ``p_e`` -- ``g_e = p_e`` for the raw gate,
        ``g_e = p_e / sum_topk p_e`` for the renormalised one -- so
        ``sum_topk p_e E_e`` is just ``y`` times a scalar. The whole correction
        therefore collapses to ``s * y``.

        That collinearity is the method's built-in limitation, and it is worth
        being explicit about: every skipped expert is estimated as a *rescaling
        of the experts that ran*, so the router learns only how much total mass
        belongs off its top-k, never which skipped expert deserves it. This
        generates a balance pressure, not credit assignment.

        Only valid while the gate is proportional to the router probability, so
        it must not be combined with exploration or a gate temperature. A
        capacity factor also biases it, because dropped assignments are missing
        from ``y`` while their probabilities still appear in the fit.
        """
        p_sel = probs.gather(-1, idx)  # (N, k)
        den = p_sel.square().sum(-1, keepdim=True)  # sum_topk p_e^2
        rest = probs.square().sum(-1, keepdim=True) - den  # sum_notchosen p_j^2
        # Undo the gate's normalisation to recover sum_topk p_e E_e from y.
        unnorm = p_sel.sum(-1, keepdim=True) if self.gate_norm == "renorm" else 1.0
        return unnorm * rest / den.clamp_min(1e-9)

    def _dispatch(
        self, x: Tensor, token_of: Tensor, flat_expert: Tensor, flat_gate: Tensor
    ) -> tuple[Tensor, Tensor, int, int]:
        """Run each expert on the tokens routed to it and gate-weight the sum.

        Takes a flat list of (token, expert, gate) assignments rather than a
        fixed-width (N, k) selection, so a batch in which some tokens take k
        experts and others take all E dispatches in one pass.

        Sorting the assignments by expert id turns dispatch into one contiguous
        slice per expert, so each expert is called at most once and only on its
        own tokens. The sort key is ``expert * 2 - gate``, which orders by expert
        and then by *descending* gate within each expert, so when a capacity
        limit bites, the assignments dropped are the ones the router was least
        confident about.

        Returns (output, chosen counts, pairs actually evaluated, pairs dropped).
        """
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        key = flat_expert.to(torch.float32) * 2.0 - flat_gate.detach()
        order = torch.argsort(key)
        tok_sorted = token_of[order]
        gate_sorted = flat_gate[order]

        capacity = None
        if self.capacity_factor is not None:
            # Fair share of the assignments actually made, which is N*k/E when
            # every token takes k experts and rises when some go dense.
            capacity = max(
                1, int(self.capacity_factor * flat_expert.numel() / self.num_experts)
            )

        y = torch.zeros_like(x)
        # Single host sync per layer: we need the slice boundaries on the CPU.
        bounds = counts.tolist()
        start, processed, dropped = 0, 0, 0
        for e, count in enumerate(bounds):
            e_start, start = start, start + count
            take = count if capacity is None else min(count, capacity)
            dropped += count - take
            if take == 0:
                continue
            processed += take
            tokens = tok_sorted[e_start : e_start + take]
            out = self.experts[e](x.index_select(0, tokens))
            gates = gate_sorted[e_start : e_start + take].unsqueeze(-1)
            y.index_add_(0, tokens, out * gates)
        return y, counts, processed, dropped

    def _threshold_assignments(
        self, logits: Tensor, probs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select every expert whose probability clears a threshold.

        The threshold is ``tau_mult / num_experts`` -- expressed in units of the
        uniform share, so it means the same thing at any E: 1.0 is "beat an even
        split", 2.0 is "hold twice an even split". Measured on this model, a
        router that has trained a while puts about 0.39 of its mass on its best
        expert, 0.17 on its second and 0.12 on its third, so the band that
        actually discriminates is roughly 1.0 to 1.6 uniform shares. Beyond ~4.0
        nothing is ever selected on merit, since at most one expert can exceed
        one half.

        Unlike top-k the count varies per token: a confident token keeps one
        expert, an undecided one keeps several. The argmax is always retained,
        so a token whose best expert is still below the threshold goes to that
        expert rather than to nothing at all.

        Returns the flat (token, expert, gate) triples ``_dispatch`` consumes.
        """
        mask = probs > self.tau_mult / self.num_experts
        mask.scatter_(1, probs.argmax(dim=-1, keepdim=True), True)

        if self.gate_norm == "raw" and not self._gate_norm_auto:
            full_gate = probs
        else:
            renorm = F.softmax(  # softmax over the selected logits alone
                logits.masked_fill(~mask, float("-inf")), dim=-1
            )
            if self._gate_norm_auto:
                # Renormalising over a *single* expert yields the constant 1.0,
                # which sends the router no gradient at all -- the trap
                # gate_norm="auto" exists to avoid at k=1, except that here the
                # count is per token, so the fallback has to be per token too.
                # Without it the router would freeze on exactly the tokens it is
                # most confident about, and those grow as the threshold rises.
                full_gate = torch.where(mask.sum(-1, keepdim=True) == 1, probs, renorm)
            else:
                full_gate = renorm

        rows, cols = mask.nonzero(as_tuple=True)
        return rows, cols, full_gate[mask]

    def _assignments(
        self, probs: Tensor, idx: Tensor, gate: Tensor, go_dense: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Flatten the routing decision into (token, expert, gate) triples.

        Tokens outside ``go_dense`` contribute their k selected experts; tokens
        inside it contribute all E, gated by the full router probability exactly
        as a dense pass would weight them.
        """
        n, e = probs.shape
        device = probs.device
        rows = torch.arange(n, device=device)
        if go_dense is None:
            return rows.repeat_interleave(self.k), idx.reshape(-1), gate.reshape(-1)

        # One host sync to size the two groups; only the confidence strategy
        # pays it, since go_dense is None whenever conf_threshold is 0.
        sparse_rows = rows[~go_dense]
        dense_rows = rows[go_dense]
        all_experts = torch.arange(e, device=device)
        return (
            torch.cat([
                sparse_rows.repeat_interleave(self.k),
                dense_rows.repeat_interleave(e),
            ]),
            torch.cat([
                idx[sparse_rows].reshape(-1),
                all_experts.repeat(dense_rows.numel()),
            ]),
            torch.cat([
                gate[sparse_rows].reshape(-1),
                probs[dense_rows].reshape(-1),
            ]),
        )

    def _dense(self, x: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
        """Every expert on every token, weighted by its full router probability."""
        y = torch.zeros_like(x)
        for e in range(self.num_experts):
            y = y + probs[:, e : e + 1] * self.experts[e](x)
        # A dense pass dispatches every token to every expert.
        counts = torch.full(
            (self.num_experts,), x.shape[0], dtype=torch.long, device=x.device
        )
        return y, counts

    # -- forward ---------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, RouterStats]:
        """Args: x of shape (..., d_model). Returns (y, aux_loss, stats)."""
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        n = flat.shape[0]

        logits = self.router(flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)

        dense = self.mode == "dense"
        if dense:
            y, counts = self._dense(flat, probs)
            expert_token_pairs = n * self.num_experts
            dropped = 0  # a dense pass has no capacity limit to overflow
            n_dense = n
            # In dense mode f_i is uniform by construction, so the aux loss has
            # nothing to correct; it is computed from the router probabilities
            # alone below and is anyway zero-weighted for strategies 2 and 3.
            f = torch.full_like(probs[0], 1.0 / self.num_experts)
        else:
            idx = None  # set only on the top-k path
            n_dense = 0
            if self.tau_mult > 0.0:
                token_of, flat_expert, flat_gate = self._threshold_assignments(
                    logits, probs
                )
            else:
                top_p, idx = probs.topk(self.k, dim=-1)  # (N, k)
                # Confidence-gated dense fallback. A token whose top-k experts
                # hold less than `conf_threshold` of the router's probability
                # mass is one the router cannot yet tell apart, so it is sent
                # through every expert rather than committed to a guess. The
                # threshold anneals down, so tokens graduate to sparse routing
                # as the router sharpens; at 0 nothing falls back and this is
                # ordinary top-k, which is what evaluation always sees.
                #
                # The useful range is (k/E, 1]: the top k of any distribution
                # hold at least a uniform k/E share, so a threshold at or below
                # that can never fire.
                go_dense = None
                if self.training and self.conf_threshold > 0.0:
                    go_dense = top_p.sum(-1) < self.conf_threshold
                explored = None
                if self.training and self.epsilon > 0.0:
                    idx, explored = self._explore(idx, self.epsilon)
                # The temperature softens the gate of *explored* tokens only.
                # Every exploited token is gated at T = 1, which is exactly the
                # regime the model is evaluated and deployed in, so the
                # temperature opens no train/deploy gap and never needs
                # annealing away.
                tempered = explored is not None and self.gate_temperature != 1.0
                temps: Tensor | float = 1.0
                if tempered:
                    one = probs.new_ones(())
                    temps = torch.where(  # (N, 1), broadcast over the k slots
                        explored.unsqueeze(-1), one * self.gate_temperature, one
                    )
                if self.gate_norm == "renorm":
                    # softmax over the selected logits; at T = 1 this is
                    # identical to renormalising the selected probabilities.
                    gate = F.softmax(logits.gather(-1, idx) / temps, dim=-1)
                else:
                    # Switch-style: the router's own probability, so absolute
                    # confidence scales the expert's contribution.
                    gate = F.softmax(logits / temps, dim=-1) if tempered else probs
                    gate = gate.gather(-1, idx)
                token_of, flat_expert, flat_gate = self._assignments(
                    probs, idx, gate, go_dense
                )
                n_dense = 0 if go_dense is None else int(go_dense.sum())

            y, counts, expert_token_pairs, dropped = self._dispatch(
                flat, token_of, flat_expert, flat_gate
            )
            if self.training and self.surrogate_alpha > 0.0 and idx is not None:
                # Straight-through: the value cancels exactly, so the forward
                # pass stays plain top-k and the deployed model is unchanged,
                # but the router logits of *every* expert -- including the E - k
                # never evaluated -- now receive gradient through the scale.
                s = self.surrogate_alpha * self._surrogate_scale(probs, idx)
                y = y + y.detach() * (s - s.detach())
            # Denominator is the assignments actually made, not n*k, since both
            # the dense fallback and threshold routing vary the count per token.
            f = counts.to(probs.dtype) / max(flat_expert.numel(), 1)

        aux_loss = flat.new_zeros(())
        if self.aux_alpha > 0.0:
            # Switch/ST-MoE load-balancing loss: alpha * E * sum_i f_i * P_i.
            # Both f and P sum to 1, so the term bottoms out at 1 when the load
            # and the router mass are spread evenly across experts.
            aux_loss = (
                self.aux_alpha * self.num_experts * torch.sum(f * probs.mean(dim=0))
            )

        stats = RouterStats(
            counts=counts.detach(),
            n_tokens=n,
            dense_tokens=n_dense,
            expert_token_pairs=expert_token_pairs,
            dropped_pairs=dropped,
            mode="dense" if dense else "topk",
        )
        return y.reshape(shape), aux_loss, stats
