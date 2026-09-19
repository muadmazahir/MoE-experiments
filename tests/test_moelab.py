"""Correctness tests for the MoE layer, the strategies and the metrics.

Runs standalone (`python tests/test_moelab.py`) or under pytest if installed.
The comparison is only worth reading if the pieces it measures are right, so
these check the things that would silently corrupt the results: sparse dispatch
arithmetic, exploration not leaking into evaluation, gradient accumulation being
mathematically equivalent to a large batch, and the collapse metrics actually
responding to collapse.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moelab.layers import MoELayer  # noqa: E402
from moelab.metrics import FlopModel, collapse_metrics  # noqa: E402
from moelab.strategies import EPS_END, Phase, build_strategies  # noqa: E402
from moelab.transformer import MoEConfig, MoETransformer  # noqa: E402


def _layer(**kw) -> MoELayer:
    torch.manual_seed(0)
    args = dict(d_model=16, d_hidden=32, num_experts=6, k=2)
    args.update(kw)
    return MoELayer(**args)


def test_dispatch_matches_naive_reference():
    """The sorted-slice dispatch must equal a plain per-token loop."""
    layer = _layer().eval()
    x = torch.randn(40, 16)
    y, _, stats = layer(x)

    probs = F.softmax(layer.router(x), dim=-1)
    vals, idx = probs.topk(layer.k, dim=-1)
    gate = vals / vals.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(layer.k):
            ref[t] += gate[t, s] * layer.experts[idx[t, s]](x[t : t + 1])[0]

    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()
    assert stats.counts.sum().item() == x.shape[0] * layer.k
    assert stats.expert_token_pairs == x.shape[0] * layer.k


def test_topk_equals_dense_when_k_is_all_experts():
    """With k=E the renormalised gate is the router distribution itself."""
    layer = _layer(k=6).eval()
    x = torch.randn(24, 16)
    sparse, _, _ = layer(x)
    layer.mode = "dense"
    dense, _, stats = layer(x)
    assert torch.allclose(sparse, dense, atol=1e-5), (sparse - dense).abs().max()
    assert stats.expert_token_pairs == 24 * 6


def test_dense_mode_costs_e_over_k_more_than_topk():
    layer = _layer()
    x = torch.randn(24, 16)
    _, _, sparse = layer(x)
    layer.mode = "dense"
    _, _, dns = layer(x)
    assert dns.expert_token_pairs / sparse.expert_token_pairs == 6 / 2


def _skew_router(layer: MoELayer, bias: list[float]) -> None:
    """Pin the router to a fixed per-expert bias.

    Perturbing the weight matrix directly does not work: with x ~ N(0, I) a
    shifted row just gets a huge-variance logit, not a consistently small one.
    Zeroing the weights and driving them from a constant input coordinate gives
    every token the same, exactly known router distribution.
    """
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, 1:].normal_(0.0, 0.05)
        layer.router.weight[:, 0] = torch.tensor(bias)


def _const_input(n: int, d: int) -> torch.Tensor:
    x = torch.randn(n, d)
    x[:, 0] = 1.0  # the coordinate _skew_router drives the bias from
    return x


def test_exploration_is_training_only():
    """Evaluation must be greedy top-k regardless of the exploration setting."""
    layer = _layer()
    layer.epsilon, layer.gate_temperature = 0.9, 4.0
    x = torch.randn(64, 16)

    layer.eval()
    a, _, sa = layer(x)
    b, _, sb = layer(x)
    assert torch.equal(a, b), "eval routing is not deterministic"
    assert torch.equal(sa.counts, sb.counts)

    layer.train()
    _, _, st = layer(x)
    assert not torch.equal(st.counts, sa.counts), "epsilon had no effect in training"


def test_exploration_flattens_a_skewed_router():
    """A router at init is already near-uniform, so collapse has to be staged."""
    layer = _layer(num_experts=6, k=2).train()
    _skew_router(layer, [6.0, 6.0, 0.0, 0.0, 0.0, 0.0])
    x = _const_input(3000, 16)

    def h_norm(c):
        p = (c / c.sum()).tolist()
        return -sum(q * math.log(q) for q in p if q > 0) / math.log(len(p))

    layer.epsilon = 0.0
    _, _, greedy = layer(x)
    layer.epsilon = 0.5
    _, _, mixed = layer(x)

    assert h_norm(greedy.counts) < 0.7, "the staged router should look collapsed"
    assert h_norm(mixed.counts) > 0.95
    assert mixed.counts.sum() == greedy.counts.sum()
    assert (greedy.counts == 0).sum() == 4  # only experts 0 and 1 ever selected


def test_exploration_draws_only_from_outside_the_top_k():
    """The defining property: an explored token gets experts the router rejected.

    At epsilon = 1 the top-2 therefore receive *nothing* -- exploration here is a
    substitution, not an addition, so a full-exploration pass inverts the load
    rather than flattening it. Flattening comes from mixing the two populations,
    which is what the epsilon < 1 case above measures.
    """
    layer = _layer(num_experts=6, k=2).train()
    _skew_router(layer, [6.0, 6.0, 0.0, 0.0, 0.0, 0.0])
    x = _const_input(3000, 16)

    layer.epsilon = 1.0
    _, _, stats = layer(x)
    assert stats.counts[0].item() == 0 and stats.counts[1].item() == 0
    assert (stats.counts[2:] > 0).all()  # every outsider gets tried


def test_exploration_keeps_selected_experts_distinct():
    layer = _layer(num_experts=6, k=2).train()
    x = torch.randn(3000, 16)
    probs = F.softmax(layer.router(x), dim=-1)
    top = probs.topk(2, dim=-1).indices
    idx, explored = layer._explore(top, 1.0)
    assert explored.all()
    assert (idx[:, 0] != idx[:, 1]).all()
    # disjoint from the top-k it replaced
    assert not (idx.unsqueeze(-1) == top.unsqueeze(1)).any()


def test_exploration_is_all_or_nothing_per_token():
    """A token either keeps its whole top-k or replaces all of it.

    That is what lets the gate temperature stay per-token: an exploring token has
    no exploited slot left in its gate softmax, so tempered and untempered logits
    are never mixed inside one normaliser.
    """
    layer = _layer(num_experts=6, k=3).train()
    x = torch.randn(4000, 16)
    top = F.softmax(layer.router(x), dim=-1).topk(3, dim=-1).indices
    idx, explored = layer._explore(top, 0.5)

    changed = (idx != top).sum(dim=-1)
    assert set(changed.tolist()) <= {0, 3}, "a token changed only some of its slots"
    assert torch.equal(changed == 3, explored)
    assert 0.4 < explored.float().mean().item() < 0.6


def test_exploration_needs_room_outside_the_top_k():
    layer = _layer(num_experts=4, k=2).train()  # complement has exactly k, fine
    top = F.softmax(layer.router(torch.randn(8, 16)), dim=-1).topk(2, dim=-1).indices
    layer._explore(top, 1.0)

    tight = _layer(num_experts=3, k=2).train()  # only 1 expert outside the top-2
    top = F.softmax(tight.router(torch.randn(8, 16)), dim=-1).topk(2, dim=-1).indices
    try:
        tight._explore(top, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected num_experts >= 2k to be enforced")


def test_renormalised_gate_is_degenerate_at_k_equals_one():
    """The trap `gate_norm="auto"` exists to avoid.

    Renormalising over the selected logits when k = 1 is softmax over a single
    number, i.e. the constant 1.0 -- so the router receives no task gradient
    whatsoever and never trains. A router frozen at init looks perfectly
    balanced, which is easy to mistake for a strategy working.
    """
    x = torch.randn(64, 16)

    trap = _layer(k=1, gate_norm="renorm").train()
    trap(x)[0].square().mean().backward()
    assert trap.router.weight.grad.abs().max().item() == 0.0

    fixed = _layer(k=1).train()
    assert fixed.gate_norm == "raw"  # what "auto" selects for k = 1
    fixed(x)[0].square().mean().backward()
    assert fixed.router.weight.grad.abs().max().item() > 0.0

    assert _layer(k=2).gate_norm == "renorm"  # ...and for k > 1


def test_raw_gate_matches_router_probability():
    layer = _layer(k=2, gate_norm="raw").eval()
    x = torch.randn(32, 16)
    y, _, _ = layer(x)

    probs = F.softmax(layer.router(x), dim=-1)
    idx = probs.topk(2, dim=-1).indices
    ref = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s_ in range(2):
            ref[t] += probs[t, idx[t, s_]] * layer.experts[idx[t, s_]](x[t : t + 1])[0]
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()


def test_capacity_drops_overflow_from_a_collapsed_router():
    """A cap is what turns imbalance into an accuracy problem instead of just a
    wasted-parameters problem."""
    layer = _layer(num_experts=4, k=1, capacity_factor=1.25).train()
    x = _const_input(400, 16)

    # A roughly even router fits inside the usual 1.25x headroom.
    _skew_router(layer, [0.0, 0.0, 0.0, 0.0])
    _, _, even = layer(x)
    assert even.dropped_pairs == 0, even.dropped_pairs

    _skew_router(layer, [30.0, 0.0, 0.0, 0.0])  # everything onto expert 0
    _, _, skewed = layer(x)
    capacity = int(1.25 * 400 * 1 / 4)
    assert skewed.counts[0].item() == 400
    assert skewed.expert_token_pairs == capacity  # only a fair share is served
    assert skewed.dropped_pairs == 400 - capacity

    # Dropped assignments cost no FLOPs, and the collapse metrics still report
    # what the router *wanted*, not what capacity allowed.
    assert skewed.counts.sum().item() == 400


def test_no_capacity_means_no_drops():
    layer = _layer(num_experts=4, k=1).train()
    _skew_router(layer, [30.0, 0.0, 0.0, 0.0])
    _, _, stats = layer(_const_input(400, 16))
    assert stats.dropped_pairs == 0
    assert stats.expert_token_pairs == 400


def test_gate_temperature_is_identity_at_one():
    """T = 1 must reproduce the plain renormalised-probability gate exactly."""
    layer = _layer().train()
    layer.gate_temperature = 1.0
    x = torch.randn(48, 16)
    y, _, _ = layer(x)

    probs = F.softmax(layer.router(x), dim=-1)
    vals, idx = probs.topk(layer.k, dim=-1)
    gate = vals / vals.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s_ in range(layer.k):
            ref[t] += gate[t, s_] * layer.experts[idx[t, s_]](x[t : t + 1])[0]
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()


def test_gate_temperature_restores_gradient_to_a_written_off_expert():
    """The reason a plain uniform gate floor is not enough.

    A floor added *after* the softmax is a constant w.r.t. the router logits: it
    revives the expert's own weights but leaves the router's preference frozen,
    because the gate softmax is saturated. Dividing the logits by T desaturates
    it, so both the expert and the router get real gradient.
    """
    layer = _layer(num_experts=4, k=2).train()
    _skew_router(layer, [4.0, 4.0, 4.0, -16.0])  # expert 3 is written off
    x = _const_input(256, 16)
    layer.epsilon = 1.0  # force expert 3 to be tried

    def grads(temp):
        layer.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        layer.gate_temperature = temp
        y, _, _ = layer(x)
        y.square().mean().backward()
        return (
            layer.router.weight.grad[3].norm().item(),
            layer.experts[3].fc1.weight.grad.norm().item(),
        )

    cold_router, cold_expert = grads(1.0)
    warm_router, warm_expert = grads(8.0)
    assert cold_router < 1e-6, cold_router  # saturated: router learns nothing
    assert warm_router > 100 * max(cold_router, 1e-12)
    assert warm_expert > 10 * cold_expert


def test_gate_temperature_touches_explored_tokens_only():
    """Why this temperature never has to be annealed.

    It is applied per token, and only to tokens exploration took over. An
    exploited token is gated at T = 1 no matter how hot the setting is, so the
    greedy path is identical to the deployed one throughout training and there is
    no train/deploy gap for an anneal to close.
    """
    layer = _layer(num_experts=6, k=2).train()
    layer.epsilon = 0.5
    x = torch.randn(512, 16)

    def run(temp):
        torch.manual_seed(11)
        layer.gate_temperature = temp
        return layer(x)[0]

    y_cold, y_hot = run(1.0), run(4.0)

    torch.manual_seed(11)  # replay the same exploration draws
    top = F.softmax(layer.router(x), dim=-1).topk(layer.k, dim=-1).indices
    _, explored = layer._explore(top, layer.epsilon)

    delta = (y_hot - y_cold).abs().amax(dim=-1)
    quiet = delta[~explored].max().item()
    loud = delta[explored].median().item()

    # An exploited token's gate never sees T, so `quiet` is zero in exact
    # arithmetic -- but not reliably zero in floating point. `_dispatch` sorts
    # assignments by `expert * 2 - gate`, so changing the explored tokens' gates
    # permutes the whole assignment array, and index_add_ then accumulates an
    # exploited token's k contributions in a different order. Addition is not
    # associative, so that perturbs them at rounding scale. Whether it does so at
    # all is platform-dependent: exactly 0.0 on an M-series CPU, ~1e-8 elsewhere.
    # The claim worth pinning is the separation, which a real leak would destroy.
    assert quiet < 1e-5, f"temperature leaked outside exploration: {quiet}"
    assert loud > 1e3 * max(quiet, 1e-12), (quiet, loud)
    assert 0.4 < explored.float().mean().item() < 0.6  # the test is not vacuous


def test_surrogate_scale_matches_the_stated_ratio():
    """The arithmetic: output is assumed proportional to router probability.

    k = 1 is the plain statement of the rule -- a chosen expert at p = 0.5
    producing 100 implies an unchosen one at p = 0.25 would have produced 50 --
    and k > 1 fits the same slope by least squares over every expert that ran.
    """
    layer = _layer(num_experts=4, k=1)  # k=1 => raw gate, no renormalisation
    probs = torch.tensor([[0.5, 0.25, 0.15, 0.10]])
    idx = torch.tensor([[0]])
    # c = E_0 / 0.5, so the skipped mass is c * (0.25^2 + 0.15^2 + 0.10^2)
    # and y = 0.5 * E_0; the ratio is 0.095 / 0.25.
    assert math.isclose(layer._surrogate_scale(probs, idx).item(), 0.38, rel_tol=1e-6)

    # The stated example, read straight off the fitted slope.
    c = 100.0 / 0.5
    assert math.isclose(c * 0.25, 50.0)

    layer = _layer(num_experts=4, k=2)  # k=2 => renormalised gate
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    idx = torch.tensor([[0, 1]])
    # den = 0.34, rest = 0.025, and y is divided by sum_topk p = 0.8, so that
    # normalisation has to be undone before the ratio is taken.
    expected = 0.8 * 0.025 / 0.34
    assert math.isclose(layer._surrogate_scale(probs, idx).item(), expected, rel_tol=1e-6)


def test_surrogate_leaves_the_forward_pass_unchanged():
    """Straight-through: it must be a gradient-only intervention.

    If it moved the forward value, the trained model would no longer be plain
    top-k and could not be compared with the other strategies on equal terms.
    """
    layer = _layer(num_experts=6, k=2).train()
    x = torch.randn(128, 16)

    layer.surrogate_alpha = 0.0
    plain, _, sa = layer(x)
    layer.surrogate_alpha = 1.0
    with_surrogate, _, sb = layer(x)

    assert torch.equal(plain, with_surrogate), "the surrogate shifted the output"
    assert torch.equal(sa.counts, sb.counts)
    assert sa.expert_token_pairs == sb.expert_token_pairs  # and costs no experts


def test_surrogate_gives_router_gradient_to_unevaluated_experts():
    """The point of the method.

    Under a renormalised gate the router logits of experts outside the top-k are
    not on the graph at all, so a written-off expert cannot be recovered: the
    router has no way to learn it was wrong. The surrogate puts them back on it.
    (Under the raw gate they already get some gradient, through the full softmax
    normaliser -- which is one reason the two gate norms collapse differently.)
    """
    layer = _layer(num_experts=6, k=2).train()
    _skew_router(layer, [3.0, 3.0, 0.0, 0.0, 0.0, 0.0])  # 2..5 never chosen
    x = _const_input(256, 16)

    def router_grad(alpha):
        layer.zero_grad(set_to_none=True)
        layer.surrogate_alpha = alpha
        y, _, stats = layer(x)
        y.square().mean().backward()
        assert stats.counts[5].item() == 0, "expert 5 should never be selected"
        return layer.router.weight.grad[5].norm().item()

    assert router_grad(0.0) == 0.0  # not on the graph at all
    assert router_grad(1.0) > 1e-5  # and now it is


def test_surrogate_signal_decays_quadratically_with_the_written_off_probability():
    """The method's core weakness, pinned so it cannot be forgotten.

    The estimate of a skipped expert's output is proportional to p_j, and it is
    then weighted by p_j again to enter the sum -- so its whole contribution, and
    the router gradient it generates, goes as p_j^2. The experts most in need of
    rescuing are exactly the ones it says least about, and quadratically so.
    This is the same self-defeating shape the exploration strategy hits, one
    power worse: there the signal was linear in p_j.
    """
    def grad_at(skew):
        layer = _layer(num_experts=6, k=2).train()
        _skew_router(layer, [skew, skew, 0.0, 0.0, 0.0, 0.0])
        layer.surrogate_alpha = 1.0
        layer.zero_grad(set_to_none=True)
        y, _, _ = layer(_const_input(256, 16))
        y.square().mean().backward()
        p = 1.0 / (2.0 * math.exp(skew) + 4.0)  # probability of a written-off expert
        return layer.router.weight.grad[5].norm().item() / p**2

    # grad / p^2 is flat across four orders of magnitude of p.
    ratios = [grad_at(s) for s in (3.0, 5.0, 7.0)]
    assert max(ratios) / min(ratios) < 1.1, ratios


def test_confidence_fallback_routes_unsure_tokens_to_every_expert():
    """High threshold: nobody is confident enough, so every token goes dense."""
    layer = _layer(num_experts=6, k=2).train()
    x = torch.randn(256, 16)

    layer.conf_threshold = 0.0
    _, _, sparse = layer(x)
    layer.conf_threshold = 1.01  # unreachable: top-k mass can never exceed 1
    _, _, dense = layer(x)

    assert sparse.dense_tokens == 0
    assert dense.dense_tokens == 256
    # ...and the FLOP driver reflects it, E per token rather than k.
    assert sparse.expert_token_pairs == 256 * 2
    assert dense.expert_token_pairs == 256 * 6


def test_confidence_threshold_at_or_below_uniform_never_fires():
    """The useful range is (k/E, 1].

    The top k of any distribution hold at least a uniform k/E share, so a
    threshold at or below that can never send a token densely -- worth pinning
    so the knob is not set to a value that silently does nothing.
    """
    layer = _layer(num_experts=8, k=2).train()
    x = torch.randn(512, 16)
    layer.conf_threshold = 2 / 8  # exactly the uniform floor
    _, _, stats = layer(x)
    assert stats.dense_tokens == 0


def test_confidence_fallback_is_training_only():
    """Evaluation is plain top-k however unsure the router is."""
    layer = _layer(num_experts=6, k=2)
    layer.conf_threshold = 1.01
    x = torch.randn(64, 16)

    layer.eval()
    _, _, ev = layer(x)
    assert ev.dense_tokens == 0 and ev.expert_token_pairs == 64 * 2

    layer.train()
    _, _, tr = layer(x)
    assert tr.dense_tokens == 64


def test_mixed_batch_dispatch_matches_a_per_token_reference():
    """The ragged dispatch is the part that could silently be wrong.

    With some tokens taking k experts and others taking all E in one batch, the
    output must equal what you would get computing each token on its own.
    """
    layer = _layer(num_experts=6, k=2).train()
    x = torch.randn(96, 16)
    probs = F.softmax(layer.router(x), dim=-1)
    top_p, idx = probs.topk(2, dim=-1)
    thresh = top_p.sum(-1).median().item()  # split the batch roughly in half
    layer.conf_threshold = thresh
    y, _, stats = layer(x)

    go_dense = top_p.sum(-1) < thresh
    assert 0 < int(go_dense.sum()) < 96, "the split must be genuinely mixed"
    assert stats.dense_tokens == int(go_dense.sum())

    ref = torch.zeros_like(x)
    gate = top_p / top_p.sum(-1, keepdim=True)  # renorm gate at k=2
    for t in range(96):
        if go_dense[t]:
            for e in range(6):
                ref[t] += probs[t, e] * layer.experts[e](x[t : t + 1])[0]
        else:
            for s_ in range(2):
                ref[t] += gate[t, s_] * layer.experts[idx[t, s_]](x[t : t + 1])[0]
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()


def test_threshold_selection_count_tracks_the_threshold():
    """Raising the threshold prunes experts; it never routes a token nowhere."""
    layer = _layer(num_experts=8, k=2).train()
    x = torch.randn(512, 16)

    counts = {}
    for mult in (0.5, 1.0, 2.0, 8.0):
        layer.tau_mult = mult
        _, _, stats = layer(x)
        counts[mult] = stats.expert_token_pairs / 512

    # Monotone in the threshold, and floored at 1: the argmax is always kept,
    # so even an unreachable threshold routes every token to exactly one expert.
    assert counts[0.5] > counts[1.0] > counts[2.0] >= counts[8.0]
    assert math.isclose(counts[8.0], 1.0)
    assert counts[0.5] > 1.0


def test_threshold_selection_survives_evaluation():
    """It is the routing rule, not a training aid, so eval must use it too.

    Every other mechanism here is switched off for evaluation because the model
    would be deployed under plain top-k. This one changes what "the model" is,
    so scoring it under top-k would measure a routing rule it never trained on.
    """
    layer = _layer(num_experts=8, k=2)
    layer.tau_mult = 8.0  # unreachable -> exactly one expert per token
    x = torch.randn(128, 16)

    layer.train()
    _, _, tr = layer(x)
    layer.eval()
    _, _, ev = layer(x)
    assert tr.expert_token_pairs == ev.expert_token_pairs == 128


def test_threshold_gate_keeps_router_gradient_on_single_expert_tokens():
    """The trap this strategy would otherwise walk into.

    Renormalising over a *single* selected expert gives the constant 1.0, so the
    router receives no gradient from that token -- and single-expert tokens are
    exactly what a rising threshold produces. "auto" therefore falls back to the
    raw probability per token, not just per layer as it does for k=1.
    """
    x = torch.randn(256, 16)

    def router_grad(gate_norm):
        layer = _layer(num_experts=8, k=2, gate_norm=gate_norm).train()
        layer.tau_mult = 8.0  # unreachable: every token selects exactly one
        layer.zero_grad(set_to_none=True)
        y, _, _ = layer(x)
        y.square().mean().backward()
        return layer.router.weight.grad.norm().item()

    assert router_grad("renorm") == 0.0, "renorm over one expert is a constant"
    assert router_grad("auto") > 1e-6


def test_threshold_dispatch_matches_a_per_token_reference():
    """Both gate paths: several experts selected, and exactly one."""
    layer = _layer(num_experts=8, k=2).train()
    x = torch.randn(64, 16)

    for mult, expect_multi in ((1.0, True), (8.0, False)):
        layer.tau_mult = mult
        y, _, _ = layer(x)

        logits = layer.router(x)
        probs = F.softmax(logits, dim=-1)
        mask = probs > mult / 8
        mask.scatter_(1, probs.argmax(-1, keepdim=True), True)
        assert (mask.sum(-1) > 1).any() == expect_multi

        ref = torch.zeros_like(x)
        for t in range(64):
            sel = mask[t].nonzero().flatten().tolist()
            gate = (probs[t] if len(sel) == 1
                    else F.softmax(logits[t].masked_fill(~mask[t], float("-inf")), -1))
            for e in sel:
                ref[t] += gate[e] * layer.experts[e](x[t : t + 1])[0]
        assert torch.allclose(y, ref, atol=1e-5), (mult, (y - ref).abs().max())


def test_aux_loss_penalises_imbalance():
    """alpha * E * sum_i f_i P_i: floor of alpha at balance, alpha * E at collapse."""
    layer = _layer(num_experts=4, k=1, aux_alpha=1.0).train()

    _, balanced, _ = layer(torch.randn(512, 16))  # near-uniform router at init
    _skew_router(layer, [30.0, 0.0, 0.0, 0.0])  # everything onto expert 0
    _, collapsed, stats = layer(_const_input(512, 16))

    assert stats.counts[0].item() == 512  # genuinely collapsed
    assert math.isclose(collapsed.item(), 4.0, rel_tol=1e-3)  # alpha * E
    assert math.isclose(balanced.item(), 1.0, abs_tol=0.05)  # alpha * 1
    assert collapsed.item() > balanced.item()


def test_aux_alpha_zero_produces_no_aux_gradient():
    layer = _layer(aux_alpha=0.0).train()
    x = torch.randn(32, 16)
    _, aux, _ = layer(x)
    assert aux.item() == 0.0
    assert not aux.requires_grad


def test_gradient_accumulation_equals_one_large_batch():
    """Strategy 3 must be a pure re-association of the same gradient."""
    cfg = MoEConfig(vocab_size=31, block_size=16, d_model=32, n_head=2, n_layer=2,
                    num_experts=4, k=2, d_hidden=48)
    torch.manual_seed(0)
    model = MoETransformer(cfg).train()
    x = torch.randint(0, 31, (8, 16))
    y = torch.randint(0, 31, (8, 16))

    model.zero_grad()
    model(x, y).loss.backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    for i in range(4):
        sl = slice(i * 2, i * 2 + 2)
        (model(x[sl], y[sl]).loss / 4).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    for a, b in zip(full, accumulated):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_collapse_metrics_separate_balance_from_collapse():
    e = 8
    balanced = collapse_metrics(torch.full((e,), 100))
    collapsed = collapse_metrics(torch.tensor([800] + [0] * (e - 1)))
    assert math.isclose(balanced.entropy_norm, 1.0, abs_tol=1e-9)
    assert math.isclose(balanced.effective_experts, e, rel_tol=1e-9)
    assert balanced.dead_experts == 0 and balanced.cv == 0.0
    assert collapsed.entropy_norm == 0.0
    assert math.isclose(collapsed.effective_experts, 1.0, rel_tol=1e-9)
    assert collapsed.dead_experts == e - 1
    assert collapsed.gini > balanced.gini


def test_flop_model_scales_with_expert_evaluations():
    cfg = MoEConfig(vocab_size=100, block_size=64, d_model=64, n_head=4, n_layer=2,
                    num_experts=8, k=2, d_hidden=128)
    fm = FlopModel(cfg)
    backbone = fm.forward_flops(1, 0)
    one_pair = fm.forward_flops(1, 1) - backbone
    assert one_pair == 2 * 2 * cfg.d_model * cfg.d_hidden
    assert fm.forward_flops(1, 10) - backbone == 10 * one_pair
    assert fm.train_flops(5, 7) == 3 * fm.forward_flops(5, 7)

    sparse = fm.inference_flops_per_token()
    dense = fm.dense_equivalent_flops_per_token()
    assert dense - backbone == (sparse - backbone) * cfg.num_experts / cfg.k


def test_strategy_schedules():
    s = build_strategies(dense_warmup_frac=0.2, eps_start=0.4)
    aux, explore, accum, none = s["aux"], s["explore"], s["accum"], s["none"]

    assert aux.aux_alpha > 0 and aux.accum_steps == 1
    assert explore.aux_alpha == 0.0 and accum.aux_alpha == 0.0 and none.aux_alpha == 0.0
    assert accum.accum_steps > 1

    assert explore.phase(0, 1000).mode == "dense"
    assert explore.phase(199, 1000).mode == "dense"
    p200 = explore.phase(200, 1000)
    assert (p200.mode, round(p200.epsilon, 3)) == ("topk", 0.4)
    assert math.isclose(explore.phase(900, 1000).epsilon, EPS_END)  # anneals away
    # ...but the temperature does not: it only ever reaches an explored token, so
    # it expires on its own as epsilon does.
    for step in (200, 400, 600, 999):
        assert math.isclose(explore.phase(step, 1000).gate_temp, explore.gate_temp)

    assert build_strategies()["explore"].dense_warmup_frac == 0.0  # starts sparse
    for step in (0, 500, 999):
        assert none.phase(step, 1000) == Phase()
        assert accum.phase(step, 1000) == Phase()

    # The confidence threshold anneals to exactly 0, so training ends as plain
    # top-k -- which is the regime those strategies are evaluated in.
    conf = s["conf"]
    assert math.isclose(conf.phase(0, 1000).conf_threshold, conf.conf_start)
    assert conf.phase(400, 1000).conf_threshold < conf.phase(200, 1000).conf_threshold
    assert conf.phase(800, 1000).conf_threshold == 0.0
    assert conf.phase(999, 1000).conf_threshold == 0.0

    # The selection threshold moves the other way: it *rises*, pruning experts as
    # the router earns the right to dismiss them, and it does not return to 0 --
    # it is the routing rule, not a training aid.
    thr = s["threshold"]
    assert math.isclose(thr.phase(0, 1000).tau_mult, thr.tau_start)
    assert thr.phase(400, 1000).tau_mult > thr.phase(200, 1000).tau_mult
    assert math.isclose(thr.phase(999, 1000).tau_mult, thr.tau_end)
    assert none.phase(0, 1000).tau_mult == 0.0  # off for every other strategy


def test_every_block_is_an_moe_layer_and_shapes_hold():
    cfg = MoEConfig(vocab_size=20, block_size=8, d_model=16, n_head=2, n_layer=4,
                    num_experts=4, k=2)
    assert cfg.d_hidden == 32  # derived as 2 * d_model
    m = MoETransformer(cfg)
    assert len(m.moe_layers) == 4
    out = m(torch.randint(0, 20, (2, 8)), torch.randint(0, 20, (2, 8)))
    assert out.logits.shape == (2, 8, 20)
    assert len(out.router_stats) == 4


def test_training_reduces_loss_for_every_strategy():
    from moelab.data import fixed_eval_batches
    from moelab.train import TrainConfig, train_one

    torch.manual_seed(0)
    data = torch.randint(0, 24, (6000,))
    cfg = MoEConfig(vocab_size=24, block_size=16, d_model=32, n_head=2, n_layer=2,
                    num_experts=4, k=2)
    tcfg = TrainConfig(steps=60, batch_size=8, block_size=16, lr=3e-3, eval_every=30,
                       eval_batches=2, device="cpu")
    ev = fixed_eval_batches(data, 16, 8, 2, torch.device("cpu"))
    for strategy in build_strategies(dense_warmup_frac=0.3).values():
        r = train_one(strategy, cfg, tcfg, data, ev, verbose=False)
        assert r["final"]["val_loss"] < r["history"][0]["val_loss"] + 0.05, strategy.name
        assert r["efficiency"]["train_flops"] > 0
        assert 0.0 <= r["final"]["collapse"]["entropy_norm"] <= 1.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
