# TODO

None of the strategies tested in the pilot produces routers that are both balanced and specialised. The pilot and prior work suggest the two goals should be served by different mechanisms, acting at different times and scales. The list below is what to build and test. Items 9 to 11 are the evaluation setup the other items need, so that a result can be judged on specialisation, capability and real-hardware speed, not just on loss and balance.

1. **Next: `conf` + `surrogate`, with a dense warm-up.** Try `conf` and `surrogate` together, with a short dense warm-up first. The warm-up gives every expert gradient from every token early on. `conf` then moves tokens to top-k one at a time as the router becomes confident. The surrogate keeps router gradient flowing to experts that did not run, which covers `conf`'s weakness: a collapsed router is confident, so it never falls back to dense routing.

2. **Balance through a gradient-free bias at global scope.** Replace the micro-batch auxiliary loss with the bias update of Wang et al. (2024), computed over the global batch:

   $b_i \leftarrow b_i + u \cdot \mathrm{sign}(1/E - f_i^{\text{global}})$

   used only when selecting experts, $\mathrm{Top}\text{-}k(p + b)$, and not in the gate. This keeps experts in use without adding the token-agnostic auxiliary-loss gradient that pushes individual tokens toward random routing, and at global scope it allows domain-skewed micro-batches (Qiu et al., 2025). If an auxiliary loss is kept, it should be computed over the global batch and given a small weight. Tokens routed densely (items 1 and 4) inflate the load histogram $f$, so leave them out of the bias and auxiliary-loss statistics.

3. **Reward decisiveness explicitly.** Add a term that lowers per-token router entropy, so that together with the balancing mechanism the objective targets the token–expert mutual information $I(X;\mathcal{E})$ rather than the balance term $H(\bar p)$ alone:

   $\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \cdot \frac{1}{N}\sum_{t=1}^{N} H(p_t) \; \big[ - \lambda \, H(\bar p^{\,\text{global}}) \big]$

   The bracketed term is needed only if the bias mechanism is not used. With the bracketed term and $\beta = \lambda$, the added term is ModuleFormer's token–expert mutual-information loss (Shen et al., 2023) scaled by $\beta$, the token-level counterpart of Mod-Squad's task–expert objective (Chen et al., 2023). It is also related in spirit to the diversity objectives of Guo et al. (2025). $\beta$ should be warmed up from zero, since rewarding decisiveness before the router has any information would speed up collapse. Entropy minimisation and threshold routing (item 6) both sharpen the router, so combining them needs the balancing mechanism to be active.

4. **Unbiased early gradients.** Train densely where the router is unsure early in training (the `conf` strategy), or use a short dense warm-up (EvoMoE, Nie et al., 2021), so that every expert gets gradient from every region of the input before the partition hardens. In the pilot, `conf` gave the best or joint-best quality; the protocol's warm-up ablation should decide between the per-token and plain warm-up forms.

5. **Dense router gradients after warm-up**, with the renormalised gate only: the `surrogate` strategy, or the per-expert estimate of Default MoE (Panda et al., 2025). The pilot's Exp. 2 shows this must not be combined with the raw gate. The surrogate assumes $g_i \propto p_i$, so it also cannot be combined with exploration or a gate temperature.

6. **Variable k at inference.** Once routing is specialised, many tokens need fewer than k experts. In the pilot, the `threshold` rule used 1.11 experts per token for about 1% higher loss. Applied at the end of training rather than throughout (where, without a capacity limit, it sharpened collapse), it would turn specialisation into compute savings, on top of the memory savings of smaller experts.

7. **A moderate capacity factor** (γ ≈ 1.25) during training, which the pilot found acts as an additional brake on runaway experts.

8. **Stop a few experts from being trained far more than the rest (to test).** Each expert's gradient grows with the number of tokens routed to it, so a popular expert gets much larger updates, gets better, and attracts even more tokens. The idea is to break this loop by limiting how far any expert can pull ahead. Clipping is the natural tool, and there are several things we could clip:
   - **Each expert's gradient.** Clip each expert's gradient norm separately instead of one global norm, or divide each expert's loss by the number of tokens it received, so every expert gets updates of a similar size. Rarely used experts then get noisier updates, so this needs watching.
   - **The router logits.** Soft-cap them, $z \leftarrow c \tanh(z/c)$, as Gemma 2 does for its attention and output logits (Gemma Team, 2024). This bounds the gap between any two of a token's logits by $2c$, so no expert's probability can fall below about $e^{-2c}$ times the top expert's. Use tanh rather than a hard clamp: a clamped logit gets no gradient and can never recover.
   - **The router's updates, as in PPO** (Schulman et al., 2017). Keep a frozen copy of the router's probabilities $p^{\text{old}}$ from the previous step (or a running average) and drop the gradient that would push the ratio $p_{t,i}/p^{\text{old}}_{t,i}$ outside $[1-\epsilon, 1+\epsilon]$. This limits how fast routing can change, so one expert cannot take over a token in a single jump and the balancing mechanism has time to react. It slows collapse but does not limit where routing ends up. A KL penalty towards the old router is the softer version.

   **A spread penalty** is the soft alternative to capping the logits: penalise how far apart each token's logits are, $\frac{1}{N}\sum_t \sum_i (z_{t,i} - \bar z_t)^2$ with $\bar z_t$ the mean of token $t$'s logits. Like the softmax, it ignores a constant added to all of a token's logits, so it measures only the gaps that decide routing. It discourages large gaps, where soft-capping rules them out.

   Caveats:
   - Clipping can only shrink large gradients. An expert that receives no tokens already has zero gradient, so none of these can revive it. They need items 2, 4 or 5 alongside.
   - Limiting the gaps between logits (soft-capping or the spread penalty) makes each token's routing more even, which works against item 3. If both are used, their settings need tuning together. Collapse is also about the same expert winning across many tokens, which a per-token limit does not target directly. So use mild settings, and consider applying them only early in training.

9. **A proper setup for testing specialisation.** Build the specialisation framework from section 8.1 of the paper, since the pilot cannot measure specialisation at all. In short:
   - Subword tokens on a domain-labelled mix (web text, code, maths, several languages), so experts have something to specialise in and we can check it against known domains. Character-level Shakespeare is too uniform.
   - Two model scales (about 100-150M active parameters for the sweep, about 1B to confirm), $E \in \{8, 32, 64\}$, and at least three seeds per run.
   - Per-layer specialisation metrics on held-out data: token-expert and domain-expert mutual information, how stable routing is across checkpoints, the loss increase when one expert is removed (per domain), and how similar the experts' outputs are (CKA).
   - A frozen random-hash router as the baseline: a method whose specialisation metrics do not beat it has not learned to route.
   - The shrinkage experiment: shrink the expert width and check whether more specialised routing loses less quality.

10. **Compare against equivalent models on capability, not just loss.** The pilot only compares validation loss. To show a routing method gives a better model, test it against the following (section 8.2 of the paper describes this too):
    - a dense model with the same active parameters (same compute per token),
    - a dense model with the same total parameters (same memory),
    - a standard MoE of the same size trained with the usual auxiliary loss,

    all trained on the same data for the same number of tokens. Measure them on downstream benchmarks as well as loss, for example with lm-evaluation-harness. At around 100M parameters many benchmarks sit near chance, so pick ones small models can do (HellaSwag, PIQA, ARC-Easy, LAMBADA, WinoGrande), and add harder ones (MMLU, GSM8K, HumanEval) at the 1B scale. Report results per domain too, since a specialised model may gain on some domains and lose on others.

11. **Check that the strategies run efficiently on GPUs, not just that they work in theory.** Whether a strategy works in principle is worth finding out on its own, but to be used it also has to run fast on real hardware. The pilot only counts FLOPs, and FLOPs are a poor guide to GPU speed. Things to watch:
    - `conf` changes the amount of work from step to step (dense tokens cost $E/k$ times a sparse one), and our implementation waits for the GPU every step to count the dense tokens.
    - `threshold` and variable k give each token a different number of experts, so expert loads and matrix shapes become uneven.
    - PPO-style clipping (item 8) needs the previous router's probabilities stored or recomputed.
    - `surrogate` and global-batch balancing should be cheap.

    Measure wall-clock time, throughput and memory on GPUs, with experts spread across several devices (expert parallelism, e.g. MegaBlocks), alongside the FLOP counts. Section 8.6 of the paper describes this too.

## References

- Wang et al. (2024), *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, arXiv:2408.15664.
- Qiu et al. (2025), *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models*, ACL 2025, arXiv:2501.11873.
- Shen et al. (2023), *ModuleFormer: Modularity Emerges from Mixture-of-Experts*, arXiv:2306.04640.
- Chen et al. (2023), *Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners*, CVPR 2023, arXiv:2212.08066.
- Guo et al. (2025), *Advancing Expert Specialization for Better MoE*, NeurIPS 2025, arXiv:2505.22323.
- Nie et al. (2021), *EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate*, arXiv:2112.14397.
- Panda et al. (2025), *Dense Backpropagation Improves Training for Sparse Mixture-of-Experts* (Default MoE), NeurIPS 2025, arXiv:2504.12463.
- Gemma Team (2024), *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv:2408.00118.
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- Gale et al. (2023), *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*, MLSys 2023, arXiv:2211.15841.
