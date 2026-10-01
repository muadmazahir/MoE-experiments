# TODO: a combined method for balanced and specialised routing

None of the strategies tested in the pilot produces routers that are both balanced and specialised. The pilot and prior work suggest the two goals should be served by different mechanisms, acting at different times and scales. This is the candidate combination to build and test once the evaluation protocol in the paper (section 8) can measure specialisation.

## Components

1. **Balance through a gradient-free bias at global scope.** Replace the micro-batch auxiliary loss with the bias update of Wang et al. (2024), computed over the global batch:

   $b_i \leftarrow b_i + u \cdot \mathrm{sign}(1/E - f_i^{\text{global}})$

   used only when selecting experts, $\mathrm{Top}\text{-}k(p + b)$, and not in the gate. This keeps experts in use without adding the token-agnostic auxiliary-loss gradient that pushes individual tokens toward random routing, and at global scope it allows domain-skewed micro-batches (Qiu et al., 2025). If an auxiliary loss is kept, it should be computed over the global batch and given a small weight.

2. **Reward decisiveness explicitly.** Add a term that lowers per-token router entropy, so that together with the balancing mechanism the objective targets the token–expert mutual information $I(X;\mathcal{E})$ rather than the balance term $H(\bar p)$ alone:

   $\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \cdot \frac{1}{N}\sum_{t=1}^{N} H(p_t) \; \big[ - \lambda \, H(\bar p^{\,\text{global}}) \big]$

   The bracketed term is needed only if the bias mechanism is not used. With the bracketed term and $\beta = \lambda$, the added term is ModuleFormer's token–expert mutual-information loss (Shen et al., 2023) scaled by $\beta$, the token-level counterpart of Mod-Squad's task–expert objective (Chen et al., 2023). It is also related in spirit to the diversity objectives of Guo et al. (2025). $\beta$ should be warmed up from zero, since rewarding decisiveness before the router has any information would speed up collapse.

3. **Unbiased early gradients.** Train densely where the router is unsure early in training (the `conf` strategy), or use a short dense warm-up (EvoMoE, Nie et al., 2021), so that every expert gets gradient from every region of the input before the partition hardens. In the pilot, `conf` gave the best or joint-best quality; the protocol's warm-up ablation should decide between the per-token and plain warm-up forms.

4. **Dense router gradients after warm-up**, with the renormalised gate only: the `surrogate` strategy, or the per-expert estimate of Default MoE (Panda et al., 2025). The pilot's Exp. 2 shows this must not be combined with the raw gate.

5. **Variable k at inference.** Once routing is specialised, many tokens need fewer than k experts. In the pilot, the `threshold` rule used 1.11 experts per token for about 1% higher loss. Applied at the end of training rather than throughout (where, without a capacity limit, it sharpened collapse), it would turn specialisation into compute savings, on top of the memory savings of smaller experts.

6. **A moderate capacity factor** (γ ≈ 1.25) during training, which the pilot found acts as an additional brake on runaway experts.

7. **Keep router logits small (to test).** The idea is that if no expert's logit can run far ahead of the others, no expert can become so much better than the rest that the router locks onto it. The obvious tool is the router z-loss of ST-MoE (Zoph et al., 2022), which penalises the log-sum-exp of each token's logits, $\frac{1}{N}\sum_t \big(\log \sum_j e^{z_{t,j}}\big)^2$, and was introduced to stop logits growing large and destabilising training (weight $10^{-3}$ in ST-MoE). Two caveats:
   - The z-loss penalises the overall size of the logits, not the gaps between them: it is small whenever the largest logit is near zero, so logits $[0, -20]$ give a z-loss of about $4 \times 10^{-18}$ even though the second expert's probability is only about $2 \times 10^{-9}$. To limit the gaps directly, test a penalty on the spread of each token's logits instead, such as $\frac{1}{N}\sum_t \sum_i (z_{t,i} - \bar z_t)^2$, alongside the plain z-loss.
   - Keeping each token's logits close together makes each token's routing more even, which works against component 2 (rewarding decisive routing). Collapse is also about the same expert winning across many tokens, which a per-token penalty does not target directly. So use a small weight, and consider applying it only early in training.

## Incompatibilities to respect

- The surrogate assumes $g_i \propto p_i$, so it cannot be combined with exploration, a gate temperature, or the raw gate.
- Tokens that fall back to dense routing inflate the training-time load histogram $f$, so they should be excluded from any bias or auxiliary-loss statistics.
- Entropy minimisation and threshold routing both sharpen the router, so combining them requires the balancing mechanism to be active.
- A logit-size or logit-spread penalty (component 7) pulls the other way from entropy minimisation (component 2); if both are used, their weights need tuning together.
- `explore` and `accum` are left out: `explore` cost quality in every setting tried, and `accum`'s benefit should be covered by global-scope balancing at a fraction of its cost.

## References

- Wang et al. (2024), *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*, arXiv:2408.15664.
- Qiu et al. (2025), *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models*, ACL 2025, arXiv:2501.11873.
- Shen et al. (2023), *ModuleFormer: Modularity Emerges from Mixture-of-Experts*, arXiv:2306.04640.
- Chen et al. (2023), *Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners*, CVPR 2023, arXiv:2212.08066.
- Guo et al. (2025), *Advancing Expert Specialization for Better MoE*, NeurIPS 2025, arXiv:2505.22323.
- Nie et al. (2021), *EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate*, arXiv:2112.14397.
- Panda et al. (2025), *Dense Backpropagation Improves Training for Sparse Mixture-of-Experts* (Default MoE), NeurIPS 2025, arXiv:2504.12463.
- Zoph et al. (2022), *ST-MoE: Designing Stable and Transferable Sparse Expert Models*, arXiv:2202.08906.
