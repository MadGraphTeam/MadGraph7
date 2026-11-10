# Standard Model stability plots

On 2026-11-10, Hua-Sheng requested Standard Model stability comparisons like
Appendix C of [arXiv:1604.01363](https://arxiv.org/pdf/1604.01363).
This supplements the integration goal and does not replace its numerical or
runtime acceptance criteria.

Compare CutTools, Ninja, COLLIER, NinjaRS and TIRust on identical saved Standard
Model points and cards. Produce logarithmic cumulative stability distributions,
with a lower panel showing the ratio to Ninja, and publish the plots, data and
reproduction commands with the Typst benchmark report. Include QCD and
electroweak examples and more than one final-state multiplicity.

The paper's plotted quantity is MadLoop's internal stability estimate, using
loop-direction and Lorentz tests. It is not an independent error against an
exact amplitude. The primary comparison must expose double-precision results
before automatic precision rescue; show rescued results separately. Retain
failed, nonfinite, timed-out and unexecuted points in explicit accounting.
Do not substitute accepted automatic-mode accuracy values for a raw-DP study.

Freeze sample definitions, cuts, scales, model, helicity/color sums, test
transformations and reducer/precision identities before execution. Report the
sample size and resulting tail resolution, and avoid ratios where Ninja's
count is zero. A bounded shared-host pilot may precede larger samples; label
the pilot and any remaining larger-sample TODO explicitly.

Keep the ongoing ordinary and stress campaigns unchanged. Add the stability
study under a separate namespace and use private workers. Preserve the known
COLLIER-cache limitation without modifying COLLIER. Precise timing acceptance
remains Hua-Sheng's local-machine TODO, as previously agreed.
