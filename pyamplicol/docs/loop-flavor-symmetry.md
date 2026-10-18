<!-- SPDX-License-Identifier: 0BSD -->
# Internal loop flavour-symmetry prerequisite

`loops.flavor_mappings.prove_flavor_relabeling` checks a signed-PDG permutation
against an explicitly retained source model and counterterm plan. It is a cold
generation prerequisite, not a process-grouping dispatcher or an evaluator.
Existing tree grouping can propose candidates, but neither a tree candidate
nor numerical agreement supplies this certificate.

```python
plan = RetainedFlavorPlan(
    structure, prepared_counterterms,
    ordinary_rules=retained_ordinary_rule_names,
    correction_names=("QCD",),
)
symmetry = prove_flavor_relabeling(plan, {1: 3, 3: 1, -1: -3, -3: -1})
```

The ordinary allow-list must describe the whole retained generation domain.
Selecting only interactions seen in a representative's diagrams is insufficient:
an omitted interaction could contribute to the other member. Every rule in the
prepared counterterm model is checked, including repeated loop alternatives.
The certificate retains the actual source, scalar-program, structural-model and
counterterm-plan owners. It does not certify that those choices constitute a
complete physical theory or renormalization scheme.

The check compares rule multisets before and after relabeling. These retain
ordered particle ports, source colour/Lorentz bases, actual prepared native
bases and propagators, coupling orders, all finite Laurent powers, loop-group
multiplicities, field counterterms and their exact scalar diagnostics. Masses,
widths, charges, quantum numbers, perturbative membership and covariant
multiplet partners must also agree. Native representatives are inspected rather
than assumed to be uniform copies of the source tensors. Symbolica performs all
scalar substitution and algebra; no graph or tensor identities are implemented
by this check.

Independent runtime parameters remain independent even when their defaults are
equal. Only existing fixed-parameter bindings can establish a structural tie.
Thus fixing both light masses to zero can admit a massless `d`/`s` relation;
equal default masses, separately adjustable Yukawas or distinct UV logarithms
cannot. Unknown annotations are retained conservatively. Exact ordered tensor
comparison can reject equivalent representations that require an additional
proved tensor identity.

The present domain is orientation-preserving Dirac species permutations with
model-supplied R2. Charge conjugation and direct R2 are rejected. External-leg
permutations, momentum routing, spin sources, directed colour histories and
interference result transport require separate proofs. No public tree grouping
behaviour changes, and this certificate alone never authorizes skipping a
member's generation or evaluating it through another process.

The opt-in Gmu control retains every ordinary and CT rule with zero QED order
under explicit fixed light-mass/Yukawa restrictions. It checks the full retained
plan and independently generates the `u u~ > d d~` and `u u~ > s s~` Born, loop,
local counterterm and field catalogues. This is not a claim of full EW flavour
symmetry or of a numerical transport implementation.
