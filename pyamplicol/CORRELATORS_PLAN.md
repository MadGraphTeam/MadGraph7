# Tree-level colour and spin correlators

## Goal and ownership

Implement opt-in tree-level colour and spin correlations in pyAmpliCol, with
requested colour connections supported through N3LO. Colour correlations are
declared at generation and selected by ID at runtime. Spin-correlation classes
are declared at generation; their numerical contraction vectors are supplied
through `set_spin_correlation_vectors(...)` before evaluation.

Owner: Codex, working in this task under an active goal. Development branch:
`correlators`, based on `main` revision
`7a25f16928feae26cf47f2ebf23e7659902ebf11`. Commit and push directly to that
branch; finish with a pull request into main whose applicable CI is green.
Do not publish packages, modify the completed paper, or change the release
monitor, which is already stopped.

The overriding constraint is **no regression in correctness, generation time,
or warmed runtime performance for uncorrelated calculations**. Correlated
calculations may use simpler, slower paths: no FFT is required, and current
recycling, helicity reductions and related optimizations may be disabled there.
Keep the implementation small and reuse existing amplitude generation,
wavefunction, colour-algebra and profiling machinery.

## References

- [Original MadNkLO correlator note](IMPLEMENTATION_DOCS/REFERENCES/ColorCorrelators_MadNkLO.pdf),
  copied unchanged from the PDF supplied by the user; retained in this branch
  for the development work.
- [MadNkLO source](https://github.com/madnklo/madnklo), especially tree-level
  correlator conventions and Ward/colour-coherence tests. A subagent is
  researching these; inspect licensing before copying implementation text.
- [Catani and Grazzini, hep-ph/9908523](https://arxiv.org/abs/hep-ph/9908523).
- [BLHA2, section 3.8](https://arxiv.org/abs/1308.3462).
- [OpenLoops 2, sections 4.4-4.5](https://arxiv.org/abs/1907.13071).

## Physics and interface contract

Complete phase-consistent complex colour/helicity amplitudes contain the
information needed for physical Born correlations. Current resolved squared
outputs do not: LC outputs are weighted squares, while full-colour outputs
already contain a colour contraction. Reuse internal amplitudes before that
lossy step; do not attempt to infer interference from separate squares.

For a requested colour operator O, build the exact matrix
`K_O[c,d] = <C_c|O|C_d>` at generation and evaluate
`sum_h A_h^dagger K_O A_h` at runtime. Keep operator definitions and public
external-leg labels with their IDs. Prepare only the requested list, and allow
several IDs to share one amplitude evaluation where practical.

Represent higher-order connections by ordered, typed emissions, including
emissions from previously emitted gluons and their splitting into a quark pair.
Support zero through three unresolved emissions, rather than claiming N3LO
support from products of independent NLO dipoles alone. Use an explicit
bra-adjoint/ket convention and document its translation to the note's oriented
non-conjugated connections. Construct complete ordered operators in the colour
tensor algebra; do not multiply overlap matrices as if they were action
matrices in an orthonormal closed basis. Preserve complex ordered correlations
until the requested Hermitian combination is formed.

For declared spin-1 legs, replace the external helicity polarization with a
runtime complex four-vector. The same vector on the amplitude and conjugate
amplitude gives a rank-one spin projection; several simultaneous replacements
give joint products of such projections, not products of separately computed
one-leg results. Mixed spin-colour requests use the selected colour operator
in that same evaluation. General independent bra/ket spin vectors or full
density-tensor output are extensions, not silently implied by this interface.

Supply vectors by public leg ID, with broadcast and per-point batch forms.
Preserve their magnitude and complex phase. Do not silently normalize or
project them. Permit longitudinal momentum substitutions for Ward tests.
Record metric, conjugation, momentum crossing, colour-generator normalization,
helicity averaging and symmetry-factor conventions. A replaced leg is counted
once, not twice as two identical overwritten helicity states. Runtime vector
updates change numerical inputs, never require generation or JIT compilation.
The setter is instance-local, supports reset, and validates batch dimensions.

This is tree-level correlated-Born functionality, not a complete N3LO
subtraction implementation, loop amplitudes, or additional dimensional-epsilon
components absent from the underlying four-dimensional amplitude engine.

## Implementation stages

- [x] Activate the goal and create the clean `correlators` worktree from main.
- [x] Commit/push this plan and the unchanged reference PDF.
- [x] Research MadNkLO's colour/spin APIs and original checks; record useful
      conventions and any intentional differences.
- [x] Implement standalone exact colour-connection algebra through three
      unresolved emissions with small independent tests.
- [x] Add opt-in generation declarations, exact requested colour matrices and
      an ID/definition catalogue to correlated process output only.
- [x] Prepare phase-complete physical amplitude/source schedules for correlated
      requests. Disable state-specific zero tests, squared-result parity
      aliases, numerical-current identities and permutation shortcuts wherever
      their assumptions are invalid for the declared vector dependence.
- [x] Implement runtime vector inputs and ID-selected direct colour reduction,
      reusing existing compiled/recurrence source and amplitude machinery.
      Extend the other applicable execution and precision paths without
      changing uncorrelated paths; make unsupported combinations explicit
      until they are implemented and tested.
- [x] Add user-facing Python/CLI configuration, native interface support where
      applicable, concise documentation and runnable examples. Do not duplicate
      existing CLI generation/profile functionality.
- [x] Complete Ward, colour-coherence, ordered-operator and independent
      low-multiplicity physics validation through N3LO connections.
- [x] Verify the uncorrelated non-regression boundary with focused correctness
      and same-settings generation/runtime comparisons using existing profiling
      commands. Investigate reproducible changes rather than hiding them in
      timing noise or loosening expected results.
- [ ] Push the implementation, open the PR, inspect and fix CI failures, and
      finish only when applicable CI is green and the declared scope is met.

## Isolation and validation

### Development record

The initial plan and PDF were pushed in `50d4af46`. The independent colour
algebra and process-basis adapter pass 55 focused tests. Identity operators
agree with every existing full-colour matrix entry in seven process families,
including three quark pairs plus a gluon. Ordered non-Hermitian examples and
three-step connections involving an emitted quark pair are tested explicitly.

The first runtime uses the existing generic compiled amplitude and retained
Symbolica evaluators through an isolated Python executor. It preserves all
four source components, with no physical-helicity pruning, FFT, replay or
numerical-current reuse. Recurrence/OTF outputs and native correlator calls are
not part of this first runtime. The ordinary native evaluation paths remain
unchanged. Public `evaluate_correlated(...)` is explicit; its spin-vector
setter does not change ordinary `evaluate(...)` calls. Eleven real generated
physics tests pass for `g g > g g`, `d d~ > g g`, `d g > d g` and
`d d~ > u u~ g`: ordinary Born recovery, physical polarization completeness,
incoming/outgoing and joint Ward identities, colour coherence, fundamental
and adjoint Casimirs, three ordered emissions, complex vector homogeneity,
batching and reset. Exact intermediate-state coherence sums additionally
cover emitted gluons and emitted quark pairs through N3LO. Intrinsic fermion
chirality zeros remain zeros; unlike physical-vector pruning, they do not
discard amplitudes that could be restored by an arbitrary vector replacement.

A small uncorrelated CLI A/B check against the base revision used built-in SM
`g g > g g`, full colour, JIT-O2, one worker, batches of 512 and three seconds
of warmed profiling per lane. Compiled runtime was 1.481 versus 1.460 microseconds
per point (base versus branch); recurrence was 7.721 versus 7.747 microseconds,
with 0.46–0.47% statistical standard error. Median generation times from three
fresh outputs were 0.558 versus 0.565 seconds (compiled) and 5.240 versus
5.264 seconds (recurrence). These checks show no meaningful regression; no
ordinary native execution or kernel code changed. The combined process-tree
guard peaked at 0.310 GiB, below the 9.3 GiB guard used for all local physics
and timing checks. The full CI gate remains the final milestone.

MadNkLO was inspected at revision
`646a3db9c8efd7b4cb00e9d89b9197cd5394c01b`. Its ordered emissions and vector-list
semantics inform the design, but the implementation is independent. Its stored
N3LO examples, symmetry assumptions and diagnostic-only checks are not used
as an unconditional oracle. This implementation keeps complex directed
contractions and uses asserting Ward/coherence tests. The original licence is
permissive Illinois/NCSA; no source routines have been copied.

Keep correlated code behind an opt-in boundary before generation/evaluation
dispatch. Default configuration serialization, model/native build identities,
ordinary process output and uncorrelated hot loops must not acquire unnecessary
correlation-dependent work. Shared edits require explicit focused regression
coverage. Do not add general compatibility or release-provenance machinery.

The first implementation uses direct exact colour contraction. Ordinary
colour-metric zeros are not necessarily zeros for an inserted operator;
recompute the requested contractions. The existing FFT permutation symmetry
need not survive tagged colour insertions and is deliberately out of scope for
this first implementation. Genuine operator-independent amplitude identities
remain reusable; it is acceptable to skip them for simplicity in the
correlated path.

Use the smallest authoritative checks at each boundary:

1. Exact algebra: identity operator, physical Casimirs, quark/antiquark signs,
   gluon charges, noncommuting repeated-leg insertions, multi-trace intermediate
   terms and unresolved gluon-to-quark-pair connections. Verify generator
   normalization independently; the existing rescaled trace basis is not a
   reason to rescale physical colour charges.
2. Colour coherence: `B_ii = C_i B`, NLO symmetry and `sum_j B_ij = 0` including
   crossed legs. At higher orders use the correctly ordered/coefficient-weighted
   connection sums, not an unjustified sum over every generated connection.
3. Ward identities: replace one or several massless gauge-boson polarization
   vectors by their longitudinal momenta. Check vanishing amplitudes or their
   projected correlators relative to a meaningful nonzero scale. Include
   nontrivial gluon/multiple-quark examples, crossing and simultaneous vectors.
   Massive-vector replacements obey Slavnov-Taylor/Goldstone relations, not a
   blanket massless Ward-zero expectation.
4. Physics: ordinary-Born recovery from physical polarization sums, vector
   homogeneity, complex phases, mixed spin-colour evaluation, batch/scalar and
   execution-mode agreement. Use explicit small SU(3) tensors and available
   independent tree-level MadNkLO/MadGraph checks as oracles, with fixed phase
   and averaging conventions. Test actual soft/collinear limits where useful.
5. Uncorrelated regressions: preserve existing results and optimized paths.
   Reuse the current green baseline and perform focused before/after timings
   only for touched performance-sensitive boundaries; then one authoritative
   applicable CI run, repairing demonstrated failures.

Keep heavy local work within a single 10 GB process-tree guard unless the user
changes that limit. Coordinate agent tests to avoid simultaneous native builds
or unlicensed Symbolica instances. No repeated performance campaign is needed.

## Reference-note conventions to resolve explicitly

The note provides the intended construction, not immutable implementation
instructions. In particular, distinguish its oriented pairings from positive
`S^dagger S` self-interferences; use `Tr(t^a t^b) = T_R delta_ab` (not C_F);
check symmetry multiplicities and coincident-leg/Casimir cases rather than
transcribing all example formulae. Publish the exact implemented convention
with examples and comparisons to the original implementation.

## User prompts (verbatim)

### Initial assessment request

```text
In the introduction, it says:
"""
Such
insertions require amplitude interference information beyond separate squared components;
they are a motivation for future interfaces and are not yet exposed by pyAmpliCol.
"""
Is it really not always possible to recover spin-correlations and color correlation from our individual helicity and colour flows? I guess not indeed.

Back in the days, I was thinking of providing this API/functionality with the ideas of the PDF above.

Can you do an assessment of how difficult it would be to add such functionality in pyAmpliCol (not code change yet).
```

### Generation/runtime interface clarification

```text
Note that I would implement it so that the list of desired color-correlations would need to be supplied at generation time, and then keyed with an ID, which can then be supplied at runtime.
Spin-correlation on the other hand would be possible to supply using a `set_spin_correlation_vectors(...)` call before runtime evaluations. Of course, at generation time the user would still have to announce the class of spin correlations they intend to capture so as to properly disable some current optimization that would be wrong when not using polarization vectors.
```

### Implementation authorization and requirements

```text
Ok Write this in a plan "CORRELATORS\_PLAN.md" with the above as a plan, adding a goal statement and assigning it to yourself as a goal. (add my prompts regarding this topic verbatim and a link to the PDF I shared that you should include in the repo for now).
You'll be working within a separate branch of pyAmpliCol called "correlators" and commit+push to it directly, eventually yielding a PR that must be green.

In your implementation try and make sure you can have support for up to N3LO correlators directly.
However, it is *OK* to remove a lot of the optimizations currently present for the correlated cases, in order to simplify implementation (e.g. no FFT, disabling some current recycling etc...).
Aim for simplicity over performance there, and importantly, *DO NOT INVOLVE ANY REGRESSION* (both in generation time, correctness and runtime performance) for the uncorrelated case!
Then for testing consider the following two:

a) Ward identities when substituting one or more polarization vectors with the longitudinal momentum (i.e. setting it as the spin-correlation vector).

b) Colour coherence when summing multiple colour-correlators.

c) Not too important here, but there is an old implementation of similar ideas in my old code [https://github.com/madnklo/madnklo](https://github.com/madnklo/madnklo). Have a subagent go through it to fetch any relevant part that may be useful in your implementation (in particular regarding the chosen API conventions, and the nature of the checks I had implemented there on the spin- and colour-correlated MEs). Of course focusing on tree-level MEs only.
```
