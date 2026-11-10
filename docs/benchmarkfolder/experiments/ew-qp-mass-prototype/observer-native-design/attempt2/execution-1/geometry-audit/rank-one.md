# Offline rank-one tensor identities

This appendix evaluates exact rational residuals using only the successful
current-arm TIRust tensor records in the frozen `original-9` and `candidate-9`
captures. It performs no native evaluation and defines no accuracy criterion.
The fixed-mode results retain public status 120 and accuracy −1. There is no
historical NinjaRS replay or prediction of tensor changes between mass arms.

## Identity and conventions

For denominator `D_i=(q+p_i)^2−m_i²`, let `P_i` denote the scalar integral with
denominator `i` omitted. For each Laurent slot, `S=RAW(1)` and
`R=(RAW(2),RAW(3),RAW(4),RAW(5))`, with coordinates `(time,x,y,z)` and metric
`(+---)`. The evaluated identity is

```text
2 (p_i−p_j)·R = P_j−P_i + (p_i²−p_j²−m_i²+m_j²) S.
```

Dividing `D_i−D_j` by the full denominator product gives the identity for the
ordinary loop-momentum moment. The MadLoop rank-one moment has its sign
reversed, giving the displayed formula. The source-bound
`convention.rs:70–82` and `direct/exec.rs:3622–3630` apply only the odd-rank sign
in MadLoop output; they introduce no propagator-count-dependent prefactor.
`direct/plan.rs:1123–1142` supplies the corresponding parent/pinch signs.
`basis.rs:13–27` establishes the first five scalar/Cartesian indices, and the
bridge uses component weights, supplied invariants and `MuSquared`.

All shifts, including the generally nonzero first shift, are preserved. The
coefficient uses exact Minkowski squares of the captured binary128 shifts;
it does not substitute the snapped supplied invariants. The finite, merged
single-pole, double-pole, UV-single and IR-single slots are evaluated separately.
The merged single pole is the exact sum of the two recorded single-pole slots.

## Scalar matching and coverage

A scalar candidate must come from the same mass arm and point, have one fewer
propagator, have native status zero with defined RAW output, and match every
mass, the full supplied invariant matrix and scale in their captured bits
under a propagator permutation. The lowest matching group ID supplies the
scalar. All alternative IDs, admissible permutations, scalar values and
pairwise spreads are retained. The observed alternative scalar spreads are
zero in every reported slot.

Scalar keys exclude absolute shifts by scalar translation/permutation
invariance with supplied invariants. A separate exact-Fraction check tests
whether the matched child shifts equal the retained parent shifts under any
admissible permutation plus one common translation. This is recorded for
every alternative match, not inferred from the scalar key.

| Point, each mass arm | Complete successful parents | Topologies | Pair equations | Selected pinches with exact common translation | Supplied-invariant-only selected pinches |
| --- | ---: | --- | ---: | ---: | ---: |
| Ordinary | 33 | 16 bubbles, 1 triangle, 16 boxes | 115 | 37 | 62 |
| Threshold | 17 | 16 bubbles, 1 triangle | 19 | 33 | 2 |

Sixteen parents in each cohort have exact common translations for every
selected pinch. A supplied-invariant-only match does not establish exact
Cartesian denominator-product consistency. Even an exact common translation
does not remove the discrepancy between supplied invariants and the Cartesian
geometry of the captured shifts.

All 37 refused threshold parents remain excluded from numerical identities,
including the 14 for which every scalar pinch is available. This is a count
of refused parents; group ID 37 is successful. For group 5, all four scalar
pinches match groups 8, 196, 4, and 16/46 at both points, but its threshold
parent has no RAW tensor. The JSON retains availability for every non-tadpole
parent, including partial and absent matches. No partial-pinch equations are
evaluated in this appendix.

## Residual observations

The residual is `LHS−RHS`, evaluated as an exact complex rational number from
the stored binary128 values. Its Euclidean magnitude is printed with
90-digit Decimal arithmetic. The first normalization divides by the sum of
the magnitudes of the four Cartesian contraction terms, both pinch scalars,
and the coefficient-times-scalar term. The second divides by
`max(|LHS|,|RHS|)` after exact summation. Both denominators and exact complex
LHS, RHS and residual values are retained. A zero-over-zero ratio is defined
as zero.

The following are maxima of the first, term-normalized diagnostic over the
covered equations. They are observations, without pass/fail thresholds.

| Arm and point | Finite | Merged single pole | Double pole |
| --- | ---: | ---: | ---: |
| Original, ordinary | 3.0247500e−18 | 2.3101382e−18 | 6.6216733e−35 |
| Candidate, ordinary | 6.4526306e−35 | 5.9922158e−35 | 6.6216733e−35 |
| Original, threshold | 3.2067191e−35 | 0 | 0 |
| Candidate, threshold | 3.3132312e−35 | 0 | 0 |

The ordinary finite absolute maxima are approximately `6.96682e−22` for the
original arm and `2.34935e−28` for the candidate. The selected threshold finite
absolute maxima are approximately `3.01306e−29` and `3.01654e−29` respectively.
The threshold pole zeros concern only the selected bubbles and triangle;
they say nothing about the missing boxes and pentagons or the amplitude poles.

The equation-normalized maxima can be of order one when both equation sides
are almost zero. For example, original ordinary group 134 has a finite
residual magnitude about `1.42665e−29`, an equation-side denominator about
`1.43097e−29`, and a primitive-term denominator about `1.71788e6`. Its two
ratios therefore differ sharply. The JSON retains these cases instead of
discarding cancellations or presenting only the smaller normalization.

These are matched external scalar evaluations, not observed internal master
states. Residuals can contain supplied/Cartesian geometry inconsistency,
rounding of the saved outputs, and reduction errors; this audit neither
separates nor bounds those contributions. It does not certify physical
accuracy, reconstruct missing tensors, establish checked-QP acceptance or
qualify runtime performance.

## Reproduction and provenance

`rank-one.json` binds the script, frozen observer parser, both RAW files,
cards, execution evidence index, observed adapters, bridge and convention
sources by SHA-256. The script verifies the frozen parser and convention
pins and verifies input hashes against the execution index before parsing.
Only the parser module is imported; its main program is never invoked.

From the repository directory, verify the committed appendix without writes:

```sh
/nix/store/463ik27ppxhpbvw7c50s7qcfc7mgfa5x-python3-3.12.12-env/bin/python3.12 -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/execution-1/geometry-audit/rank-one.py --check
```

Without `--check`, the script creates `rank-one.json` exclusively and refuses
to overwrite an existing result. The neighboring geometry audit and all
frozen source and execution records are unchanged.
