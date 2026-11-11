# Private exact geometry postcondition, source only

This compact proposal contains an uncompiled Fortran helper and a separately
tested Python algorithm specification. It changes no production source,
existing certificate/factory, generated export, library or scientific result.
There is no compiler, loader, reducer, amplitude or performance qualification.

The helper proves only exact dyadic equality between supplied component words
and supplied signed linear combinations. It does not own or authenticate the
point, route, numerator, control flow or precision provenance. All those caller
gates remain required and generic scientific dispatch remains disabled.

## Mathematical contract

For an owned, current, same-kind repaired reference point `P`, source-derived
coefficients `a` define `q* = sum(a_e * P_e)`. If the helper returns true, all
four supplied real difference components satisfy `qhat_mu == q*_mu` exactly as
dyadic numbers. Consequently the component construction error is exactly zero;
a caller may then set all four existing error exponents to `zero_bound` and
`have_bounds=true`. A false result provides no bound and must retain fallback.
Do not reuse an earlier successful error array or freshness flag after false.

This postcondition requires no assumption about intermediate rounding,
underflow modes or writes to wavefunction momentum storage: the observed final
components are compared directly with independent owned point bits. It does
not prove arbitrary producer/numerator behavior correct. Snapshot ownership,
current-point/numerator association and exact source routing remain distinct
obligations. In particular, this helper cannot replace their proof with a
matching integer answer.

The original spacelike kernel still includes a separate bound for the **final
dot arithmetic**. Its arithmetic/environment/lowering contract remains required.
Zero construction error is not permission to skip that contract. An alternative
integer enclosure of the actual raw-dot error would be a separate proposal.

## Checked signed128 algorithm

`exact_geometry.f90` exports `exact_geometry_difference(width,reference_width,
external_bits,coefficients,q_bits)`. GNU default INTEGER32 coefficients include
their full signed range; the GNU INTEGER128 kind and default integer width
are compile-time requirements. Widths are binary64/128, with the independently
declared reference width required to match. That label does not establish a
genuine QP numerator or a fresh snapshot.

Fortran arrays have shapes `external_bits(2,0:3,nexternal)` and `q_bits(2,0:3)`;
part1 is real, part2 imaginary. Python specification arrays use
`external_bits[leg][component][part]` and `q_bits[component][part]`. Python raw
words are unsigned canonical bit patterns; a Fortran INTEGER128 carries the
same full128 bits as a signed integer. Binary64 patterns must be zero-extended.

Finite words decode into sign, positive odd significand and integer exponent.
Both signs of zero are accepted. Every imaginary component must be exactly
zero; nonfinite values are refused even for zero-coefficient legs. There are
no FP operations, conversions, ambient reads, callbacks or heap allocations.

Powers of two in coefficients are moved into exponents before multiplication.
The positive significand product is checked against `HUGE/abs(coefficient)`
before multiplication. Terms are aligned to their minimum nonzero exponent.
Every shift and aligned magnitude is bounded before executing it. The complete
sum of absolute aligned magnitudes must fit the positive signed128 range
before **any signed accumulation**. Thus every possible partial signed sum is
representable. Equality is accepted only when the final signed sum is zero in
every component. Cancellation cannot bypass a capacity guard.

Full-range finite IEEE exponent differences do not generally fit in128 bits.
Such cases safely return false. This first helper uses exact arithmetic only:
there is no truncation, guessed error budget or silent multiword approximation.
The fixed64-leg capacity is an explicit unsupported-request fallback, independent
of numerical point values; it covers the requested SM external multiplicities.

## Four prospective retained witnesses

`witnesses.json` records the source coefficients, external ALOHA signs, captured
W/shift/raw-dot words and reconstructed difference words for CMS20/21,
DP1/QP4, route3 pair(2,4). `derive_witnesses.py` verifies the published source
indices and selected compressed/raw stream hashes before parsing anything.

The old trace does **not** observe the temporary `qhat` words, an owned snapshot
or new caller lifetime. Here `qhat` is reconstructed by exact nearest-even
subtraction of retained shifts; replay of products/sequential subtractions
matches retained raw DIFFSQ. Reconstructed zero words use positive zero.
These facts are explicitly conditional on the final arithmetic interpretation;
they are prospective usefulness witnesses, not runtime or signed-zero transport
qualification. Future compiled recording must capture actual q words at use.

| Point/mode | Component absolute-sum bit counts | Equality |
|---|---|---|
| CMS20 DP1 | 8,54,0,53 | all four |
| CMS20 QP4 | 8,54,0,113 | all four |
| CMS21 DP1 | 8,54,0,8 | all four |
| CMS21 QP4 | 8,54,0,110 | all four |

The source DAG's signed coefficients are relative to current repaired `P` and
already contain external wavefunction signs. The captured `W=sign*P` words
must first undo those signs `[-1,-1,+1,+1]` to infer `P`. Applying those same
coefficients directly to `W` double-counts signs and fails the four examples.
The existing bound factory only uses unsigned magnitudes; this is a requirement
for the prospective equality caller, not an observed production regression.

Retained trace capsule index:
`730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`.
Retained source factory index:
`aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047`.
Exact decoder, DAG and compressed/raw trace hashes are stored in the witnesses.
No historical source or result was changed.

## Remaining caller and native work

Capture current repaired and rotated same-kind `P`/`MP_P` into owned local
integer arrays before the relevant numerator producer. Current source3 reads
mutable COMMON `W%P` at the consumer and captures only environment/freshness
tokens earlier; it has no such owned reference. Fixed `MP_PS`, original
`P_USER`, unrepaired input and current rotated `MP_P` are not interchangeable.

Refresh/invalidate the snapshot with point repair, rotations, helicities,
rescue, skipped producers and coefficient/split-order reuse. Preserve the
genuine-QP and source-kind gates. No callee before use may receive or alias the
owned snapshot except a qualified read-only interface. Check actual route
`WE`/`PAIRING`, source signs and point ordering; never infer them from equality.

Keep protected shells/structural zeros, unknown routes, mass-snap precedence,
complex-mass ordering, direct-output policy and all legacy fallback unchanged.
Equality authorizes only zero construction error, never a physical projection
change by itself. The independently qualified spacelike kernel and original
mass/structural selection still decide whether a zero veto is allowed.

Pure tests compare the Python specification against an independent Fraction
oracle, including500 exact random cases,500 deliberate perturbations,600 random
finite-word cases and explicit malformed/kind/nonreal/nonfinite, signed-zero,
subnormal/extreme, multiplication/alignment/absolute-sum capacity controls.
These tests do not execute or validate Fortran lowering, ABI, sticky flags,
snapshot lifetime, final-dot behavior, physical amplitudes or performance.

From the repository cwd, using the pinned Python:

```sh
python3 -B .tirust-build/generic-equality-postcondition-proposal-1/derive_witnesses.py
python3 -B -m unittest discover -s .tirust-build/generic-equality-postcondition-proposal-1 -p test_exact_geometry.py -v
python3 -B -O -m unittest discover -s .tirust-build/generic-equality-postcondition-proposal-1 -p test_exact_geometry.py -v
```

Source-only review must precede a separately admitted future compiled helper
test against independent Fraction answers. Scientific integration requires
actual owned-snapshot and q-at-use observation under the generated caller;
the four reconstructed witnesses do not admit it automatically.
