# Pure-gluon amplitude optimisation — plan

Branch `claude/gluon-amplitude-optimization-8706f5`. Everything is behind
`MG_MERGE_QUARTIC` (off by default), so nothing changes until it is set.

## Goal

Each colour structure of the 4-gluon vertex carries the *same colour factor*
as the diagram obtained by splitting that vertex into two cubic ones. So the
quartic current and the cubic current carrying that colour factor can be
summed into one current, and the whole subtree below it emitted once instead
of twice — a Berends-Giele style recursion, without colour ordering.

```
TMP = VVVVk_1(W_a,W_b,W_c) + VVV1P0_1(VVV1P0_1(W_a,W_b), W_c)
```
Both carry the same `1/P^2` through the same propagator, so this is a plain
sum. Same for amplitudes.

## Established facts (measured, not assumed)

**1. The colour identity holds exactly.** Grouping every amplitude piece by
its colour vector over the ColorBasis:

| process | amplitude pieces | colour groups | cubic diagrams per group |
|---|---|---|---|
| `g g > g g`     | 6   | 3   | exactly 1 |
| `g g > g g g`   | 45  | 15  | exactly 1 |
| `g g > g g g g` | 510 | 105 | exactly 1 |

Every quartic piece is colour-proportional (±1) to exactly one cubic diagram.
Merged count == pure-cubic diagram count == `(2n-5)!!`. `g g > 4g` also
verified in generated Fortran: folding 405 of 510 amplitudes and zeroing the
sources leaves `|M|^2` bit-identical.

**2. The seed rule — forbid two 3-gluon vertices from sharing a line.**

| process | full | seed | seed % | reconstructed by unrolling | missing |
|---|---|---|---|---|---|
| `g g > g g`       | 4    | 1   | 25%   | 4    | 0 |
| `g g > g g g`     | 25   | 10  | 40%   | 25   | 0 |
| `g g > g g g g`   | 220  | 55  | 25%   | 220  | 0 |
| `g g > g g g g g` | 2485 | 385 | 15.5% | 2485 | 0 |

This is forced, not tuned. Unrolling a quartic vertex always yields two
**adjacent** cubic vertices (joined by the line that replaced the vertex), so
a diagram is reachable iff it has an adjacent cubic pair to contract back.
The diagrams with no such pair are exactly the ones that must be in the seed,
and every diagram either has such a pair or is in the seed — necessary and
sufficient, hence exact coverage.

Note "no 3-gluon vertices **at all**" is the over-restrictive special case: it
loses the diagrams whose cubic vertices are non-adjacent (60 of 220 at six
gluons).

**3. Reconstruction gives matched rootings for free.** A reconstructed diagram
and its quartic partner come from the *same* seed diagram, so they share a
decomposition and every current except the one being summed. This is the
property the whole optimisation needs.

## Already committed

| commit | what |
|---|---|
| `fcd8218b6` | link map (`unroll_quartic_vertices`, verified vs colour algebra, N=2..5) + amplitude sums |
| `5c9cdb301` | why the current sums fail on the un-rewired DAG |
| `1c1722ae8` | revert of the auxiliary-particle generation |
| `98d288f41` | `reroot_diagram`, validated 755/755 — probably NOT needed under the seed rule |

Useful pieces to keep: `get_unrollable_quartic_vertices`,
`unroll_quartic_vertices`, `diagram_colour_signature`, `UnrollDiagramTag`,
`split_quartic_vertex`, `unrolled_diagram`, `get_quartic_amplitude_merges`,
`get_amplitude_merge_lines`.

## Plan

**Step 1 — enforce the seed rule in generation.** Reject any combination that
puts two 3-gluon vertices on the same line, inside `reduce_leglist` /
`merge_comb_legs`. Verify the generated seed is exactly the filtered set
measured above (1 / 10 / 55 / 385). *Do not assume it is* — `from_group`
decides which combinations are offered and that interaction has been
mis-predicted before.

**Step 2 — reconstruct the full set by unrolling the seed.** For every seed
diagram, every subset of its quartic vertices, every colour structure. Dedup
with `UnrollDiagramTag`. Gate: diagram count exactly equal to baseline
(4/25/220/2485) and `|M|^2` unchanged. A double count would hide here.

**Step 3 — record the link during reconstruction.** Free: the quartic diagram
and its cubic partner are the same seed diagram unrolled differently. Replaces
the colour-vector matching, which stays as the independent cross-check.

**Step 4 — the current sum.** Where a quartic current and its cubic partner
sit at the same node, emit `TMP = W1 + W4` and the subtree once. Prerequisite,
checked explicitly: their consumers must correspond 1:1 (see pitfall 6).

**Step 5 — validate and time.** `|M|^2` for `g g > N g`, N=2..5 against
baseline; per-call timing with a driver looping `SMATRIX` (the shipped
`check_sa` measures startup, not the ME).

## Pitfalls — all of these cost real time in the previous session

1. **Do not fragment the diagram list.** Generating one diagram per colour
   structure (220 -> 510) works numerically but changes a user-visible number
   and the MadEvent multichannel. It was reverted for that reason.
2. **`copy.deepcopy` on a `Model` silently breaks colour.** `ColorObject`
   derives from `array.array` and deepcopy degrades it to a plain `array`, so
   the structures stop being recognisable. Use `create_copy`.
3. **Two colour-chain conventions exist.** `helas_objects` colorizes the
   *reconstructed* amplitude (`get_base_amplitude`), whose vertex order can
   differ from the generated diagrams. They agree up to six gluons and diverge
   at seven. Anything mapping chains to amplitudes must use the reconstructed
   one.
4. **Pinning a single colour structure needs colour and Lorentz in the same
   frame.** The sum over the three structures is permutation invariant but the
   individual terms are not, so the structure must be chosen against
   `sorted_mothers` (what ALOHA receives), not the vertex leg order.
5. **The `get_color_amplitudes` filter and the writer emitting the folds must
   land together.** Filtering without emitting silently drops the quartic
   contributions from the JAMPs — this produced a wrong `|M|^2` that survived
   three rounds of debugging because the helas calls looked byte-identical to
   baseline. When a number is wrong, diff the JAMP block first.
6. **A current sum is only legitimate if the consumers correspond 1:1.**
   Summing into a shared current hands the contribution to *every* consumer.
   Measured counter-example on the fragmented structure: quartic current 11
   had consumers {7,9,36,38,60,62} mapping to {1,31,55}, while its partner had
   {1,5,31,34,55,58} — three consumers would have gained a term they must not
   have.
7. **The amplitude sums buy nothing at runtime.** The JAMP CSE already finds
   those pairs (`TMP_JAMP(2) = AMP(1) + AMP(4)`). They do shrink the JAMP block
   (1091 -> 697 lines at six gluons), which helps the optimiser and compile
   time. Expect the speedup to come from the currents, not from these.

## Measuring

Generation: `MG_MERGE_QUARTIC=1 ./bin/mg5_aMC` then `generate g g > g g g g`.
Compare `matrix.f` against a run without the variable. Tests:
`./tests/test_manager.py -p U test_diagram_generation test_color_amp
test_helas_objects test_base_objects` (199, must stay green with the flag off).
