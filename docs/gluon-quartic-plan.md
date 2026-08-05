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
| `25fc6d1cc` | step 1, the seed rule inside `reduce_leglist` |
| `7da06bac7` | step 2+3, `expand_seed_diagrams` and the recorded links |
| `8e634cf9a` | step 4, `TMP = W1 + c*W4` at the amplitudes |

Useful pieces to keep: `get_unrollable_quartic_vertices`,
`unroll_quartic_vertices`, `diagram_colour_signature`, `UnrollDiagramTag`,
`split_quartic_vertex`, `unrolled_diagram`, `get_quartic_amplitude_merges`,
`get_amplitude_merge_lines`.

## Plan

**Step 1 — enforce the seed rule in generation.** DONE, `25fc6d1cc`. Rejected
inside `reduce_leglist`, tracking by leg number which lines a cubic vertex
produced. Measured against the full generation filtered by an independently
written adjacency detector, comparing the diagrams and not only the counts:
1 / 10 / 55 / 385, and 4165 of 34300 at eight gluons.

The closing vertex needs two cases, not one. A real n->0 interaction takes
the legs left over as lines coming in, but a 2->2 like reduction ends on the
*identity* vertex, which only states that its two legs are the two ends of
one line — and that line joins the two vertices which produced them. Missing
that left `g g > g g` with all four of its diagrams.

**Step 2 — reconstruct the full set by unrolling the seed.** DONE,
`1b9474f69`, `expand_seed_diagrams`. Diagram count exactly the baseline
(4/25/220/2485), same diagrams by tag, `|M|^2` bit-identical at four and five
gluons and to 1e-14 at six and seven.

The dedup has to run on the *glued* form. While the identity vertex is still
there the same diagram has several spellings, and a quartic vertex sitting
just in front of it is not yet the last one — which is what decides how ALOHA
indexes its colour structures. Without gluing first, the dedup caught nothing
(40 diagrams for `g g > g g g`) and 8 of 30 recorded colour chains disagreed
with the colour algebra.

**Step 3 — record the link during reconstruction.** DONE, same commit,
`get_quartic_unroll_links`. 3 / 30 / 405 / 6300 links, every one agreeing
target for target with `unroll_quartic_vertices`, which stays as the
independent colour-algebra cross-check.

**Step 4 — the current sum.** DONE at the amplitudes, `8e634cf9a`, and
blocked deeper. `TMP = W1 + c*W4` where a quartic current and its cubic
partner feed the same vertex; the amplitude reads the sum and the quartic
amplitude is never computed:

```
W(20) = W(19)
W(20)%W(:) = W(19)%W(:) + W(12)%W(:)
CALL VVV1_0(W(4),W(5),W(20),GC_10,AMP(33))
```

The sum is shared by every amplitude reading it, so it pays for as many calls
as it has users: 60 amplitude calls for 30 sums at six gluons, 432 for 60 at
seven. Substituting several mothers of one amplitude also produces the
amplitude with all of them substituted, so every subset has to be a merge into
that same target weighing the product of the coefficients — that check is what
keeps the count honest.

Two things it costs. `reuse_outdated_wavefunctions` works out when a slot is
free from the diagrams alone and cannot see the extra read, so it has to be
told, or the two currents get handed the same slot (`W(11) + W(11)`). And the
sums take a slot each, never reused: `NWAVEFUNCS` 51 -> 91 at six gluons.

| | `g g > g g g` | `g g > g g g g` | `g g > 5 g` |
|---|---|---|---|
| helas calls | 94 -> 93 + 7 sums | 637 -> 612 + 30 | 8159 -> 7784 + 60 |
| JAMP lines | 131 -> 101 | 1082 -> 688 | 23672 -> 7864 |
| per call | 34.88 -> 35.12 s | 47.77 -> 46.02 s | 42.16 -> 39.80 s |

**The sum stays at the amplitudes.** Measured
on the reconstructed matrix element (`g g > g g g g`): of the 275 places where
a cubic current is fed by another cubic current *and* the quartic partner
taking the same four lines exists, 225 are the last vertex — the amplitude
sum, which pitfall 7 says buys nothing — and 50 are genuine currents. **None
of the 50 pass the 1:1 consumer test.** At seven gluons, none of 135. Only at
five gluons do all 7 pass.

Why, from a failing pair: quartic current 30 has consumers
`{Wav(3,30), Amp(3,4,6,30)x3, Amp(6,8,30), Amp(4,10,30)}` while its cubic
partner 53 has `{Amp(6,8,53), Amp(4,10,53), Amp(3,12,53)}`. Two correspond;
the rest do not, because the same diagram is rooted differently on the two
sides.

That is forced, not a bug in the reconstruction. Expanding the seed produces
more (seed, choice) instances than there are diagrams, so some diagrams are
reached from several seeds:

| process | seeds | (seed, choice) instances | diagrams |
|---|---|---|---|
| `g g > g g`     | 1   | 4    | 4    |
| `g g > g g g`   | 10  | 40   | 25   |
| `g g > g g g g` | 55  | 340  | 220  |
| `g g > 5 g`     | 385 | 4900 | 2485 |

A diagram carries one decomposition, so it can be rooted to match at most one
of its quartic partners, while the current sum needs the match at *every*
node. Fact 3 above holds per (seed, choice) and breaks under the dedup that
fact 2 requires. Not fixable by choosing the spelling more cleverly: the
counting alone rules it out, and `g g > g g` — the one row with no collision
— is also the one process where the seed rule reaches every partner.

That is why the sum is taken only where the vertex reading it is an
*amplitude*: there is nothing above it, so no consumer can be handed a term
it must not have and the substitution is a plain swap of one argument. A sum
at a current would have to be carried through every current above it, and
those are shared. What would work there is a partial rewrite — hand the sum
only to the consumers which do correspond and leave W1 and W4 serving the
rest — which splits shared consumers and cascades upward. That is a DAG
rewriting problem, not this plan.

The same counting is what leaves the two-substitution cases at seven gluons
on the table: their target has a source with both mothers substituted, but
the source with only the *other* one substituted is spelled with a different
rooting, so the subset check refuses it. 432 of the 864 amplitude calls the
structure allows — and the rest are not simply waiting to be picked up, see
"where to go next".

**Step 5 — validate and time.** DONE. `|M|^2` for `g g > N g`, N=2..5, and
per-call timing from the shipped `check` driver, which already loops
`SMATRIX` when given a second argument (`./check 1000 20000`).

| | `g g > g g g g` | `g g > 5 g` |
|---|---|---|
| flag off | 47.73 / 47.77 / 47.80 s | 42.15 / 42.16 s |
| flag on, before steps 1-4 | 47.76 s | 40.86 s |
| flag on, steps 1-3 only | 48.03 / 48.07 s | 40.41 s |
| flag on, at HEAD | 46.14 / 45.90 s | 39.80 s |

and the code that produces it:

| | helas calls | JAMP lines |
|---|---|---|
| flag off | 637 / 8159 | 1082 / 23672 |
| flag on, before steps 1-4 | 637 / 8159 | 688 / 8012 |
| flag on, steps 1-3 only | 672 / 8216 | 688 / 7864 |
| flag on, at HEAD | 612 + 30 / 7784 + 60 | 688 / 7864 |

So the flag is worth **+3.8% at six gluons and +5.6% at seven**. The JAMP
shrink from `fcd8218b6` carries seven gluons on its own; six gluons only
turns positive with the current sum, because the reconstruction deviates from
the canonical decomposition — which is the whole point — and thereby weakens
the wavefunction CSE, 35 calls at six gluons and 57 at seven. Five gluons is
a wash (34.88 -> 35.12 s): 7 sums against 7 calls is too little to pay for
the 14 extra slots.

`|M|^2` bit-identical at four and five gluons, 1e-15 at six and seven, and
unchanged for `g g > t t~ g g` and `u u~ > g g g`.

With the flag off, `matrix.f` is byte-identical to `3b3ed9e85` for N=2..5.

## Step 6 — madevent, and what it does to AMP2

`29b0c670e`. Two things had to give before the optimisation survived the
madevent path, neither of them about AMP2:

1. **The sum has to be a `CALL`.** Helicity recycling rebuilds the whole DAG
   from the calls alone, so a bare assignment was invisible to it — the summed
   current never entered the graph and the amplitude reading it died on a
   `KeyError`. Written as `CALL SUMW_1(W(a),W(b),W(c))` it is an ordinary
   internal wavefunction with two mothers and everything downstream handles
   it. `sumw_1`/`subw_1` live in `aloha_functions.f`; the coefficient is
   restricted to ±1, which is all it has ever been.
2. **`hel_recycle.add_indices` could not index a statement-initial `AMP(`.**
   The pattern ate the character in front of it and there is none at the
   start of a line, so `AMP(31) = AMP(31) + AMP(1)` came out as
   `AMP(31) = AMP( K,31) + AMP( K,1)`. Latent until now: nothing had ever
   emitted a line beginning with `AMP(`.

**AMP2 is left alone and picks up the merged amplitude**, which is the right
thing and turns out to matter more than the matrix element speedup. The fold
lines run before the AMP2 block, so the channel weight is
`|AMP_cubic + AMP_quartic|^2`. Nothing else was needed: `get_amp2_lines`
already skips any diagram with a four point vertex, so the folded amplitudes
were never referenced.

That skip is the point. With the flag off, at six gluons:

| | amplitudes computed | distinct AMP reaching AMP2 |
|---|---|---|
| flag off | 510 | **105** |
| flag on | 450 (+30 sums) | 105, now carrying all 510 |

so four fifths of the amplitude — every quartic contribution — used to enter
**no** channel weight at all. Folding puts each of them in the channel whose
colour factor it shares, which is exactly where it belongs.

`g g > g g g` through madevent, 10000 events, three seeds:

| | cross section | rel. error | ME cpu |
|---|---|---|---|
| flag off | 3.680-3.694e+07 pb | 0.326% | 52.7 s |
| flag on | 3.684-3.694e+07 pb | 0.300% | 49.0 s |

8% less error for 7% less cpu, consistently across the three seeds. At six
gluons the end-to-end runs are far noisier — the refine stage adapts, and the
same configuration swings by a factor two between seeds — so that comparison
needs a fixed-work measurement rather than `generate_events` wall time.

## Where to go next

What is left is the sums which do not sit at an amplitude, and they need a
node to have exactly one rooting *per merge*, which a diagram list cannot
give. Three ways on, in increasing size:

1. **The second substitution.** Cheapest of the three, ceiling 432 amplitude
   calls at seven gluons, but measured to be at most 282 of them and possibly
   much less. Two things stand in the way and only the first is bookkeeping:

   *Identification.* All 432 targets which have a two-substitution source have
   exactly one of the two singles present; the other is the same diagram
   rooted differently, so it is a different amplitude object with different
   mothers and `match_quartic_mothers` cannot see it. Finding it by its
   diagram and colour chain — the frame `compute_quartic_amplitude_merges`
   already works in — rather than by its mothers would take it.

   *The coefficients have to multiply.* Substituting two mothers also produces
   the amplitude with both substituted, weighing the product of the two
   coefficients, so the merge map has to agree. It does not always: taking the
   double's coefficient over the known single's, the missing single would have
   to weigh -1 for 360 of the 432 targets and +1 for 72, and for **150 of them
   no merge source into that target weighs that at all**. The sign from
   `diagram_colour_signature` does not factorise over two contractions in
   general. `subset_is_merged` already refuses those, which is what keeps the
   present code sound; extending the identification does not remove the check.
2. **Partial CSE.** Keep the DAG, add `TMP = W1 + W4` alongside W1 and W4,
   and split the consumers. Bounded gain: at six gluons only 2 of the 6
   quartic consumers correspond, so it saves 2 subtrees per node out of 50
   nodes.
3. **Drop the diagram list for the currents.** Build the wavefunctions by a
   Berends-Giele recursion over subsets — at each node, cubic pair plus
   quartic, which generates exactly the matchings and never double counts —
   and keep the 220 diagrams only for what they are actually needed for
   (multichannel, `matrix.ps`).

Anything that fragments the diagram list to get a per-rooting copy runs into
pitfall 1, and anything that expands each seed independently double counts —
the fully cubic diagrams get reached once per matching of their adjacency
graph (225 instead of 105 at six gluons).

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
   *Measured since:* wrong at seven gluons, where the JAMP block goes 23672 ->
   7864 lines and that alone is +3.1%. Right at six, where it is a wash.
8. **A wavefunction number is not a slot.** `reuse_outdated_wavefunctions`
   hands the same `me_id` to wavefunctions whose lifetimes do not overlap, and
   works those lifetimes out from the diagrams alone. Anything emitting an
   extra read of a wavefunction has to extend the lifetime there, or it reads
   whatever else has since been written into that slot -- which showed up as
   `W(11)%W(:) = W(11)%W(:) + W(11)%W(:)`.

## Measuring

Generation: `MG_MERGE_QUARTIC=1 ./bin/mg5_aMC` then `generate g g > g g g g`.
Compare `matrix.f` against a run without the variable. Tests:
`./tests/test_manager.py -p U test_diagram_generation test_color_amp
test_helas_objects test_base_objects` (199, must stay green with the flag off).
