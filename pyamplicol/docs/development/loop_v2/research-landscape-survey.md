# Engine v2 landscape survey: automated one-loop multi-leg generators (2006–2026) and the structure pyAmpliCol should adopt

Date: 2026-10-20. Research report. The repository (`pyamplicol`, branch `loop`, 4faafcad) was read only; nothing in it was modified.

**Companion reports in this directory.** This survey does not repeat their derivations; it supplies the broad evidence base they lack and cross-checks their recommendation against it.

- `understand-algorithms.md`: OpenLoops/Recola/MadLoop algorithms, toy operation counts, a Born-seeded core.
- `understand-recursion-literature.md`: recursion relations at one loop, topology keys, the parent-family lift and the NinjaRS probe.
- `understand-tree-engine.md`: the pyAmpliCol tree engine as host.
- `understand-contracts.md`: contracts the new engine must satisfy.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[L]** | Primary literature read in this session: arXiv LaTeX source, PDF text or program source. The location is cited as arXiv id plus section, table label or file line. |
| **[L-s]** | Primary literature verified by a sibling report in this directory and not re-read here. |
| **[L\*]** | Secondary: an abstract, a search-engine summary or a project web page only. |
| **[V]** | Code read in this session (`file:line`). |
| **[I]** | Inference, extrapolation or estimate. |

**Where the sources are**

- Sources downloaded in this session: `workspace/tmp/engine-v2/landscape/src/<arXiv id>/x/`.
- Sources downloaded by sibling sessions: `workspace/tmp/engine-v2/papers/` and `workspace/tmp/engine-v2/papers-lc/`.
- The RADCOR2015 Recola proceedings (PDF only): `workspace/tmp/engine-v2/landscape/pdf_recola_pos/`.
- Program sources: `pyamplicol/FOR_REFERENCE_ONLY/{MG5_aMC_v3_7_0, recola2-2.3.0}`.

**Timings.** All timings are quoted exactly as published, with the hardware. They come from different CPU generations (2008–2025) and are not normalised. A rough single-thread rule of thumb [I]: an i7-4790K (2014, 4 GHz Haswell) is within about ±30 % of one AMD EPYC 9754 thread, the host of the pyAmpliCol benchmarks.

---

## 0. Answers first

### 0.1 Which structures demonstrably scale to 2→5 / 2→6 at one loop

**Full colour, exact per point, colour- and helicity-summed, 7 external legs.** Four structurally different codes have *published per-point timings*:

| Code | Structure | Example | Time | Source |
|---|---|---|---|---|
| OpenLoops 2 | Diagram-based open-loop recursion with recycled subtrees, Born-seeded on-the-fly helicity summation and merging | gg→tt̄ggg | 21 s | [L 1907.13071 tab:perfnew] |
| OpenLoops 2 | same | ud̄→W⁻gggg | 3.8 s | same |
| Recola | Berends–Giele loop currents keyed by propagator history, then Collier | uū→e⁺e⁻ggg (EW+QCD) | about 1.0 s | [L PoS(RADCOR2015)023] |
| GoSam+Ninja | Feynman diagrams in groups, Born-contracted integrand, Ninja | dd̄→e⁺e⁻ggg | 1.9 s | [L 1312.6678 tab:summary] |
| GoSam+Ninja | same | gg→tt̄gg | 13.8 s | same |
| GoSam+Ninja | same | gg→tt̄e⁺e⁻g | 19.6 s | same |
| NJet | Generalised unitarity on primitive amplitudes from Berends–Giele trees, then a colour matrix | 7 gluons | 171 s (including a ×2 scaling test) | [L 1209.0100 tab:timechsum] |
| NJet | same | 2u5g | 195 s | same |

- **Fastest at high multiplicity with many coloured partons:** the OpenLoops 2 class.
- **Recursion-native generation (no diagram enumeration):** Recola, with generation times of seconds to minutes. The diagram-based codes take hours to days (§3.5).
- **Beyond 7 legs:** examples are GoSam+Ninja at 8 legs (H+5j in VBF, 29 s) [L 1312.6678], Recola and HELAC-NLO up to 9 legs with leptons [L 1506.07448; 1609.01659], and MadLoop per helicity at 9 legs [L 1604.01363].
- **No program has published full-colour per-point timings for 8-gluon (2→6) virtuals.**

**Leading colour.** Colour-ordered primitive amplitudes scale polynomially: n⁴ per evaluation for numerical integration [L 1111.1733], and observed n^4.5–n^6 for NJet primitives [L 1209.0100 §5].

- BlackHat ran W+5j production entirely at leading colour (LC) [L 1304.1253].
- Becker, Reuschle and Weinzierl ran e⁺e⁻ → 7 jets at LC [L 1111.1733].

**Every mature code costs about 20–35× per additional gluon at 5→7 legs** in exact full colour with helicity sums (§3.3). The constant and the amount of sharing therefore decide the outcome, not the asymptotic class.

### 0.2 Which of these are compatible with NinjaRS, Born–virtual interference and model UV/R2

**Directly compatible: per-topology polynomial numerators in q, Born-weighted and helicity-summed before one integrand reduction per denominator family.** Three published precedents:

- **OpenLoops 1 + CutTools/Samurai** [L 1111.5206]. Colour- and helicity-summed open-loop coefficients are fed to OPP. Loop diagrams with identical loop propagators are combined. OPP is "always slightly lower" in speed than tensor reduction, with relative efficiency improving with multiplicity.
  - This route was used in production at **2→5**: the tt̄+3-jet NLO calculation (gg→tt̄ggg, 152,070 loop diagrams, exact N_c = 3) used CutTools+OneLOop, with Collier as the alternative [L 1607.06934 §2].
- **MadLoop5 + Ninja tensor interface** [L 1604.01363; 1405.0301 eq. redML5]. This is the direct ancestor of NinjaRS's `TensorNumerator` layout.
- **GoSam + Ninja** [L 1111.2034 algebra.tex; 1312.6678]. The Born contraction is done at the integrand level, per diagram group.

All three take UV/R2 from model counterterms (Feynman rules) inserted in tree amplitudes. That is pyAmpliCol's production R2 path.

**Compatible after reordering: Recola's forward recursion.** Recola computes tensor coefficients per tensor integral and contracts with Born amplitudes *after* Collier. With NinjaRS the Born weight and helicity sum must instead be applied at closure, before reduction. This keeps Recola's generation scheme but loses its "reduce once, contract per (colour, helicity)" economics. The loop-current work is still helicity-explicit.

**Not compatible as a production path:**

- **OpenLoops on-the-fly reduction (OFR).** It is itself a reduction method that outputs low-rank tensor integrals.
- **Unitarity codes (NJet, BlackHat, Rocket, Caravel).** They own the reduction. Feeding NinjaRS from them needs Laurent jets of the numerator per cut, which would mean re-running the recursion per cut. The cost estimate is 50–100× [I, understand-recursion-literature §3.3].
- **Colour-dressed unitarity.** It needs colour sampling [L 0911.1962].
- **Direct numerical loop integration** [L 1111.1733, 1205.2096].

### 0.3 Pitfalls that slowed or capped other programs

Each pitfall maps to an inherited pyAmpliCol weakness (W#, assessment §2.3):

1. **Per-diagram × per-helicity numerator construction.** MadLoop5+Ninja spends 36 s per helicity on coefficients against 0.78 s on reduction for gg→tt̄ggg [L 1604.01363]. This is W4/W5.
2. **Diagram-based symbolic code generation.** It costs 15–25 h of generation, 600–700 MB of code and 3.6–8.3 GB of RAM [L 1405.0301 tab:MLperf]. This is W1/W2.
3. **Explicit helicity loops.** Recola's tensor coefficients for ud̄→W⁺ggg cost 221 ms summed against 14 ms for a single helicity [L 1311.6662]. This is W5.
4. **Factorial colour bookkeeping without a fixed basis or seed.** NJet's 7-gluon full colour costs 56× its de-symmetrised sum [L 1209.0100]; HELAC enumerates 6! colour connections [L 0903.4665]. This is W3.
5. **Double-precision failure rates that grow with n.** NJet needs quadruple precision for 8.6 % of 7-gluon points [L 1209.0100]; a coarse rescue doubles the runtime. Cure: targeted hybrid precision [L 1907.13071]. This is W7.
6. **Arbitrary caps.** NJet's current cache stops at 13 legs [L 1209.0100 §5]; the colour-dressed code stops at n ≤ 12 because of memory [L 0911.1962]. pyAmpliCol has `MAX_ADJOINT_GLUONS = 12` [V `color/adjoint_kernel.py:20`].
7. **Leading-colour approximations used blindly.** LC is about 10 % off for W bb̄ + jets [L 1712.05721].

### 0.4 Recommendation

The literature supports, component by component, the structure the sibling reports converged on. It also sharpens it.

> **A recursion-native, Born-seeded open-loop engine on pyAmpliCol's colour-ordered Berends–Giele currents.**

1. Skeletons and denominator families are generated by a Recola-type keyed recursion over the (n+2)-leg cut process, with no diagram enumeration.
2. The skeletons are dressed with the OpenLoops-2 schedule: the colour-summed Born dual is the seed, helicities are summed on the fly, and objects are merged on (family, undressed suffix).
3. Each family closes into one Born-weighted polynomial V_t(q). It is reduced **a posteriori** by NinjaRS with a master-integral cache.
4. Counterterms are one-insertion trees in the same tree recursion.
5. Rescue escalates per family to DoubleFloat and then per point.
6. All lanes execute one lockstep instruction stream.

Every component has a published precedent (§4.4). Two things are new, and they are the risks to measure first:

- running the OpenLoops-2 schedule over colour-ordered Berends–Giele blocks instead of individual subtrees;
- running a generic-kernel arena, rather than per-process unrolled Fortran, at OpenLoops-class constants.

Expected per-point cost in binary64 on one thread [I]:

| Process | Expected range | Comparison |
|---|---|---|
| gg→ggggg | 10–30 s | NJet about 85 s per evaluation on 2012 hardware; OpenLoops-class extrapolation about 50 s |
| gg→gggg | a few hundred ms to 2 s | |

Lockstep SIMD across phase-space lanes is the only lever that no competitor has.

---

## 1. Constraints the survey is judged against

| Constraint | Source |
|---|---|
| Born–virtual interference only. Finite, 1/ε and 1/ε² coefficients per squared coupling order. | MASTER_PLAN §1.2 |
| Reduction: NinjaRS per denominator family, with numerator polynomial coefficients in q (and μ² for direct R2 only), rank ≤ n+2, plus oneloopmaster masters. | MASTER_PLAN §2.12–2.13; understand-recursion-literature §1 |
| UV and R2 from the loaded UFO loop model (production). Direct R2 only up to 4-point loops. | MASTER_PLAN resumption notes |
| LC, NLC and full colour; colour and spin correlations; flavour mappings. | MASTER_PLAN §2.15–2.17 |
| Double, DoubleFloat and arbitrary precision; native batching and rescue. | MASTER_PLAN §2.18 |
| No enumeration of loop or one-counterterm Feynman diagrams in generation or execution. FeynKit is an oracle only. | User clarification, MASTER_PLAN 2026-10-20 |
| No caps on legs, currents, graphs, ranks or batch sizes. | User rules |
| Speed: faster than MadLoop, and within 1.3× of Recola/OpenLoops in full colour on the required 2→5 set. | MASTER_PLAN §5.7–5.8 |

---

## 2. Program-by-program survey

The fields for each program are: representation, colour, topology tracking and merging, reduction, highest multiplicity, published timings with hardware, scaling, and what pyAmpliCol should take or avoid.

### 2.1 OpenLoops 1 (2011): open loops fed to tensor or OPP reduction

**Representation [L 1111.5206].**

- Diagrams come from FeynArts. A Mathematica program "concatenates [sub-tree and open-loop topologies] in a recursive way, reduces colour factors, and returns Fortran 90".
- Each diagram factorises into a colour factor times a colour-stripped amplitude.
- Subtrees are computed once and reused across tree and loop diagrams.
- The loop numerator is a product of segments. Its tensor coefficients N_{μ1…μr} are built recursively.
- Parent–child recycling shares open-loop prefixes.

**Colour.** One interference matrix over a trace basis; the colour sum costs zero at runtime. Helicity sums are done on the coefficients at the end:

δW_{μ…} = 2 Σ_h (Σ_col M* C) N_{μ…}

**Reduction backends.**

- Collier tensor integrals.
- CutTools or Samurai OPP, **starting from the unpolarised, Born-contracted numerator polynomial**. Quote: "in this way open loops lead to extremely fast helicity sums … The OPP reduction is further improved by combining sets of loop diagrams with identical loop propagators but different external sub-trees" [L].
- Measured: OPP is "always slightly lower" in speed than tensor integrals, "the relative OPP efficiency seems to improve with particle multiplicity", and the same holds for Samurai [L].

**Timings.** Intel i5-750: below 1 ms for 2→2; at most 1 s for 2→4; "almost linearly" proportional to the number of diagrams [L]. The code for gg→W⁺W⁻bb̄ became two orders of magnitude smaller, and its generation time fell from more than one week to 4 minutes [L].

**Highest multiplicity.** 2→5. Sherpa+OpenLoops tt̄+0–3 jets (2016) used "CutTools in combination with OneLOop and, alternatively, the Collier library", with 152,070 one-loop diagrams in gg→tt̄ggg [L 1607.06934 §2, tab:diagcounting].

**R2.** Restored by process-independent counterterms through the tree generator [L].

**For pyAmpliCol.** This is the **closest historical precedent** to the NinjaRS path: per-topology, colour- and helicity-summed tensor numerators fed to integrand reduction. It has been used up to 2→5, and it shows integrand reduction is not the speed limiter.

### 2.2 OpenLoops 2 (2017–2019): on-the-fly helicity summation, merging, OFR and hybrid precision

**Representation.** Same open loops as OpenLoops 1. Three on-the-fly moves [L-s 1710.11452 §3]:

1. The Born interference U₀ = 2 Σ_col M₀*(h) C(I) is injected as the **initial condition**.
2. The helicities of each dressed segment are summed immediately: U_k = Σ_{h_k} U_{k−1} S_k.
3. Objects sharing (topology, cut, undressed segments) are merged. Colour lives in the seed, so diagrams with different colour structures merge.

Measured gains [L 1710.11452, lines 1650–1665]:

- Merging alone and parent–child recycling each give about 2×.
- On-the-fly treatment applied to both helicity sums and merging gives "a further speed-up factor that can vary from two to three".
- Parent–child recycling is incompatible with the on-the-fly seed [L-s].

**OFR.** Pinches during dressing keep rank ≤ 2 (del Aguila–Pittau identities). It is up to 2–3× faster than OpenLoops 1 + Collier. Restricting the on-the-fly approach to helicity sums plus merging, with a-posteriori Collier reduction ("OpenLoops2+Collier"), "can result in a further significant speed-up", while OFR trades a moderate slowdown for stability [L 1710.11452 §6.2.1].

**Stability.**

- Instabilities correlate "over twenty orders of magnitude" with small rank-two Gram determinants. A propagator-permutation trick plus analytic expansions cures most of them [L 1710.11452 near line 4040].
- A hybrid double/quadruple precision system raises the target accuracy from 8 to 11 digits at a cost of 1–50 % [L 1907.13071 §5]. The overhead of quadruple precision over double fell "from two orders of magnitude to a few percent" [L 1907.13624].

**Timings.** Per point, colour- and helicity-summed, i7-4790K at 4.0 GHz [L 1907.13071 tab:perfnew]:

| Process | QCD | EW |
|---|---|---|
| gg→tt̄ | 0.80 ms | |
| gg→tt̄g | 21.4 ms | |
| gg→tt̄gg | 600 ms | 582 ms |
| gg→tt̄ggg | 21.1 s | 16.9 s |
| ud̄→W⁻gggg | 3.76 s | |
| uū→W⁺W⁻ggg | 3.66 s | |
| dd̄→e⁻ν̄μ⁺ν g | 5.6 ms | 42 ms |

- EW costs about the same as QCD for on-shell final states. It is "drastically more expensive" with leptonic W/Z decays, because the loop gains one propagator per decay [L 1907.13071 §5].
- Runtime stays linear in the number of diagrams up to 2→5 [L 1710.11452 §6.2.1].

**Highest multiplicity.** 2→5 QCD phenomenology:

- tt̄+3 jets: 152,070 loop diagrams in gg→tt̄ggg, exact N_c = 3 [L 1607.06934];
- tt̄bb̄+jet: 25,431 diagrams, where OpenLoops 2 was 3× faster than OpenLoops 1 [L 1907.13624].

**Generation.** Mathematica-generated process libraries [L 1907.13071 line 4211]. Users download pre-built libraries; no published generation times were found.

**For pyAmpliCol.**

- **Adopt:** the Born-seeded schedule, on-the-fly helicity summation and merging. The OpenLoops2+Collier variant is the published analogue of a-posteriori NinjaRS reduction.
- **Do not adopt:** OFR, which is a reduction method incompatible with NinjaRS, and FeynArts/Mathematica diagram-based generation, which violates the no-diagram rule.

### 2.3 Recola and Recola2: Berends–Giele loop currents with denominator-history keys

**Representation** [L-s 1211.6316 §2; V `recola2-2.3.0/src/core/skeleton_rcl.f90`, cited in understand-recursion-literature §2.5]:

- The loop is cut into an (n+2)-leg tree.
- Loop currents are interned by (particle, leg bitmask, propagator history `ho`, mass history `hm`, counterterm flag, orders, colour structure).
- Coefficients are summed only for equal denominators. The last current fixes the tensor integral (the family).
- Canonical cut and orientation rules are applied during the recursion. Tadpoles and external self-energies are filtered on the fly.

**Recola is a library, not a code generator.** `generate_processes_rcl` builds the skeleton at runtime and `compute_process_rcl` evaluates it [L 1605.01090 lines 1660–1676].

**Colour.** Colour-flow structures in the current key. Branches that differ only by a colour factor are computed once. The output is structure-dressed amplitudes A₁^{(c,h)}, and the Born interference is taken last [L-s 1605.01090 §2.5–2.6].

**Helicity.** Explicit, with recycling of identical currents across configurations. Tensor coefficients for ud̄→W⁺ggg: 14 ms for one helicity, 76 ms partially summed, 221 ms fully summed; tensor integrals 130 ms; generation 14–52 s [L 1311.6662 Table 1; Intel i5-2400 per L\* search summary].

**Reduction.** Collier once per tensor integral and point. An optional "otter" tensor-reduction backend (mode 4) was added in the 2.2.x series ("interface to the otter tensor reduction library", `recola2-2.3.0/CHANGELOG`) [V `recola2-2.3.0/src/api/input_rcl.f90:982-1005`, `src/CMakeLists.txt:282` `find_package(otter 0.9.2)`]. No publication on otter was found.

**Counterterms.** One-insertion trees, controlled by the x-flag [L-s].

**Timings.** Colour- and helicity-summed, Intel i5-2450M at 2.5 GHz, laptop, mean of 100 RAMBO points [L PoS(RADCOR2015)023 §4]. Times are tensor integrals (TI) + tensor coefficients (TC):

| Process | Corrections | Generation | TI | TC |
|---|---|---|---|---|
| ud̄→W⁺gg | QCD | 2.4 s | 4.0 ms | 1.1 ms |
| ud̄→W⁺ggg | QCD | 15 s | 67 ms | 45 ms |
| uū→W⁺W⁻gg | QCD | 76 s | 83 ms | 16 ms |
| uū→e⁺e⁻gg | EW+QCD | 3.2 s | 27 ms | 25 ms |
| uū→e⁺e⁻uū | EW+QCD | 5 s | 68 ms | 35 ms |
| uū→e⁺e⁻ggg | EW+QCD | 44 s | 331 ms | 684 ms |
| uū→e⁺e⁻uūg | EW+QCD | 50 s | 835 ms | 632 ms |

Recola 1 needed about 30 ms per point for ug→ugZ on an i7-2720QM [L 1211.6316 line 3517].

**Highest multiplicity.** 2→7 off-shell tt̄H, with 9,074 loop diagrams in the gg channel, up to heptagons [L 1506.07448], and 2→6 EW vector-boson scattering [L\* 1611.02951, no timing given].

**Local measurements.** Recola is 10–14× slower than MadLoop at 4 legs and 2.3× at 5 legs [L-s understand-algorithms §3.4, assessment]. This points to a per-call overhead that matters at low multiplicity.

**For pyAmpliCol.**

- **Adopt:** recursion-native generation with history keys, canonical cut rules during the recursion, species merging by masses, and the counterterm flag in the current key.
- **Do not adopt:** the helicity-explicit forward schedule as the production path.
- **Keep in mind:** Recola is the proof that a generic, runtime-built skeleton (an interpreter over generic vertex routines) can be competitive with process-specific generated code at 2→5. Examples: uū→e⁺e⁻ggg takes about 1.0 s on a 2015 laptop against 1.9 s for GoSam+Ninja on a 2014 desktop. Its per-call overhead dominates only at 4–5 legs.

### 2.4 MadLoop4/5 (MG5_aMC): diagrams, topology groups, Born weighting before reduction, Ninja tensor interface

**Representation.**

- **MadLoop4** generates loops by L-cutting MadGraph tree diagrams, then reduces with OPP per diagram and per helicity [L-s 1103.0621 §3].
- **MadLoop5** groups loops by identical denominators and Born-weights and helicity-sums the coefficients before one reduction per topology [L-s 1405.0301 eq. redML5, ML5inputs].
- The code groups loops by denominator structure [V `MG5_aMC_v3_7_0/madgraph/loop/loop_helas_objects.py:652-683` `identify_loop_groups`].
- Open loops are implemented with ALOHA loop-vertex polynomials [L-s].

**Ninja tensor interface [L 1604.01363].** Single-helicity, colour-summed timings on an i7 at 2.7 GHz:

| Process | Integrand coefficients | Ninja | CutTools | Samurai | Generation |
|---|---|---|---|---|---|
| gg→tt̄gg | 215 ms | 85 ms | 240 ms | 362 ms | 14 min |
| gg→tt̄ggg (82,470 diagrams, 9,940 groups, rank 6) | 36 s | 0.78 s | 5.4 s | 9.5 s | about 2 days |
| uū→tt̄bb̄dd̄g (99,981 diagrams, 18,469 groups) | 24 s | 2.0 s | | | about 2 days |

- Loop-induced gg→5Z has one group per diagram (11,700). There reduction dominates: Ninja 5.5 s against 2.5 s of integrand.
- **Lesson:** with Born weighting and grouping, numerator construction dominates; without grouping, reduction dominates.

**Large-scale performance [L 1405.0301 tab:MLperf].** Per helicity, i7 at 2.7 GHz:

| Process | Diagrams | Topologies | Non-zero helicities | Time per helicity | Generation | Code | RAM |
|---|---|---|---|---|---|---|---|
| gg→dd̄bb̄tt̄ | 54,614 | 8,190 | 128 | 18.6 s (19 % reduction) | 15 h | 600 MB | 3.6 GB |
| ud̄→dd̄W⁺ZH | 187,138 | 8,098 | 27 | 26.4 s (32 % reduction) | 25 h | 700 MB | 8.3 GB |

Coefficient time "scales linearly with the number of helicity combinations"; reduction time does not [L].

**Runtime machinery [V `Template/loop_material/StandAlone/Cards/MadLoopParams.dat`].**

- Default reduction cascade Ninja → COLLIER → CutTools (lines 6–21).
- Quadruple-precision integrand optional (lines 23–32).
- "Loop-direction" stability test: re-reduce with propagators reversed, `CTModeRun=-1`, threshold 1e-3 (lines 56–83).
- Rescue is per point: "you are better off rerunning the full PS point in quadruple precision than just a single loop because the two things would almost take the same time" (lines 65–68).
- Helicity filter at level 2 drops zero helicities *and* those identical by symmetry (lines 204–217).
- COLLIER global cache (lines 269–275); internal stability test against the direction-switch test (lines 257–267).
- The same cascade, then quadruple Ninja/CutTools with OneLOop masters, is documented in [L 1804.10017 §4].

**Integrator-level helicity sampling [L 1405.0301 §2.4.3, around line 2916].** MG5_aMC samples **one helicity per point** for the virtual, weighted by |M₀_h|². It relies on V_h/|M₀_h|² being slowly varying and nearly helicity-independent.

**For pyAmpliCol.**

- **Adopt:** Born-weighted per-family numerators fed to Ninja/NinjaRS, the direction test as a cheap reduction-only check, and helicity filtering.
- **Avoid:** per-diagram coefficients, per-helicity cost, diagram-based generation and per-point-only rescue.
- **Note:** MadLoop's exact helicity sum at 7 legs is of order 10³ s per point [I from 36 s per helicity], so "faster than MadLoop" is easy at 2→5. The binding targets are OpenLoops and Recola.

### 2.5 GoSam 1/2/3 (+Samurai, Golem95, Ninja): diagrams, FORM, d-dimensional integrands

**Representation [L 1111.2034; 1404.7096].**

- QGRAF diagrams. FORM algebra produces optimised code; FORM 4 output is "about a factor of ten" faster than the previous output.
- Diagrams are grouped by kinematic matrix S_ij before reduction.
- `diagsum` merges diagrams that differ only by a tree part or by the loop particle content into "meta-diagrams" [L 1404.7096 newfeatures].
- Numerical polarisation vectors replace per-helicity code for massless bosons.

**Colour.** Each diagram is projected onto a basis of generator strings. The Born contraction is done **at the integrand level** before reduction: N_α(q) = Σ ⟨c_i|c_j⟩ C⁽⁰⁾_i* C⁽¹⁾_j(q) [L 1111.2034 algebra.tex].

**Reduction.**

- Samurai: d-dimensional integrand reduction by DFT polynomial interpolation [L 1006.0710].
- Golem95: tensor reduction.
- Tensorial reconstruction: the numerator is re-sampled into tensor coefficients, which speeds up and stabilises reduction [L 1008.2441].
- Ninja (Laurent expansion), the default since 2.0 [L 1404.7096; 1312.6678].
- Ninja's OneLOop wrapper keeps a hash-table master-integral cache, cleared once per point [L 1403.1229 lines 995–1005, 2632–2645].

**Rescue.** Pole test, then rotation test only in a grey zone [L 1404.7096 rescue.tex]. GoSam-3 adds a K-factor test and optional quadruple-precision rescue; it says unstable fractions typically do not exceed the percent range [L 2507.23549 newfeatures].

**Timings [L 1312.6678 tab:summary].** Full colour and helicity sum, i7 at 3.4 GHz with ifort (\* = Xeon E5-2650):

| Process | Diagrams | Time per point |
|---|---|---|
| dū→ν̄e⁻ggg | 1,411 | 226 ms |
| dd̄→e⁺e⁻ggg | 2,928 | 1.91 s |
| gg→tt̄gg | 4,700 | 13.8 s |
| gg→Hggg | 9,325 | 9.0 s (20 s before Ninja) |
| ud̄→e⁺νbb̄gg | 3,923 | 5.4 s |
| gg→tt̄e⁺e⁻g | 4,230 | 19.6 s |
| uū→Hgggu ū (8 legs) | 15,036 | 29.2 s |

GoSam 1.0 needed about 2.5 s for ud̄→e⁺ν ggg on a 2 GHz laptop and 12.3 s for gg→tt̄bb̄ [L 1111.2034].

**GoSam-3.0 (2025).** Still diagrammatic (QGraf, then FORM, then compile) [L 2507.23549 usage.tex]. Speed comes from spinor-bracket reuse across helicities (6 % for ggtt, more than 20× for tttt) and Meson-parallel builds [L]. EW renormalisation is reported as not automated [L\* SciPost abstract].

**For pyAmpliCol.** GoSam is a precedent for Born contraction before integrand reduction and for meta-diagram merging. Its numbers on three of the required processes (dd̄→e⁺e⁻ggg, gg→tt̄gg, gg→tt̄e⁺e⁻g) give useful 2014 anchors. **Do not adopt** its per-diagram symbolic algebra.

### 2.6 HELAC-1Loop / HELAC-NLO: numerator from tree recursion, OPP, colour and helicity sampling

**Representation [L-s 0903.4665; 1110.1499].**

- A loop "structure" (a partition of the legs into blobs, plus loop flavour and colour connection) is evaluated as part of an (n+2)-point HELAC Dyson–Schwinger tree.
- CutTools samples the numerator at its own loop momenta, so this is evaluation-first.
- Colour connections are enumerated: 6! = 720 for six gluons [L 0903.4665 lines 490–494, 770–774].

**Timings [L 0903.4665 lines 1143–1160].**

- uū→bb̄bb̄: 100–200 ms per colour connection and helicity; about 10 s fully summed.
- With colour and helicity sampling: 200 ms per event.
- Re-weighting a sample of unweighted Born events gives "a speed-up factor of 10³" against a straightforward Monte Carlo integration of the virtual.

**Highest multiplicity.** Off-shell tt̄j, 2→7: 39,180 one-loop diagrams in the gg channel, including 120 heptagons, rank up to 6 [L 1609.01659].

- That run needed Monte Carlo over colour and helicity, reweighting, parallel per-topology skeleton generation (one order of magnitude faster), and Ward-identity tests with higher-precision re-evaluation of failures [L 1609.01659].
- R2 comes from a counterterm skeleton [L-s].

**For pyAmpliCol.** HELAC shows recursion-native skeletons work. Its evaluation-first numerator is incompatible with NinjaRS tensor numerators. Its scaling relies on sampling, which our per-point exact contract forbids for the default path.

### 2.7 NGluon / NJet / NJet3: generalised unitarity over Berends–Giele primitives, full colour by a colour matrix

**Representation [L 1209.0100 §2].**

- Primitive amplitudes, each with one "parent diagram" (using placeholder propagators where a vertex is absent).
- D-dimensional unitarity with OPP-style subtraction. Cuts are products of Berends–Giele trees with cached currents, O(n⁴) per tree.
- Rational terms come from massive-scalar cuts (FDH), not model R2.
- Full colour comes from a once-per-process decomposition of partial into primitive amplitudes (RREF).

**Scaling.** Primitive cost grows as n^4.5–n^6 observed and n⁸ asymptotically. "Up to 13 external legs, the maximally allowed number with the cache system turned on" [L 1209.0100 §5].

**Timings [L 1209.0100 tab:timechsum, tab:timedsymgluons].** Full colour and helicity sum, Xeon E3-1240 at 3.3 GHz, including **two** evaluations for the scaling test:

| Process | Double precision | To 4 digits | Points fixed in quadruple precision |
|---|---|---|---|
| 6g | 6.19 s | 6.81 s | 1.37 % |
| 7g | 171.3 s | 276.7 s | 8.63 % |
| 2u5g | 195.1 s | 241.2 s | 3.25 % |
| 4u3g | 92.5 s | | |

- Quadruple precision is "7–8 times" slower than double.
- Gram determinants "are not the only sources of numerical instability" [L].
- For 7 gluons "it might be required switching to octuple precision" [L §5].
- **De-symmetrised colour sums**, valid only after symmetric phase-space integration: 6g 0.57 s and 7g 3.07 s [L].

**Phenomenology.** Five-jet production at NLO with full-colour virtuals for all 7-point channels [L 1309.6585]. A leading/subleading split was explored: the subleading part is about 10 % of the virtual at low p_T [L 1312.7140].

**Other timings.**

- e⁺e⁻→5 jets one-loop: about 23 s per point (6.39×10³ h for 10⁶ points) [L 2002.07516 Table "comp_comparison"].
- NGluon 2011: n gluons 0.03 s / 0.5 s / 15 s for n = 4/5/6, on an i7 at 2.7 GHz, full colour [L 1201.1187 tab:times].

**NJet3.** Version 3.0 (June 2021) added two-loop finite remainders [L\* Bitbucket wiki via search; 2106.08664]. The one-loop engine is unchanged.

**For pyAmpliCol.** NJet provides the only published full-colour 7-gluon and 2u5g anchors. Two strategies are worth offering as options:

- de-symmetrised colour sums (not per point);
- leading/subleading split for the integrator.

Its reduction path is incompatible with NinjaRS for production. It is a valuable evaluation-first oracle.

### 2.8 BlackHat: on-shell primitives, leading colour at very high multiplicity

**Representation [L 0803.4180; 1304.1253].**

- Primitive amplitudes from on-shell methods: Forde, with OPP-like subtraction, and Badger's massive continuation for rational terms.
- Assembly into partial amplitudes uses Ita–Ozeren colour decompositions [L 1111.4193].
- Unstable points recompute **only the affected terms** in higher precision with QD (double-double, then quad-double) [L 1304.1253].

**Timings [L 0803.4180 TimingTable].** Per colour-ordered helicity primitive, 2.33 GHz Xeon, with multiprecision: 11–114 ms for 6 gluons, 23 ms for 7 and 43 ms for 8 (MHV). This is about 2× per gluon.

**Highest multiplicity.** W+5 jets: "eight-point loop" amplitudes [L 1304.1253 line 343].

- Virtuals are LC. Subleading terms are dropped because they are "typically 10 % of the leading-color virtual terms, and under 3 % of the total cross section". Poles keep full colour [L].
- Four-jet production used full colour [L 1112.3940].
- W bb̄ + jets: LC is "at the order of 10 %" wrong, so results use full colour, split into LC and subleading terms sampled independently [L 1712.05721 lines 590–600, 1028–1040].
- No per-point timing is published for W+5j.

**For pyAmpliCol.** The colour split must be exact and separately callable; LC is a cost mode, not a default approximation. Partial-term rescue corresponds to our per-family rescue.

### 2.9 Rocket and the D-dimensional unitarity line

- **Giele–Zanderighi** (0805.2152): N⁹ per colour-ordered helicity primitive, 4–20 gluons [L-s; review L 1105.4319 §10].
- **Ellis–Giele–Kunszt–Melnikov** [L-s 0806.3467]: masses and fermions.
- **Rocket W+3j** [L 0901.4101 §num]: LC only, about 50 ms per primitive per helicity on a 2.33 GHz Xeon. That makes 112 evaluations, or 5.6 s per point, so virtuals were computed on a fixed grid.
- **Ellis–Kunszt–Melnikov–Zanderighi review**: subleading-colour primitives cost (n−2)× the LC ones, so they are integrated with less relative accuracy [L 1105.4319 §10].

**For pyAmpliCol.** Historical. It confirms the polynomial-per-primitive, factorial-in-colour picture.

### 2.10 Colour-dressed generalised unitarity (Giele–Kunszt–Winter 2009)

Colour-dressed Berends–Giele recursion inside unitarity cuts [L 0911.1962 tab:timepsps]:

- Exponential cost, ×6 per gluon in 4D and ×7 in 5D, per colour and helicity assignment.
- Colour-ordered with sums grows ×8–13.
- At 7 gluons, 12 s (ordered) and 13 s (dressed) per configuration on a 3 GHz Core2, including a ×2 consistency recomputation. The crossover is at 7 < n < 8.
- Memory limits the code to n ≤ 12 (4D) and n ≤ 10 (5D).
- It needs Monte Carlo over colours.

**For pyAmpliCol.** Not exact per point, so not suitable. The tree-level analogue agrees: PEPPER's explicit colour sum over colour-ordered Berends–Giele beats Comix's colour dressing until about 5 extra jets [L 2311.06198 near line 956].

### 2.11 Caravel (2020)

A C++ framework for multi-loop numerical unitarity: colour-ordered, Berends–Giele off-shell recursion for cut products, finite-field and functional reconstruction [L 2009.11957]. One-loop is a special case. Its focus is analytic multi-loop reconstruction, not fast full-colour Born–virtual evaluation.

**For pyAmpliCol.** Not a structural template.

### 2.12 Numerical loop integration (Becker–Reuschle–Weinzierl) and Loop-Tree Duality

- Becker–Reuschle–Weinzierl: colour-ordered Berends–Giele recursion for the integrand, local UV/IR subtraction, contour deformation, Monte Carlo over the loop momentum.
- At LC the cost per evaluation scales as n⁴, the same as the Born.
- e⁺e⁻→7 jets took "a few days on a cluster with 200 cores" [L 1111.1733; 1205.2096; 1010.4187].
- Loop-Tree-Duality contour deformation (1912.09291) is a related direct-integration programme [L\* abstract].

**For pyAmpliCol.** Incompatible with the NinjaRS/master-integral contract.

### 2.13 Reduction libraries

| Library | Method | Notes |
|---|---|---|
| CutTools [L 0711.3596] | OPP | Multi-precision routines "with up to 2000 digits"; N=N tests. |
| Samurai [L 1006.0710] | D-dimensional integrand reduction with DFT interpolation | Full rational term. |
| Ninja [L 1203.0291; 1403.1229; 1604.01363] | Laurent-expansion integrand reduction | Diagonal system. 2.5–7× faster than CutTools and 4–12× faster than Samurai in MadLoop's tables. Master cache. |
| Collier [L 1604.06792] | Tensor reduction, COLI/DD branches, expansions | **Global cache keyed by call position:** identical arguments are detected on the first points; requires "all calls … exactly in the same order for each phase-space point". |
| Golem95 | Tensor reduction | Rescue in GoSam. |
| otter | Tensor reduction | Recola2 backend only [V]. |

**For pyAmpliCol.** Its master cache should follow the Ninja hash-table design or the Collier call-position design. The understand-recursion-literature report counted a 107× master-request redundancy at 7 legs.

### 2.14 GPUs, SIMD and surrogates

**No published production code evaluates one-loop amplitudes on GPUs or with SIMD batching** (several targeted searches; [L\*]). Evidence for amplitudes:

- **Trees:** MG5_aMC CUDACPP reaches lockstep execution with 100 % branch efficiency on GPUs and "the maximum theoretically possible SIMD speedups (x8 and x16 in double and single…)" [L 2303.18244]. PEPPER and BlockGen run colour-ordered and colour-dressed Berends–Giele recursion on GPUs [L 2311.06198; 2106.06507].
- **NLO:** after Born and real-emission matrix elements are offloaded, "Virtuals are now the bottleneck of the calculation" [L 2609.23500, MadVfold, 2026].
- **One-loop integrals on GPUs:** Yuasa et al. and pySecDec [L\* refs in 2106.06507].
- **Surrogates:** neural networks emulate NJet virtuals [L 2002.07516; 2206.14831]. They need training data and do not provide a per-point guarantee.

**For pyAmpliCol.** Native lane batching through NinjaRS and the Rust arena is unique among one-loop generators. To exploit SIMD/GPU the instruction stream must be data-independent: the same schedule for all lanes, with instability handled by lane compaction and rescue rather than branching.

### 2.15 Integrator-level strategies used in production (not engine structure, but the engine must enable them)

| Strategy | Who | Effect | What the engine must offer |
|---|---|---|---|
| Single-helicity virtual sampled by \|M₀_h\|² | MG5_aMC [L 1405.0301] | Coefficient work / #helicities | A helicity subset or single helicity at proportionally lower cost |
| Virtual on unweighted Born events (reweighting) | HELAC-NLO [L 1110.1499]; Rocket fixed grid [L 0901.4101] | ~10³ fewer virtual calls | Fast cold start; batches |
| LC + subleading sampled separately | BlackHat [L 1712.05721]; NJet [L 1312.7140]; review [L 1105.4319] | Subleading ~10 % of virtual, evaluated less often | Exact LC and remainder as separate outputs |
| De-symmetrised colour sums | NJet [L 1209.0100] | 7g: 171→3 s | Optional, labelled "integrated-equivalent" |
| Neural surrogates | [L 2002.07516] | Large but unverified per point | Training data at scale |

---

## 3. Cross-cutting tables

### 3.1 Structural comparison

| Program | Generation | Loop object | Topology tracking | Colour | Helicity | Reduction | UV/R2 | Max one-loop multiplicity |
|---|---|---|---|---|---|---|---|---|
| OpenLoops 1 | FeynArts + Mathematica | Open-loop tensor coefficients per diagram; recycled subtrees | Parent–child prefix sharing; diagrams with equal propagators combined | Per-diagram factor × interference matrix (trace basis) | Summed at end on coefficients | Collier tensor integrals **or CutTools/Samurai on Born-weighted per-topology polynomials** | Counterterms via tree generator | 2→5 (tt̄+3j, CutTools or Collier) |
| OpenLoops 2 | Mathematica libraries | Same, Born-seeded | Merging on (topology, cut, undressed suffix) | In the seed U₀ | **On the fly** | OFR (rank ≤ 2) + Collier scalars, or a-posteriori Collier | Same + R1 power counting | 2→5 QCD/EW (tt̄+3j, tt̄bb̄j) |
| Recola / Recola2 | **Runtime recursion, no diagrams** | Berends–Giele loop currents (tensor coefficients) | Key: legs, `ho`, `hm`, CT flag, orders, colour | Colour flow in key; structure-dressed amplitudes | Explicit with recycling | Collier (or otter) once per tensor integral | x-flag trees; REPT1L | 2→7 off-shell (leptons) |
| MadLoop5 | MadGraph L-cut diagrams | Per-diagram coefficients (ALOHA loop polynomials) | Groups by denominator set | Per diagram Λ_lb | Per helicity, linear cost | Ninja/Collier/CutTools cascade, once per topology | UFO counterterms (tree-matching) | 2→7 per helicity |
| GoSam 2/3 | QGRAF + FORM | d-dimensional numerator per diagram group | S_ij groups + diagsum | Basis projection, Born-contracted integrand | Per helicity (numerical polarisation vectors) | Ninja / Samurai / Golem95 | Explicit or counterterms | 2→6 (H+5j VBF) |
| HELAC-1Loop | HELAC skeleton (recursion) | Numerical numerator from (n+2)-tree | Loop structures (blobs) | Colour connections ((n_q+n_g)!) | Sampled | CutTools (OPP) | R2 skeleton | 2→7 (sampling) |
| NGluon / NJet | Primitives (parent diagrams) | Cuts = products of Berends–Giele trees | Per primitive | Primitives + colour matrix; LC natural | Summed per primitive | Own (D-dimensional unitarity, OPP) | Rational from massive cuts (FDH) | 2→5 full colour |
| BlackHat | Primitives | On-shell cuts | Per primitive | Ita–Ozeren decomposition; LC production | Per primitive | Own | Own (massive continuation) | W+5j at LC |
| GKW (colour-dressed) | Recursion | Colour-dressed cuts | Per colour config | Dressed + Monte Carlo | Per config | Own | — | 12 gluons per config |
| Weinzierl | Recursion | Integrand | None | LC | Summed | None (Monte Carlo integration) | Local subtraction | e⁺e⁻→7 jets at LC |
| **Inherited pyAmpliCol** | FeynKit diagrams + symbolic projectors | Per-diagram coefficient DAGs | Families; no sharing | Pairwise exact overlaps | All 2ⁿ | NinjaRS per family (Born-weighted) | UFO counterterms by one-CT tree enumeration | 2→4 validated |

### 3.2 Published anchors for the required 2→5 set

These are the nearest published processes, not the same processes.

| Required process | Nearest published | Code | Time per point | Hardware | Colour / helicity |
|---|---|---|---|---|---|
| gg→gggg | 6g | NJet [L 1209.0100] | 6.19 s (two evaluations); de-symmetrised 0.57 s | Xeon E3-1240, 3.3 GHz | Full / summed |
| gg→gggg | 6g | NGluon [L 1201.1187] | about 15 s (with rescaling) | i7, 2.7 GHz | Full / summed |
| gg→ggggg | 7g | NJet | 171 s (277 s to 4 digits); de-symmetrised 3.07 s | Xeon E3-1240 | Full / summed |
| uu~→ggggg | 2u5g | NJet | 195 s (241 s to 4 digits) | Xeon E3-1240 | Full / summed |
| gg→tt~gg | gg→tt̄gg | OpenLoops 2 [L 1907.13071] | 600 ms (QCD), 582 ms (EW) | i7-4790K, 4.0 GHz | Full / summed |
| gg→tt~gg | gg→tt̄gg | GoSam+Ninja [L 1312.6678] | 13.8 s | i7, 3.4 GHz | Full / summed |
| gg→tt~gg | gg→tt̄gg | MadLoop5+Ninja [L 1604.01363] | 300 ms per helicity | i7, 2.7 GHz | Full / one helicity |
| uu~→e⁻e⁺ggg | dd̄→e⁺e⁻ggg | GoSam+Ninja | 1.91 s | i7, 3.4 GHz | Full / summed |
| uu~→e⁻e⁺ggg | uū→e⁺e⁻ggg (EW+QCD) | Recola [L PoS RADCOR2015] | 1.02 s | i5-2450M, 2.5 GHz | Full / summed |
| uu~→e⁻e⁺dd~g | uū→e⁺e⁻uūg (EW+QCD) | Recola | 1.47 s | i5-2450M | Full / summed |
| gg→tt~e⁻e⁺g (QCD) | gg→tt̄e⁺e⁻g | GoSam+Ninja | 19.6 s | i7, 3.4 GHz | Full / summed |
| gg→tt~e⁻e⁺g (EW) | none | — | — | — | OpenLoops 2: EW ≫ QCD with leptonic decays |
| Context: uu~→ggg (5 legs) | same | local EPYC 9754 [L-s assessment] | MadLoop 36 ms, OpenLoops 37 ms, Recola 82 ms, **inherited pyAmpliCol 125 ms** | EPYC 9754, one thread | Full / summed |

**Implied targets [I].** These are rough and must be replaced by the local provider campaign of MASTER_PLAN §5.8:

| Process | OpenLoops-class target, EPYC single thread |
|---|---|
| gg→tt̄gg | about 0.5–0.8 s |
| uu~→e⁻e⁺ggg | about 0.3–1 s |
| gg→gggg | about 1–3 s, extrapolated from 26,975 diagrams × about 60 µs/diagram (the OpenLoops 2 per-diagram cost at 6 legs) |
| gg→ggggg | about 40–80 s (3×10⁵ diagrams × about 140–250 µs) |

The OpenLoops numbers measured locally so far look anomalous: 11.9 ms for gg→gg, against 0.8 ms published for gg→tt̄. They are under audit and should not be used as the gate yet.

### 3.3 Scaling with n: cost ratio per added gluon

| Code / family | 4→5 legs | 5→6 | 6→7 | Notes |
|---|---|---|---|---|
| OpenLoops 2 gg→tt̄+Ng | ×27 | ×28 | ×35 | Diagrams 47 / 630 / 9,438 / 152,070 [L 1607.06934]. **Time per diagram 17 / 34 / 64 / 139 µs** [I from L]: linear in diagrams only approximately; per-diagram cost about doubles per gluon. |
| OpenLoops 2 ud̄→W⁻+Ng | ×17 | ×24 | ×28 | [L 1907.13071] |
| NJet ng | ×7.3 | ×28 | ×28 | Full colour [L 1209.0100] |
| NJet 2u+(n−2)g | ×11 | ×21 | ×27 | |
| Recola ud̄→W⁺(→e⁺ν)+Ng | | | ×22 | [L PoS RADCOR2015]; leg count includes the decay leptons |
| Recola uū→e⁺e⁻+Ng (EW+QCD) | | | ×19.5 | |
| MadLoop5 gg→tt̄+Ng, per helicity, integrand | ×16 | ×39 | ×167 | Per diagram: 9 / 13 / 39 / 437 µs [I from L]. The jump at 7 legs coincides with GB-scale code/RAM [I]. |
| MadLoop5, Ninja reduction | ×15 | ×15 | ×9 | About 78 µs per group at rank 6 [I from L] |
| GKW colour-dressed, per configuration | ×6 | ×6 | ×6 | [L 0911.1962] |
| BlackHat, per LC primitive and helicity | | | ×2.1 | [L 0803.4180]; 7→8 legs: ×1.9 |
| Weinzierl LC per evaluation | | n⁴ | | [L 1111.1733] |

**Reading [I].**

- Exact full colour with all helicities costs ×20–35 per gluon at 5→7 legs in every mature code. This is the combinatorial growth of colour orderings and helicities, partly offset by sharing.
- LC primitives grow only ×2.
- Per-diagram codes show super-linear constants at 7 legs, consistent with memory and code-size effects.
- The engine choice therefore buys a constant factor (sharing, helicity, colour-in-seed) and avoids pathological constants (code size, symbolic artefacts). No exact method changes the class.

### 3.4 Compatibility with pyAmpliCol's contract

| Route | NinjaRS per-family polynomial | Born weighting before reduction | UFO UV/R2 counterterms | Exact full colour + LC/NLC | Any precision | No diagram enumeration | Verdict |
|---|---|---|---|---|---|---|---|
| OpenLoops 2 schedule without OFR | Yes (OpenLoops1+OPP precedent) | Yes (seed) | Yes (tree counterterms) | Yes (seed; truncate by N_c degree) | Yes (generic arithmetic) | Only with recursion-native skeletons | **Core** |
| OFR | No | Yes | Yes | Yes | Hybrid | — | Reject |
| Recola forward currents | Yes after closure weighting | Only at closure | Yes | Yes | Yes | **Yes** | Generation model; secondary schedule |
| MadLoop5 / GoSam per-diagram | Yes (Ninja) | Yes | Yes | Yes | Yes | No | Oracle only (FeynKit) |
| Unitarity primitives (NJet/BlackHat) | Only via per-cut Laurent jets (50–100× cost [I]) | No (amplitude level) | No (FDH rational) | Yes | Quadruple | Yes | Oracle |
| Colour-dressed unitarity | No | No | No | Sampling only | — | Yes | Reject |
| Numerical integration (Weinzierl, LTD) | No | — | No (local subtraction) | LC in practice | — | Yes | Reject |

### 3.5 Generation cost

| Code | Example | Generation | Source |
|---|---|---|---|
| Recola | uū→e⁺e⁻uūg, 2→5 | 50 s | [L PoS RADCOR2015] |
| Recola | ud̄→W⁺ggg | 15–52 s | [L PoS; 1311.6662] |
| OpenLoops 1 | gg→W⁺W⁻bb̄ | 4 min, against more than 1 week for a FeynArts-based code | [L 1111.5206] |
| MadLoop5 | gg→tt̄ggg | about 2 days | [L 1604.01363] |
| MadLoop5 | gg→dd̄bb̄tt̄ | 15 h (600 MB code) | [L 1405.0301] |
| MadLoop5 | gg→5Z | 29 h | [L 1604.01363] |
| GoSam 1.0 | 2→2 | 71–1,225 s | [L 1111.2034] |
| HELAC-NLO | tt̄j off-shell | Skeleton generation parallelised per topology for 10× | [L 1609.01659] |
| Inherited pyAmpliCol | 7-leg multiquark | More than 10⁶ s projected | [assessment §3] |

**Recursion-native generation is the only demonstrated way to keep 2→5 generation at minutes.**

---

## 4. Synthesis

### 4.1 Which structural choices demonstrably scale to 2→5/2→6, in full and leading colour

**Full colour, per point.**

1. **Born weighting and helicity summation before reduction, with one reduction per denominator family.** This is demonstrated by OpenLoops 1+OPP (up to 2→5, tt̄+3j), OpenLoops 2+Collier, MadLoop5+Ninja and GoSam+Ninja.
   - It turns reduction into a small fraction of runtime: MadLoop5+Ninja gg→tt̄ggg 0.78 s against 36 s per helicity; pyAmpliCol measured 0.4–2 %.
   - It is necessary but not sufficient. MadLoop5 still pays linearly in helicities.
2. **Sharing of tree subcurrents across all loop attachments and counterterms**, with no per-diagram recomputation: OpenLoops subtrees, Recola currents, NJet cached Berends–Giele trees.
3. **Helicity summation inside the loop recursion.** OpenLoops 2's on-the-fly sum is 2–3× better than end summation plus merging. Recola's explicit-helicity coefficient time is 16× the single-helicity time for W+3j.
4. **Merging of open-loop states by (family, undressed suffix)** in OpenLoops 2 (about 2×), or by denominator-history prefix in Recola. Both make cost proportional to distinct states rather than diagrams.
5. **Colour carried outside the kinematic recursion.** Either the colour-summed Born dual is the seed (OpenLoops), or a fixed colour basis/flow is used with per-structure amplitudes contracted once (Recola, GoSam). Never per-diagram pairwise overlaps.
6. **Targeted higher precision**:
   - OpenLoops 2 hybrid precision costs a few % instead of 100×;
   - BlackHat recomputes only the unstable terms;
   - coarse per-point quadruple precision doubles the 7-gluon runtime (NJet).

**Leading colour.** Ordered primitives or colour-ordered skeletons with structurally pruned subleading terms scale polynomially (BlackHat, Rocket, Weinzierl, NJet LC).

- In production the subleading remainder is computed exactly but sampled less often (BlackHat W bb̄+jets, NJet five jets).
- The LC approximation alone is accurate to 1–3 % for W+jets but about 10 % off for W bb̄+jets.

**2→6.** Only codes with fewer coloured partons, or with colour and helicity sampling, have gone beyond 7 legs: GoSam+Ninja H+5j (8 legs), Recola and HELAC 2→7 with leptons, MadLoop per helicity. An exact full-colour 8-gluon virtual has no per-point timing in the literature.

### 4.2 Compatibility with NinjaRS and model counterterms

The OpenLoops-2 schedule *without* OFR, closed per family, delivers exactly NinjaRS's input:

- V_t(q) per family, Born-weighted and helicity-summed, rank ≤ N_t in a renormalisable Feynman gauge.
- The tensor layout is MadLoop/Ninja's, which NinjaRS adopts [L-s; V NinjaRS `fast/tensor.rs:1-6` per sibling].

Three independent groups (OpenLoops, MadLoop, GoSam) have shipped "numerator polynomial per topology → integrand reduction". OpenLoops+CutTools did so in a 2→5 phenomenology run (tt̄+3 jets, gg→tt̄ggg) [L 1607.06934].

Counterterms are tree amplitudes with exactly one counterterm vertex in every code reviewed (OpenLoops, Recola, MadLoop, HELAC, GoSam), so the model UV/R2 path needs no loop machinery.

**Precision.**

- The literature says most instabilities live in the reduction (Gram determinants [L 1710.11452]).
- This supports per-family DoubleFloat re-reduction from binary64 V_t. The analogues are OpenLoops 2's hybrid precision and MadLoop's quadruple-precision reduction with a double-precision integrand option [V MadLoopParams.dat:23–32].
- Coefficient-level cancellations, as in infrared regions and NJet's "not only Gram" remark, require a full-point fallback.

### 4.3 Pitfalls that killed or slowed other programs, mapped to pyAmpliCol

| # | Pitfall | Evidence | pyAmpliCol | Design rule |
|---|---|---|---|---|
| P1 | Per-diagram coefficients × per-helicity cost | MadLoop5: 36 s per helicity, coefficients ≫ reduction; per-diagram cost ×11 at 7 legs [L 1604.01363] | W4, W5 (54 %/95 % duplicate nodes; 87.5 % zero rows) | Shared Berends–Giele subcurrents; Born-seeded on-the-fly helicity sums; helicity filter |
| P2 | Diagram-based symbolic generation (days, GB of code and RAM) | MadLoop 15–25 h, 600–700 MB, 3.6–8.3 GB [L 1405.0301]; GoSam hours | W1, W2 (projectors 98.5 % of artifact bytes; one-CT tree enumeration 77 GB) | Recursion-native generation; numeric sparse contractions; no symbolic whole-process objects |
| P3 | Helicity-explicit forward recursion | Recola ×16 summed against single helicity [L 1311.6662] | Tree engine is helicity-explicit (fine for trees) | Sum helicities in loop dressing (OpenLoops 2), not at closure |
| P4 | Factorial colour bookkeeping | HELAC colour connections; NJet primitives (7g 56× de-symmetrised); colour-dressed needs sampling | W3 (pairwise exact overlaps; formal N_c fails at 7 gluons) | Colour in the seed via a fixed basis and integer N_c-Laurent Gram; combinatorial skeleton colour vectors [L 1310.0413, 1908.02695 show combinatorial decompositions] |
| P5 | Coarse rescue | MadLoop "rerun full point in quadruple precision"; NJet ×2 at 7g | W7 (two rescue implementations) | Per-family DoubleFloat re-reduction, then full point; cheap tests first (pole, direction, K-factor) |
| P6 | High-rank reduction instabilities | OpenLoops 1 needed COLI-vs-DD cross-checks; Ninja much more stable than CutTools/Samurai [L 1312.6678] | NinjaRS n = 7 fast-engine issues (dependency reading) | Per-lane rank lowering (next NinjaRS main); Gram-aware propagator ordering; family rescue |
| P7 | Arbitrary caps | NJet 13 legs; GKW n ≤ 12; Collier DD rank limits | `MAX_ADJOINT_GLUONS=12` [V] | No caps; streaming suffix-trie traversal for memory |
| P8 | LC used as an approximation | W bb̄+j LC about 10 % off [L 1712.05721] | LC/NLC are required outputs | LC/NLC by exact N_c-degree truncation; expose the exact remainder |
| P9 | Process-specific code size hurting runtime | MadLoop per-diagram cost jump at 7 legs [I]; GoSam FORM optimisation ×10 [L] | 394 MB PACBIN at 6 legs | Process-independent compiled kernels (per Lorentz structure and rank) plus a compact instruction table |
| P10 | Branchy per-point code blocks SIMD/GPU | CUDACPP needs lockstep for ideal SIMD [L 2303.18244]; virtuals are the bottleneck after GPU trees [L 2609.23500] | Rust executor already batched | Data-independent schedules; lane compaction for rescue |

### 4.4 Recommendation: the most promising structure for pyAmpliCol

**Each component, its precedent and its status**

| Layer | Choice | Literature precedent and measured benefit | Status |
|---|---|---|---|
| L0 Model | Keep UFO loop-model import and counterterm rules. Lower each UFO colour structure once to colour-flow/trace pieces with Idenso. | Recola2 colour-flow lowering from UFO [L-s 1705.06053 App. A] | Proven |
| L1 Trees | pyAmpliCol colour-ordered Berends–Giele currents (Rust recurrence builder) shared by the Born, loop attachments and counterterm trees. Extend `CurrentIndex` with a counterterm-insertion count of 0 or 1 [V `generation/dag_types.py:121-141`]. | Recola x-flag; MadLoop tree-matching; HELAC R2 skeleton; OpenLoops subtree recycling | Proven elsewhere; local extension |
| L2 Skeleton and family generation | Recola-type keyed recursion on the (n+2)-leg cut process: (cut species, ordered blocks as the leg→propagator map, mass history, colour-flow state, orders, CT flag). Canonical cut and orientation during recursion. Tadpoles and external self-energies rejected on the fly. Families keyed by (ordered denominators, masses), so species merge. **No diagrams.** | Recola skeleton generation in seconds to minutes; HELAC skeletons | Proven (Recola) |
| L3 Seeds and colour | Born dual w_c(h) = Σ_b A₀_b(h)* G_bc(N_c), with integer N_c-Laurent G. Skeleton seed U₀(s,h) = 2 Σ_c κ_s(c) w_c(h). LC/NLC by N_c-degree truncation and structural pruning of skeletons. Observable axis for correlations and spin sources. | OpenLoops U₀ [L-s 1907.13071 eq. initOL2]; MadLoop Λ_lb B* [L-s]; GoSam integrand contraction [L]; combinatorial loop colour decompositions [L 1111.4193, 1310.0413, 1908.02695] | Proven idea; new on Berends–Giele blocks |
| L4 Dressing | OpenLoops-2 schedule: on-the-fly helicity summation per segment, merging on (family, undressed suffix), dressing order "largest attached block first", depth-first suffix-trie streaming. Segment kernels compiled once per (UFO Lorentz structure, spins, input rank): SymJIT for binary64, Symbolica evaluators for DoubleFloat/arbitrary precision. | OpenLoops 2: merging about 2×, on-the-fly a further 2–3× [L 1710.11452]; MadLoop loop-vertex polynomials [L-s 1405.0301 App. C] | Proven schedule. **New:** over Berends–Giele blocks; generic kernels instead of unrolled Fortran |
| L5 Closure and reduction | V_t(q) += Tr V_N, written straight into NinjaRS `TensorNumerator` lanes. One reduction per (family, observable, order). Position-keyed or hash-keyed **master cache**. | OpenLoops1+OPP; OpenLoops2+Collier ("further significant speed-up" vs OFR); MadLoop5+Ninja; Ninja/Collier caches | Proven |
| L6 Counterterms | One-insertion tree amplitudes contracted with the same Born dual; δZ × Born for external fields | All codes | Proven |
| L7 Stability | Free pole test; same-coefficient direction re-reduction; per-family DoubleFloat re-reduction; full-point DoubleFloat/arbitrary precision. Rotation and scaling tests in validation. | MadLoop direction test; GoSam pole + rotation + K-factor; OpenLoops 2 hybrid precision; NJet scaling test; BlackHat partial recompute | Proven components |
| L8 Execution | One data-independent instruction stream per process; SIMD across phase-space lanes; lane compaction for rescue; GPU-portable structure-of-arrays layout | CUDACPP for trees [L 2303.18244]; no one-loop precedent | **New** |
| Options | Exact LC/subleading split outputs; helicity-subset and sampled virtuals; de-symmetrised colour sums (labelled non-pointwise); forward Recola-like schedule over the same kernels (validation and many-observable requests); an evaluation-first NinjaRS-callback oracle; the FeynKit per-diagram oracle for small processes | BlackHat, NJet, MG5_aMC, HELAC | Proven elsewhere |

**Why this, and not a straight port of one program**

- **Not OpenLoops as is.** Its generation enumerates FeynArts diagrams, which violates the no-diagram rule. At 7 gluons its objects are individual diagrams: about 3×10⁵, against 4.1×10⁴ colour-ordered skeletons over Berends–Giele blocks [L-s understand-recursion-literature §6]. It relies on per-process Mathematica/Fortran libraries, and OFR is incompatible with NinjaRS.
- **Not Recola as is.** Its helicity-explicit forward currents cost ×16–24 in helicity work (published table; sibling toy model). Its economics depend on reducing tensor integrals once and contracting per (colour, helicity), which NinjaRS cannot do. We keep its generation and keying, its runtime-skeleton library model and its colour lowering.
- **Not MadLoop or GoSam.** They are per-diagram, cost linearly in helicities (MadLoop) and take days to generate. MadLoop5+Ninja is the template only for the NinjaRS tensor interface and stability tests.
- **Not unitarity or colour-dressed.** They have their own reduction, sampling, FDH rational terms and per-cut cost with NinjaRS. NJet's full colour costs 171 s at 7 gluons.

**What is genuinely unproven (measure first)**

1. **The OpenLoops-2 schedule on colour-ordered Berends–Giele blocks.** Seeds become skeleton colour vectors with 2^{N_3g} flow terms for gluon loops.
   - The sibling estimate puts seeding below 2 % of dressing at 7 gluons [I].
   - Multi-quark full-colour flow states and U(1) (−1/N_c) pieces need tagging for LC/NLC [L-s understand-tree-engine (d)].
2. **Constants of a generic-kernel arena against unrolled Fortran.**
   - Recola is a generic library and is competitive at 5 legs.
   - SIMD lanes must recover the rest: CUDACPP shows ×8 in double on AVX-512 for lockstep trees.
3. **NinjaRS heptagon rank-7 stability in binary64.** It is fixed only on an unpushed integration branch (journal dependency reading). EW needs heptagons with complex masses, because leptonic decays add a loop propagator at NLO EW [L 1907.13071 §5].
4. **Memory of merged objects** at rank 7 × helicity rows × observable axis. Streaming the suffix trie depth-first is the mitigation. Colour-dressed unitarity hit its n limit through memory [L 0911.1962].
5. **Correlated virtuals.** Colour and spin correlators multiply dressing by N_obs on the Born-seeded schedule (the seed is linear).
   - Recola2 lists "Complete NLO correlations (spin-colour/colour/spin + loop-induced)" [V `recola2-2.3.0/CHANGELOG`]. That it serves them from structure-dressed amplitudes is an inference [I].
   - Keep the forward schedule over the same kernels for correlator-heavy requests (MASTER_PLAN §2.14 schedules 1/3).

**Expected outcome [I]**

| Process | Expected binary64, one thread | Comparison |
|---|---|---|
| gg→gggg | about 0.2–2 s | OpenLoops-class about 1–3 s; NJet 3.1 s per evaluation (2012) |
| gg→ggggg | about 10–30 s | NJet about 85 s per evaluation (2012); OpenLoops-class about 40–80 s |

- The sibling toy model gives 3–10 s for gg→ggggg if per-operation efficiency matches OpenLoops [I].
- "Faster than MadLoop" is assured at 2→5, since MadLoop's helicity sum is of order 10³ s per point.
- "Within 1.3× of OpenLoops" is plausible but **not** assured. It depends on items 1–2 above.

**Measurement plan (smallest experiments that decide the open points)**

1. **Prototype dressing (L4) on gg→ggg and gg→gggg** with real kernels and seeds. Measure:
   - MAC count and time per point against the toy model;
   - the merge factor and helicity-sum factor against the OpenLoops 2 published 2× and 2–3×;
   - V_t equality with the inherited per-diagram numerators to 1e-13.
2. **Same-host provider campaign** (OpenLoops ppjjjj/gggggg, ppttjj, pplljjj, Recola, MadLoop) per MASTER_PLAN §5.8, after the OpenLoops audit. This converts §3.2 into gates.
3. **NinjaRS heptagon stability** on physical V_t from item 1 (gg→ggggg sample families). Measure the rescue fraction against NJet's 8.6 % with double-precision unitarity.
4. **Seed and colour cost** for uu~→ggggg and gg→tt̄gg, including multi-quark U(1) tagging, LC/NLC truncation and the observable axis with 21 colour correlators.

---

## 5. What the literature cannot settle

| Question | Why it is open | How to settle |
|---|---|---|
| Do Berends–Giele blocks beat OpenLoops subtrees *after* merging? | OpenLoops kept subtrees because colour factorises per diagram [L-s 1710.11452 §2.2]. The Berends–Giele advantage (7× fewer objects at 7g) is a count, not a time. | Measure item 1 |
| Generic kernels against unrolled code | No published one-loop code uses SIMD lanes | Measure item 1 with batch 1/8/32/128 |
| Real competitor timings for gg→gggg / gg→ggggg / uu~→ggggg | OpenLoops publishes no multi-gluon timings; NJet numbers are 2012 hardware with ×2 tests | Local campaign |
| EW heptagons with complex masses | No program publishes EW timings for gg→tt̄e⁺e⁻g; OpenLoops says leptonic decays make EW much more expensive | Measure after CMS/admission work (assessment R6/R7) |
| Best default for many-observable requests | Recola (amplitude level) and OpenLoops (seed) differ; no published comparison for correlated virtuals | Toy break-even (sibling: N_obs ≈ 15–20 at 7g [I]) to be measured |

---

## Appendix A. Primary sources read in this session (by program)

All are [L] unless marked.

| Program / topic | Sources (what was read) |
|---|---|
| OpenLoops | 1111.5206 (full method and timing text); 1710.11452 (§3 speed-up statements, §6.2.1 benchmarks, Gram correlation); 1907.13071 (tab:perfnew and text); 1907.13624 (tab:Ndiagrams, OpenLoops 2/hybrid-precision statements); 1607.06934 (diagram counts) |
| Recola | 1211.6316 (timing remark); 1311.6662 (Table 1); PoS(RADCOR2015)023 (performance list, hardware footnote; PDF text extracted locally); 1605.01090 (program flow); 1506.07448 (diagram counts); 1611.02951 (searched, no timing); Recola2 2.3.0 source [V] |
| MadLoop / MG5_aMC | 1405.0301 (§2.4.3 helicity sampling, App. tab:MLperf); 1604.01363 (timing tables); 1804.10017 (reduction cascade); MG5_aMC 3.7.0 source [V `MadLoopParams.dat`, `loop_helas_objects.py:652-683`] |
| GoSam | 1111.2034 (timings, colour algebra, reduction options); 1404.7096 (new features, rescue, Ninja); 1312.6678 (tab:summary, bad points); 2507.23549 (GoSam-3 performance and rescue) |
| Reduction libraries | 0711.3596 CutTools; hep-ph/0609007 OPP (fetched, not re-read); 1006.0710 Samurai; 1008.2441 tensorial reconstruction; 1203.0291 Ninja method (fetched); 1403.1229 Ninja library (cache); 1604.06792 Collier (cache, running time) |
| HELAC | 0903.4665 (colour connections, timing); 1110.1499 (reweighting); 1609.01659 (2→7 setup); 1012.4230, 1002.4009 (grep only) |
| NJet / NGluon | 1209.0100 (§2.2, §5 tables); 1011.2900 (runtime section); 1201.1187 (timing table); 1312.7140 (LC/subleading); 1309.6585 (full-colour 5 jets); 2002.07516 (NJet 5-jet cost); NJet3 [L\* Bitbucket via search] |
| BlackHat | 0803.4180 (TimingTable); 0902.2760 (LC W+3j); 1112.3940 (full colour 4j); 1304.1253 (W+5j LC, stability); 1712.05721 (LC 10 % for Wbb̄) |
| Unitarity line | 0810.2762, 0901.4101 (Rocket W+3j); 1105.4319 §10 (review); 0911.1962 (colour-dressed tables); 2009.11957 (Caravel) |
| Colour decompositions | 1111.4193 (Ita–Ozeren); 1310.0413 (Reuschle–Weinzierl); 1612.04366, 1908.02695 (Ochirov–Page), abstracts |
| Numerical integration | 1111.1733 (7 jets, n⁴); 1205.2096; 1010.4187; 1912.09291 [L\* abstract] |
| GPU / SIMD / surrogates | 2303.18244 (CUDACPP); 2311.06198 (PEPPER); 2106.06507 (BlockGen); 2609.23500 (MadVfold: virtuals become the bottleneck); 2102.00773 (helicity recycling, fetched); 2206.14831 |
| Reviews | 1006.5594 (Maître: precision strategies); 1310.3214 (Ossola, fetched) |
| [L-s] via siblings | 0805.2152, 0806.3467, 0902.0094, 0905.1005, 1103.0621, 1507.00020, 1705.06053, 1711.07388, hep-ph/0002082, hep-ph/0607057 |

## Appendix B. Reproduction

```sh
# arXiv e-print fetcher used for this survey (writes to landscape/src/<id>/x/)
/shared/vhirschi/t3code/pyamplicol/workspace/venv/bin/python -I \
  /shared/vhirschi/t3code/pyamplicol/workspace/tmp/engine-v2/landscape/scripts/fetch.py <arXiv-id> ...
# Timing lines are found with: LC_ALL=C grep -a -n -i -E "GHz|seconds|[0-9] ?ms\b|time per" <dir>/*.tex
# The PoS PDF text was extracted with a FlateDecode inflater into
#   workspace/tmp/engine-v2/landscape/pdf_recola_pos/pos235_023.txt :
/shared/vhirschi/t3code/pyamplicol/workspace/venv/bin/python -I \
  /shared/vhirschi/t3code/pyamplicol/workspace/tmp/engine-v2/landscape/scripts/pdftext.py \
  /shared/vhirschi/t3code/pyamplicol/workspace/tmp/engine-v2/landscape/pdf_recola_pos/pos235_023.pdf
```
