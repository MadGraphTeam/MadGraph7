# Engine v2: exploiting leading- and next-to-leading-colour structure in one-loop Born–virtual interference

Date: 2026-10-20. Research report; the repository was not modified. Host repo: `pyamplicol`, branch `loop`, HEAD `4faafcad` (read-only).

Companion reports in this directory (this one does not repeat them):

- `understand-recursion-literature.md`: recursion families, NinjaRS lifting, the cut-restricted lift, master cache.
- `understand-algorithms.md`: OpenLoops/Recola/MadLoop survey and the recommended Born-seeded core.
- `understand-tree-engine.md`: the tree engine as host.
- `understand-contracts.md`: contracts the new engine must satisfy.

**Evidence labels**

| Label | Meaning |
|---|---|
| **[V]** | Verified here: code read (`file:line`), or a probe/count run in this session. |
| **[L]** | Literature, read in the arXiv LaTeX source in this session (arXiv number and section given). |
| **[L\*]** | Literature seen only through an abstract or search summary. Plausible, not verified. |
| **[I]** | Inference, estimate or design judgement. |

Paths are relative to `pyamplicol/src/pyamplicol/` unless written in full. Scratch scripts and logs are in `workspace/tmp/engine-v2/lcs/` (this session) and `workspace/tmp/engine-v2/lc/` (an earlier, interrupted attempt at this task, whose probes were re-run here). Paper sources are under `workspace/tmp/engine-v2/papers/`, `papers-lc/` and `lcs/papers/`.

---

## 0. Answers first

### 0.1 What leading colour can save at one loop, and why it is less than at tree level

At tree level the dominant cost of exact colour is the colour sum: (n−1)! ordered amplitudes are cheap (Berends–Giele, O(n⁴) each) but the full-colour matrix is dense. LC makes that matrix diagonal, which is why LC is so effective in the tree engine.

At one loop the colour sum is negligible. The cost is the loop numerators and their reduction. The colour-ordered decomposition (§2) shows that **for an all-gluon process, full colour (FC) and LC need exactly the same gluon-loop kinematic objects**: the (n−1)!/2 planar primitive amplitudes A_{n;1}(σ). FC reaches the subleading colour structures through permutation sums of the same primitives (Bern–Dixon–Dunbar–Kosower COP formula; Del Duca–Dixon–Maltoni ring form). So the LC saving can only come from contribution classes that are absent at LC:

1. **Closed quark loops.** They are degree P−1 under MASTER_PLAN's fixed-N_f convention (`MASTER_PLAN.md:580`), so they are NLC, not LC.
2. **For quark lines, every primitive except the "left-turner, adjacent q̄q" one.** These are A^R, the mixed primitives with gluons on both sides of the quark line, and arcs with coloured attachments.
3. **The 1/N_c parts of counterterms.** These are the T_F N_f parts of β₀ and δZ, heavy-quark logarithms, the −1/(2N_c) part of C_F, and the δδ part of the four-gluon R₂ vertex.
4. **U(1)/Fierz-singlet pieces** of the colour-flow representation.

The verified counts (§6, [V] `lcs/lc_structure_counts.py`), with dressing measured by a rank-weighted open-loop-state proxy, are:

| Process | LC orderings | Planar families per ordering | LC / FC skeletons | LC / FC dressing proxy | LC / FC NinjaRS families |
|---|---|---|---|---|---|
| gg→gggg | 60 | 51 | 0.25 | 0.25 | 0.33 |
| gg→ggggg | 360 | 113 | 0.25 | 0.25 | 0.33 |
| uu~→ggggg | 120 | 113 | 0.13 | 0.15 | 0.47 |
| uu~→e⁻e⁺ggg | 6 | 51 | 0.07 | 0.08 | 0.32 |
| uu~→e⁻e⁺dd~g | 4 | 36 | 0.05 | 0.08 | 0.45 |
| gg→tt~gg | 24 | 51 | 0.16 | 0.17 | 0.32 |

Expected overall LC/FC cost per point [I]:

- **About 0.25–0.35** for the all-gluon processes. This is the whole of the quark-loop work, with b and t loops counted as separate mass classes.
- **About 0.1–0.2** for one quark line.
- **About 0.05–0.15** for vector boson + jets.

Literature anchor: BlackHat found subleading-colour virtuals for W+3 jets to cost about 20× the LC virtual per point. It sampled them on 10⁵ points against 2×10⁶ for LC, at comparable total time (arXiv:0907.1984 §2.6 "Phase-Space Integration") [L].

**The inherited engine gets nothing from LC today.** It computes everything and projects afterwards. Measured LC/FC per-point ratios are 1.10 (DY), 0.95 (uudd), 0.99 (tt), 0.95 (gg) (`pyamplicol/docs/performance_reports/loop/data/colour-measurements.json`) [V].

**NLC costs essentially the same as FC** (§5). It needs all quark-loop primitives at P−1. It needs A^R and the mixed primitives at P−2, through the subleading partial amplitudes.

### 0.2 The structure to land on

The structure is **ordering-major generation, family-major reduction, colour in the seed**. All of it is generic, derived from colour flows and model vertex content, and uses no process-specific code.

1. **LC orderings = the tree engine's LC colour sectors.** Each LC Born flow (a trace, an open string q→σ→q̄, or a product of crossed strings) closes into exactly one cyclic word. Colour singlets sit in the colourless gaps of the fermion lines they couple to (§3, §4). The tree plan already enumerates these words (`color/plan_build.py:17-129`, reflection folding at `:255`) [V].
2. **LC skeletons of an ordering = the compositions of its cyclic word**, i.e. the pinches of that ordering's n-gon parent. Each has a *unique* LC loop routing: gluon/ghost propagators everywhere, except *bare* quark arcs in colourless gaps. A pure-gluon process has 2ⁿ−2n−1 of them per ordering (113 at n = 7) [V]. No Feynman diagrams are enumerated.
3. **Generic structural pruning (the "face rule", §4.1).** A planar skeleton reaches the leading degree P iff:
   - (a) its outer colour flow is a leading Born flow;
   - (b) the loop face is bounded only by colour lines: gluon or ghost propagators, or quark arcs without coloured attachments that lie behind a colourless gap of the word;
   - (c) it carries no U(1)/Fierz-singlet grade.

   Everything else is ≤ P−1 (closed quark loops) or ≤ P−2. The rule was verified against exact SU(N_c) overlaps from pyAmpliCol's own colour engine for every class (§4.1). A colour-flow recursion applies it locally to open-loop states, using the tree engine's LC segment table extended by the cut labels (§4.2).
4. **Diagonal LC seeds.** The Born-dual weight of every skeleton of ordering σ is w_σ(h) = N_c^P κ A₀(σ,h)\*: one product, no Gram matrix. These are the Born sector amplitudes that the tree runtime already holds. They must be built in the strict U(N_c) leading-flow convention: the tree engine's numeric 1/3 Fierz-singlet weights must be graded or excluded (§7.3).
5. **DDM/Kleiss–Kuijf canonical blocks (optional, large win, LC and FC alike).** Pure-gluon tree blocks are attached in their (|B|−1)! canonical orders instead of |B|!. A canonical skeleton then contributes to 2^{|B|−1} Born orderings with ±1 signs; this was verified exactly (§4.4). The effect at 7 gluons:
   - skeletons fall 40,680 → 8,011;
   - the dressing proxy falls by 3.2× (8.5×10⁶ → 2.7×10⁶) [V].
6. **Family-major, Born-weighted, helicity-summed NinjaRS reduction plus a master cache.** For gg→ggggg at LC, 40,680 skeleton numerators merge into 4,707 family reductions. A per-ordering "parent n-gon" reduction (cut-restricted lift) would do **3.6× more** expansion work (541,800 vs 150,570 family–cut expansions) and lose 1.5 digits [V counts, §8]. The family-tree view is therefore a generation and verification tool, not the production reduction.
7. **Counterterms at LC are diagonal colour-ordered tree insertions**, using only the N_c-leading part of each CT vertex (§4.3).
8. **Checks.**
   - The LC IR poles depend only on colour-adjacent pairs (EKMZ arXiv:1105.4319 §6.4.2) [L]. This gives a per-ordering analytic check.
   - The inherited engine's LC projection is a correct but slow oracle on the small controls it already validates (up to 2→4).
9. **FC/NLC on the same machinery.** Add the skeleton classes that LC prunes. Seeds become Born-dual weights from an exact flow Gram: `color/connections.py:492` takes about 1.5 ms per 7-gluon overlap [V, `lcs/rerun_lc_colour_timing.log`]. Use the tree DDM/FFT planner where certified. NLC is FC with truncated seeds.
10. **Optional observable modes from the literature.** These are not defaults, because they change the per-point quantity:
    - **De-symmetrised LC** (NJet). Final-state Bose symmetry reduces the 360 orderings of gg→ggggg to 3 and the 120 of uu~→ggggg to 1. The result is valid only after symmetric phase-space integration.
    - **LC-ratio** (BlackHat W+3j).
    - **Separately requestable LC and FC−LC pieces**, for colour-expansion sampling.

### 0.3 Corrections to the sibling reports

- `understand-recursion-literature.md` §0.4 says "LC never reduces the number of NinjaRS reductions for all-gluon processes". That holds only for massless loops. With b and t loops kept as separate mass classes, FC has 3 × 4,707 = 14,121 families at 7 gluons and LC has 4,707 [V].
- The same section labels the uu~→ggggg LC counts (2,712 families, 13,560 skeletons) as [L\*] assumptions. They are now verified by the generic face-rule enumeration [V]. The enumeration adds the multi-quark and top cases (§6).

---

## 1. Definitions used

- **MASTER_PLAN convention.** Per squared-order sector, a structural leading degree P is chosen, independent of helicity, ε coefficient and numerical cancellations. LC keeps degree P. NLC keeps P, P−1 and P−2, so odd suppression powers are included. N_f has colour degree 0. Everything is combined exactly before projection, with the same convention for bare, UV and R₂ contributions (`MASTER_PLAN.md:561-584`, `:33`) [V].
  - The inherited implementation is `loops/color_projection.py:26-132` (`StructuralColorProjection`, `prepare_color_projection`) and `loops/physical_color.py:257-416` [V].
  - For gg→gg the Born degree is 4 and the virtual degree P = 5. Quark rows sit at degrees 4 and 2; gauge/ghost rows at 5 and 3 (`MASTER_PLAN.md:2196-2197`). "LC removes fixed-Nf degree4 UV/field terms and NLC retains them" (`:2262`) [V].
- **Literature convention differs.** BlackHat, NJet and Ita–Ozeren take N_c → ∞ with **N_f/N_c fixed**:
  - "suppressed by at least one power of either 1/N_c² or n_f/N_c" (arXiv:0907.1984 §2.2);
  - "N_c ≫ 1 and n_f ≫ 1 with N_c/n_f fixed" (arXiv:1111.4193 §4.3) [L].

  So literature LC **includes** the (n_f/N_c) A^{L,[1/2]} quark-loop primitive, while MASTER_PLAN LC excludes it. Every literature LC timing therefore *understates* the saving available under MASTER_PLAN.
- **Structural degree of a one-loop QCD sector.** P = P_B + 1, because a gluon loop adds one closed colour-index loop. For EW loops in a QCD-Born sector, P = P_B, since no index loop is gained [I].

---

## 2. Colour decompositions: which kinematic objects exist

### 2.1 n gluons [L]

- **Trace decomposition.** The one-loop amplitude is a sum over the single-trace structure N_c Tr(T^{σ₁}⋯T^{σₙ}) and the double traces Tr(⋯)Tr(⋯) (arXiv:hep-ph/9403226 §2, eq. `ColorDecomposition`).
  - "the leading contribution for large N_c comes from A_{n;1}; the subleading-in-N_c corrections A_{n;c>1} are down by a factor of 1/N_c²".
  - "For internal particles in the fundamental representation, only the single-trace color structure (c=1) would be present, and the corresponding color factor would be smaller by a factor of N_c" (same section).
- **COP formula** (hep-ph/9403226 §7, eq. `sublanswer`): A_{n;c>1}(1..c−1; c..n) = (−1)^{c−1} Σ_{σ∈COP{α}{β}} A_{n;1}(σ). The paper derives it from string-based rules and outlines a field-theory argument. DDM note its close analogy to the tree Kleiss–Kuijf relation, with legs 1 and n "sewn together" (hep-ph/9910563 §3).
- **Ring form** (Del Duca–Dixon–Maltoni, arXiv:hep-ph/9910563 §3, eq. `NewLoopColor`):
  𝒜ₙ^{1-loop} = gⁿ Σ_{σ∈S_{n−1}/𝓡} [Tr(F^{σ₁}⋯F^{σₙ}) A^{[1]}_{n;1}(σ) + 2n_f Tr(λ^{σ₁}⋯λ^{σₙ}) A^{[1/2]}_{n;1}(σ)].

  "Thus the number of linearly independent subamplitudes is (n−1)!/2." The same form is in EKMZ arXiv:1105.4319 §6.1, eqs. `1loopglu` and `nfpart`.
- **Consequence.** FC and LC need the same (n−1)!/2 gluon-loop primitives. FC adds only the quark-loop primitives and a different colour weight. MASTER_PLAN NLC (P−1 quark loops, P−2 double traces and off-diagonal traces) needs no new gluon-loop object.

### 2.2 One quark line plus gluons [L]

- **Primitives** (Bern–Dixon–Kosower, arXiv:hep-ph/9409393 §3):
  - A^{L,[J]} and A^{R,[J]} with "the fermion line from 1q̄ to 2q passes to the left (L) or to the right (R) of the loop";
  - J = 1, ½, 0 separates closed fermion and scalar loops;
  - reflection A^R(w) = (−1)ⁿ A^L(w^T) (eq. `flipsym`);
  - A^{R,[1/2]}(1q̄,2q,…) = 0, because only tadpoles and external bubbles remain (eq. `floopvanish`).
- **Leading partial amplitude** (§4, eq. `Anoneformula`):
  A_{n;1}(1q̄,2q;3..n) = A^L(1q̄,2q,3..n) − (1/N_c²) A^R(1q̄,2q,3..n) + (n_f/N_c) A^{L,[1/2]} + (n_s/N_c) A^{L,[0]}.

  "Inspecting color flows in the double-line formalism … this amplitude receives contributions only from diagrams where the two fermions are adjacent." The A^R diagrams "would not contribute at all to A_{n;1}, because of their wrong color flow, were it not for the −1/N_c term in the SU(N_c) Fierz identity" (§4).
- **Subleading partial amplitudes** A_{n;j>1} are COP sums of mixed primitives A^{L}(σ(1q̄,2q,3..n)), with gluons on both sides of the quark line (eq. `qqsublanswer`; DDM §3, eq. `qqsublanswer`).
- **The interference formula for q̄qggg** (hep-ph/9409393 §2) shows the degree structure explicitly. Relative to the leading (N_c²−1)² A_{5;1}(σ) term:
  - (N_c²+1) A_{5;1}(reversed) and (N_c²−2) A_{5;4} enter at relative 1/N_c², i.e. P−2;
  - the A_{5;3} terms enter at relative N_c⁻⁴, i.e. P−4.
- **DDM ring form for q̄q** (hep-ph/9910563 §3, eq. `LoopColorqqNew`):
  Σ_{p=2}^{n} Σ_σ (λ^{x₂}λ^{σ₃}⋯λ^{σₚ}λ^{x₁})_{i₂ī₁} (F^{σ_{p+1}}⋯F^{σₙ})_{x₁x₂} A^{R,[1]}(1q̄,σ_{p+1}..σₙ,2q,σ₃..σₚ) + n_f terms.

  "Most of the subamplitudes A^{R,[1]} do not contribute at leading order … (Only those of the form A^{R,[1]}(1q̄,…,2q) do.)" The LC term is p = 2: no gluon is attached to the loop's fermion arc. Ellis–Melnikov–Zanderighi: "In the leading color approximation we retain only the p=2 term" (arXiv:0901.4101 §4) [L].
- **Primitive counts.**
  - NJet: N^{[m]}_pri = ½ n_q (n−1)! and N^{[f]}_pri = (n−1)! (arXiv:1209.0100 §2.2, eq. `colorM`).
  - Kälin's minimal bases (arXiv:1712.03539 §3): (n−1)!/k! for a closed quark loop, and 2^{2k−1}(n−1)! k!/(2k)! for a mixed/gluonic loop with k quark pairs.
  - For uu~→ggggg (n = 7, k = 1): 720 mixed primitives in FC, against 5! = 120 LC primitives.

### 2.3 Several quark lines [L]

- **Ita–Ozeren** (arXiv:1111.4193 §2.4). Primitives are first "coarse": quarks are treated as adjoints and the coefficients of N_c Tr(…) are taken. They are then split by fermion routing: every fermion line is a left- or right-turner (L, R, LL, LR, …), read off "by cutting a single loop propagator". Primitive amplitudes "can be interpreted as partial amplitudes of a theory with non-standard gauge groups and colour charges", which is the gauge-invariance argument.
- **Reuschle–Weinzierl** (arXiv:1310.0413 §2.2). Admissible cyclic words are those with "no crossed fermion lines" on the annulus; e.g. (q̄₁^R q₁^R q̄₂^R q₂^R) is non-planar. The method is purely combinatorial (shuffle relations, no linear solve). §4 gives U(1)-gluon exchange and "loop closing with a U(1) gluon" as tree-like operations, which is where the 1/N_c pieces live.
- **Kälin** (arXiv:1712.03539 §4.1). "Cutting through the quark loop of the Mario world representation of a color factor directly recovers a tree level color factor with n+2 external particles … Conversely, closing the base quark line of a tree level color factor produces a loop-level color object."

  This is exactly the open-loop picture: the loop colour factor is the colour factor of the (n+2)-point tree with the two cut legs closed.

### 2.4 Colour singlets [L]

- BDK e⁺e⁻→q̄qgg (arXiv:hep-ph/9708239 §§2.2–2.3) keeps the vector boson in the colourless gap between the quark and antiquark of the line it couples to.
- BlackHat W+3j (arXiv:0902.2760; 0907.1984 §2.2): "In the leading-color primitive amplitudes, the W boson is between the q and q′ external legs, with no other partons in between." Subleading primitives have "an external gluon, or a gluon splitting to a Q̄Q pair, … emitted from the q–q′ line, between the W boson and one of the external quarks", or a closed fermion loop.

### 2.5 Counterterms [L/V]

- HELAC-1Loop computes R₂ and UV as "tree-order contributions, with given extra Feynman rules … including only one of these extra vertices" (arXiv:0903.4665 §2) [L].
- The supplied loop_sm CT colour origins are tabulated in `workspace/reports/loop-general-nc-r2-input-audit.md`, together with the general-N_c HV four-gluon R₂ rule (Lang–Pozzorini–Zhang–Zoller, arXiv:2007.03713 eqs. 5.30–5.38) [V]:
  - single-trace part: C_C = Tr(T¹T³T²T⁴)/T_F × (r_CI V_I + r_CII V_II), with r_CI = −8C_A/3 − (10/3)T_F N_f;
  - double-δ part: C_B = T_F δ₁₂δ₃₄ × (−C_A/(3N_c)).

---

## 3. Why LC interference is diagonal in orderings, and the LC formula

### 3.1 Argument [L + I]

Work in the U(N_c) colour-flow basis {|c⟩}: products of Kronecker deltas, single cycles for gluon traces, open strings for quark lines. The Gram matrix is ⟨c|c′⟩ = N_c^{#cycles(c⁻¹c′)} (`color/adjoint_kernel.py:1-9`, `:36-62` implements this cycle counting [V]). It is maximal only on the diagonal.

- Two different single-trace gluon flows overlap at ≤ N_c^{n−2}, i.e. relative 1/N_c². "Cross terms between different color flows yield monomials N^{n−m}, where m is even" (Maltoni–Paul–Stelzer–Willenbrock, arXiv:hep-ph/0209271 §3) [L].
- Two quark-line flows differing by one transposition of antiquark endpoints overlap at relative 1/N_c. This can contribute only if *both* flows carry a leading Born amplitude: identical flavours, or a singlet-exchange Born.

Verified with pyAmpliCol's exact overlaps (`lc/lc_colour_probe.py`, re-run log `lcs/rerun_lc_colour_probe.log`) [V]:

| n | ⟨Tr σ\|Tr σ⟩ | ⟨Tr σ\|planar gluon ring of σ⟩ | ⟨Tr σ\|planar ring of τ≠σ⟩ (max) |
|---|---|---|---|
| 4 | N_c⁴ + … | **N_c⁵** + … (degree P) | N_c³ (P−2) |
| 5 | N_c⁵ + … | **N_c⁶** + … (degree P) | N_c⁴ (P−2) |
| 6 | N_c⁶ + … | **N_c⁷** + … (degree P) | N_c⁵ (P−2) |

A planar LC skeleton has colour N_c|c(s)⟩ + (lower), where c(s) is its outer flow. Its overlap with the Born is therefore N_c^{P} δ_{c,c(s)} + O(N_c^{P−1}). The O(N_c^{P−1}) term is present only in the identical-flavour transposition case; otherwise it is O(N_c^{P−2}).

### 3.2 Precise LC statement

For a squared-order sector with structural degree P and coefficient k ∈ {0, −1, −2}:

V^{LC}_k = 𝒩 Σ_h Σ_{c ∈ 𝓕_LC} κ_c N_c^{P} · 2 Re[ A₀(c,h)\* · A^{LC}_{1,k}(c,h) ],

where:

- 𝒩 is the external average and symmetry normalisation;
- κ_c are the generator-normalisation constants of the LC Born. OpenLoops writes the LC colour matrix as K_ij = δ_ij 2^{−n} N_c^{n+m} + subleading (arXiv:1907.13071 §4.5, eq. 4.36) [L];
- 𝓕_LC is the set of leading Born flows, up to reflection for gluon traces;
- A^{LC}_1(c) is the sum over all skeletons that pass the face rule (§4.1) with outer flow c, after reduction and masters, plus the LC CT trees in flow c.

What 𝓕_LC is for each Born:

| Born content | Leading flows 𝓕_LC | LC loop words (cyclic) |
|---|---|---|
| n gluons | all traces, (n−1)!/2 up to reflection | the trace itself |
| one q̄q line + k gluons | strings (q σ q̄), k! | (q̄, q, σ): q̄ and q adjacent |
| two distinct-flavour lines joined by QCD | crossed strings (q₁→…→q̄₂)(q₂→…→q̄₁), gluons distributed in all ways | (q₁…q̄₂, q₂…q̄₁) |
| plus colour singlets | unchanged | each singlet in a colourless gap q̄_f \| q_f of a line it couples to |

**Singlets.** A singlet changes neither 𝓕_LC nor P. If it is attached to the loop, it must sit on a bare quark arc: a vertex correction. A singlet attached to a fermion arc that also carries gluons is A^R-like, ≤ P−2. A singlet attached to a closed quark loop is P−1. These are the BDK and BlackHat statements in §2.4 [L], reproduced by the enumeration [V].

**EW loops.** A colour-singlet boson in the loop adds no index loop. So P = P_B, and the LC EW skeletons are those whose colour structure equals a leading Born flow with no 1/N_c [I]. With mixed QCD and EW orders, P is the maximum over all Born×loop pairings in the sector. A pairing that never reaches P drops out at LC entirely. This is structural, so it is computable before dressing [I].

**Pole check.** For n-gluon primitives (N_f = 0) EKMZ give:

m_n^{(1)} = −Σ_i [1/ε² + (1/ε)(11/(3n) + L_{i,i+1})] m_n^{(0)},

so only colour-adjacent pairs appear (arXiv:1105.4319 §6.4.2) [L]. This is an analytic per-ordering check of A^{LC}_{1,−2} and A^{LC}_{1,−1}.

---

## 4. What is zero at LC, and how a colour-flow recursion prunes it

### 4.1 The face rule, and its verification

In the 't Hooft double-line picture of a planar one-loop skeleton, the loop face yields the extra N_c iff it is bounded only by colour lines. A quark propagator carries one colour line. That line can lie on the face side only if no coloured current attaches to that arc, and only if the arc sits behind a gap of the outer word that carries no colour line. This gives the classification below. All rows were checked by maximising exact overlaps over *all* Born flows with pyAmpliCol's engine (`color/connections.py:492`): logs `lcs/rerun_lc_colour_probe.log` and `rerun_lc_colour_probe2.log` [V].

| Skeleton class | Example colour (cut pair c) | Max degree over Born flows | Kept at |
|---|---|---|---|
| Planar gluon/ghost ring of a leading word | Tr(c σ c) → N_c Tr(σ) | **P** | LC |
| q̄q adjacent, gluons on the gluon side (A^L, p = 2) | (q σ c c q̄); quark cut: (q σ q̄) × Tr() | **P** | LC |
| Closed quark loop (any mass class) | Tr(σ) with no N_c | P−1 (odd) | NLC |
| Two-sided ring (cut pair not adjacent, double trace) | Tr(c 1 2 c 3…n) | P−2 | NLC |
| A^R: gluons attached to the loop's fermion arc | (q c σ c q̄) | P−2 | NLC |
| Abelian vertex correction across an emission | (q c g c …) | P−2 | NLC |
| Gluon on a quark arc inside a 4-quark box | (1 a g b 2)(3 b a 4) | P−2 | NLC |
| 4-quark box, uncrossed colour | (T^aT^b)⊗(T^aT^b) | P−2 | NLC |
| 4-quark box, crossed colour; gluon outside or on a gluon arc | (T^aT^b)⊗(T^bT^a) | **P** | LC |
| Flow overlap differing by one quark transposition | (1→4)(3→2) vs (1→2)(3→4) | P−1 (identical flavours only) | NLC |

Each row of the generic enumeration (§6) carries one of these tags. "Fermion arc behind a coloured gap" is the colour-ordered half of a bubble or vertex correction that sits on the wrong side. An example is the m = 2 skeleton {V, q} | {g g g q̄} with its quark propagator on the q|g side rather than in the q̄|q gap. It was found during the enumeration and is ≤ P−2 by the same rule [V].

### 4.2 Pruning inside the recursion, before any numerics

The open-loop current is a colour-ordered current of the (n+2)-label tree with the two cut legs as extra labels (Kälin §4.1; HELAC-1Loop: the planar case needs only "the mere addition of two more color lines", arXiv:0903.4665 §2 [L]). Its LC colour state is fully described by:

- the ordered coloured word segment attached since the cut;
- the loop-line species history;
- a face flag (closed so far / open);
- a grade counter (number of 1/N_c factors).

Rules, all local and process-generic:

1. **Word admissibility.** The attached word segment, including the cut label at its end, must be a contiguous segment of a rotated LC sector word. The tree engine already builds exactly this table for trees: all contiguous sub-words of every sector word, rotated so the sink label is last (`generation/dag_color.py:186-243`), and it tests membership in `_shared_lc_ordered_proposed_labels` (`:1041-1076`) [V]. For loops, add the entries with a cut label at either end.
   - Pure-gluon processes are never pruned by this rule: every ordered sequence is a segment of some cyclic word.
   - Quark processes lose every word with a gluon in a colourless gap.
2. **Arc rule.** When the loop line is a quark, attaching a current that carries colour (a gluon block) sets face = open, so the skeleton is pruned at LC. Attaching a singlet is allowed. A quark propagator may be created only in a colourless gap: an antiquark followed by a quark in the leading reading direction.
3. **No closed quark loops at LC.** A quark cut line with no external fermion entering is pruned.
4. **No U(1)/Fierz-singlet currents at LC.** These are the tree engine's `u1-subtraction-color-flow-vector` auxiliaries (`models/compiler_color_flow.py:26`, `:413-425`) and the LC Fierz-singlet basis key with numeric weight 1/3 (`generation/dag_color.py:475-562`) [V].
5. **NLC.** Each defect lowers an upper bound on the reachable degree: closed quark loop −1, U(1) ≥ −1, open face ≥ −2, non-leading word ≥ −2. Prune when the bound is below P−2. Because NLC ≈ FC (§5), this saves little; it is a correctness guard, not an optimisation.

These rules reproduce the BDK left-mover structure for one quark line without any process-specific code. For q̄q + gluons, the LC loop contains at most one quark propagator, in the q̄|q gap (more only if singlets are attached there). Otherwise the loop is a pure gluon/ghost loop with the quark line inside a tree block [V, enumeration].

LC skeleton generation, with no diagrams [I; the enumeration in `lcs/lc_structure_counts.py` implements the same rules]:

```text
for each LC Born flow c (tree LC sectors; reflection-folded for traces):
    w = cyclic word of c with singlets placed in the colourless gaps of their lines
    for each composition of w into m >= 2 contiguous blocks
            (drop m = 2 with a single on-shell external leg as a block):
        currents  = tree-engine block currents (gluon if flavour-neutral,
                    quark/antiquark if net +-1, V if a lone singlet)
        species   = gluon/ghost on every gap, except quark on gaps that are
                    colourless in w (antiquark -> quark); fermion currents must pair
                    across exactly those gaps
        accept iff every vertex is a model vertex, no gluon current sits between
                    two quark propagators, and no closed quark loop is formed
        family    = canonical cyclic sequence of (block leg-set, propagator mass)
        emit skeleton (c, blocks, species) with seed N_c^P kappa A0(c,h)*
```

### 4.3 Counterterms and R₂ at LC [V table source, I classification]

A CT vertex is Σ_g N_c^g × (colour structure). At LC only the combinations that keep the Born flow at degree P survive, inserted into the colour-ordered Born tree of the same flow. So CT contributions are diagonal in orderings, exactly like the bare loop.

| loop_sm CT family (`loop-general-nc-r2-input-audit.md`) | Colour origin | LC | NLC adds |
|---|---|---|---|
| G_UVg (β₀) | 11 C_A | C_A part | — |
| G_UVq/c/b/t, heavy-quark decoupling logs | 4 T_F (per flavour) | — | all (P−1) |
| quark field and mass CTs (c/b/tWcft_UV, Mass_UV, UV_yuk) | C_F = N_c/2 − 1/(2N_c) | N_c/2 part | −1/(2N_c) part (P−2) |
| R2_3Gg, R2_GGg | explicit N_col | all | — |
| R2_GQQ, R2_QQq (C_F) | (N_col²−1)/(2N_col) | N_c/2 part | rest |
| R2_3Gq, R2_GGq (closed fundamental trace) | T_F | — | all (P−1) |
| four-gluon R₂ | C_C (single trace) × C_A | C_C × C_A part, colour-ordered | C_C × T_F N_f (P−1); C_B = δδ, degree 0 (P−2) |
| ggV, ggS, ggVV, ggSS (closed fundamental trace, two gluons) | δ^{ab}/T_F | — | all (P−1) |

How the table translates into engine work:

- The LC four-gluon R₂ vertex is a **colour-ordered single-trace** vertex. The tree engine's colour-flow lowering accepts singlet, identity, T and f only, not the d·d, δδ or d·f forms of the fixed-SU(3) rule (`understand-tree-engine.md` §0 (c)) [V via sibling]. The LC part needs only a cyclic δ-product, which is the native form of a colour-flow lowering. The fixed-SU(3) loop_sm rule must be replaced by the already integrated general-N_c HV asset (`MASTER_PLAN.md:4477-4483`) before any LC/NLC split [V].
- **Massless UV is proportional to the Born.** Insertions of renormalisable-vertex δZ's into a Born tree sum to (Σ factors) × Born per ordering. With massless quarks the LC UV term is analytic, and no CT trees are needed. Mass CTs (top), R₂ and external-field terms still need trees [I].
- The external-field term "δZ × order-masked Born interference" (assessment R2) uses the LC part of δZ and the diagonal LC Born interference [I].

### 4.4 DDM/Kleiss–Kuijf canonical blocks, verified at the colour level

A pure-gluon block B attached through an off-shell gluon x has colour-dressed current

J^x(B) = Σ_{π∈S_{|B|−1}} (F^{π(b₂)}⋯F^{π(b_k)})_{b₁x} J(b₁, π, x).

This is the DDM two-anchor decomposition, with the off-shell leg as one anchor. The DDM Jacobi/multi-peripheral argument uses only the f^{abc} structure of the vertices (arXiv:hep-ph/9910563 §2) [L]. BG currents obey rank-(p+1) Kleiss–Kuijf relations with the off-shell leg left out (Mafra–Schlotterer arXiv:1404.4986) [L\*]. The tree engine already certifies "connected f-tree colour" per model and coupling order before using DDM (`color/plan_build.py:139-170`, `color/fft_basis_strategy.py:93-105`) [V].

Colour-level check (`lcs/lc_kk_block_probe.py`, log `lc_kk_block_probe.log`) [V]: B = {1,2,3} in canonical order J(1,2,3), attached to a planar ring with legs 4, 5, 6. The canonical skeleton reaches degree P with **exactly 4 = 2^{|B|−1}** of the 120 Born traces, with coefficients:

- (123456): +1
- (124563): −1
- (134562): −1
- (145632): +1

All four keep B contiguous. The other 92 traces with non-zero overlap appear only at lower degree.

So the LC seed of a canonical skeleton is a signed sum of at most Π 2^{|B_i|−1} Born amplitudes. That costs nothing next to the dressing it saves (§6).

---

## 5. NLC: what it needs and what it costs

| Process class | P−1 terms | P−2 terms | New kinematic objects vs LC |
|---|---|---|---|
| All-gluon | All closed quark loops (massless N_f class, b, t), diagonal Tr(σ) overlaps; T_F N_f CT parts | The same gluon-loop primitives with off-diagonal and double-trace colour weights (COP/ring form, §2.1) | **Only quark loops.** No non-planar primitive exists for adjoint loops. |
| One quark line | (n_f/N_c) A^{L,[1/2]}; T_F CT parts | −A^R/N_c²; off-diagonal A^L contractions; A_{n;j}, j ≥ 3, via COP sums of mixed primitives; −C_F/(2N_c) CT parts | Quark loops, A^R, mixed primitives: ≈ the full FC set (BDK q̄qggg interference, §2.2) |
| Several quark lines | Quark loops; identical-flavour transposition overlaps | Uncrossed boxes, U(1) exchanges, A^R-type arcs | ≈ FC |

**Cost.** NLC ≈ FC in skeletons and reductions, for every required process. The only extra cost of FC over NLC is colour weights at degree ≤ P−3, which is negligible [V classes, I cost].

Where the odd powers come from: in the q̄q case, the subleading n_f terms "arise from the 1/N_c term in the SU(N_c) Fierz identity … 1/N_c factors come from explicitly projecting out the U(1) factor in U(N_c) = SU(N_c) × U(1), while structure constants accomplish this projection automatically" (DDM hep-ph/9910563 §3) [L]. In a colour-flow recursion every P−1 skeleton or CT term therefore comes from one of two sources, both structurally identifiable (§4.2, rules 3–5):

- a closed fermion loop, including the T_F N_f counterterm parts;
- a graded U(1)/Fierz-singlet current.

The remaining odd terms, overlaps between flows that differ by one quark transposition (identical flavours), live in the seeds, not in the skeleton set.

**Recommendation.** Implement NLC as FC skeletons with seeds truncated to degrees ≥ P−2. Build the truncation on exact Laurent overlaps, not on the tree NLC rule:

- the tree NLC factor is N_cⁿ − nN_c^{n−2} on the diagonal (`color/contraction_factors.py:100-108`), i.e. relative 1/N_c² only;
- that is a different definition from the loop's P, P−1, P−2 with odd powers [V].

**What the literature does to get accuracy cheaply.**

- **Colour-expansion sampling** (BlackHat W+3j, arXiv:0907.1984 §2.6 and §8).
  - LC virtual on 2×10⁶ points, subleading on 10⁵ points.
  - "Running times for leading- and subleading-color virtual contributions are also comparable."
  - Subleading-colour contributions are < 3 % for W+3/4 jets (arXiv:1111.4193 §4.3) and about 10 % of the virtual for pp→5j (NJet, arXiv:1312.7140 §2).
  - NJet: "separate leading and sub-leading contributions, which enables Monte-Carlo integrator to sample the dominant however simpler terms more often" (1312.7140 §2) [L].
- **LC-ratio** (BlackHat W+3j PRL, arXiv:0902.2760). Drop subleading terms "in the ratio of the finite virtual terms to the tree-level cross section", then multiply by the full-colour tree. Keep the exact-colour poles. This is "a much better estimate than a strict leading-color approximation" (0907.1984 §8) [L].

  In pyAmpliCol it would be a separate, explicitly named observable: it needs only the LC loop plus FC Born and colour-correlated Borns. It is not MASTER_PLAN LC.
- **De-symmetrised colour sums** (NJet, arXiv:1209.0100 §2.2, eq. `dsymXS`).
  - For gg→n g the orderings drop from (n−1)!/2 to (n−1); 7g FC falls from 171 s to 3.07 s (§4.3, Table `tab:timedsymgluons`). The result is valid only after Bose-symmetric phase-space integration.
  - At LC the reduction is larger, because only the orbits of the Born orderings under final-state permutations remain [I, counted]:

    | Process | LC orderings | De-symmetrised orbits |
    |---|---|---|
    | gg→gggg | 60 | 3 |
    | gg→ggggg | 360 | 3 |
    | uu~→ggggg | 120 | 1 |
    | uu~→e⁻e⁺ggg | 6 | 1 |

  - The per-point value changes, so this can only be an opt-in MC mode.

---

## 6. Counts and expected LC/FC cost ratios

### 6.1 Method [V]

`lcs/lc_structure_counts.py` enumerates every cyclic word of the external legs (rotation removed) and every composition into m ≥ 2 blocks:

- tadpoles are excluded, as are m = 2 compositions where one block is a single on-shell leg (scaleless or LSZ);
- every block current must be valid: net flavour 0 gives a gluon (or V if colourless); net ±1 of one flavour gives a quark;
- every fermion routing is formed from non-overlapping arcs between adjacent fermion-current blocks;
- for blocks with no fermion current, the loop is a gluon/ghost loop or a closed quark loop in each mass class: massless {u,d,s,c}, b and t, following the loop_sm default restriction (MB = 4.7, MT = 173; earlier attempt's check).

Skeletons are deduplicated under rotation and reflection. Families are the canonical cyclic sequence of block sets with propagator masses, so gluon and massless-quark propagators share families (Recola's `hm` convention). LC uses the face rule of §4.1. FC is every valid skeleton; for one quark line these are Kälin's minimal bases.

Cost proxies:

- **Forward open-loop prefix states**, cut before the block containing leg 0. A depth-j state carries C(j+4,4) monomials of rank ≤ j. Both reading orientations are counted, so absolute values are about 2× a canonical-orientation engine; ratios are unaffected.
- **"KK"** restricts pure-gluon blocks to canonical internal order (§4.4).
- **Cut visits** = Σ_families [C(m,4)+C(m,3)+C(m,2)].

Runtime 39 s, 0.55 GB. Output: `lcs/lc_structure_counts.json`.

Cross-checks against independent counts:

- 51 and 113 compositions per ordering for n = 6 and 7.
- 550 and 4,707 massless families, which equal Σ_m S(n,m)(m−1)!/2 with the m = 2 correction (sibling §5).
- 3,060 and 40,680 gluon-loop skeletons.
- 785 and 8,011 KK skeletons (earlier attempt).
- 720 FC mixed primitives for uu~→ggggg, as in NJet and Kälin.

### 6.2 Results

| Quantity | gg→gggg | gg→ggggg | uu~→ggggg | uu~→e⁻e⁺ggg | uu~→e⁻e⁺dd~g | gg→tt~gg |
|---|---|---|---|---|---|---|
| LC orderings (up to reflection) | 60 = 5!/2 | 360 = 6!/2 | 120 = 5! | 6 = 3! | 4 | 24 = 4! |
| LC planar families per ordering | 51 | 113 | 113 | 51 | 36 | 51 |
| LC skeletons | 3,060 | 40,680 | 13,560 | 306 | 144 | 1,224 |
| LC skeletons, KK blocks | 785 | 8,011 | 7,348 | 242 | 144 | 784 |
| LC families (by number of propagators 2…n) | 550 (25/90/195/180/60) | 4,707 (56/301/1050/1680/1260/360) | 2,712 (56/301/765/930/540/120) | 164 | 114 | 375 |
| FC skeletons | 12,240 | 162,720 | 101,520 | 4,440 | 2,652 | 7,656 |
| FC families | 1,650 | 14,121 | 5,809 | 513 | 254 | 1,164 |
| LC/FC dressing proxy (plain; KK) | 0.25; 0.25 | 0.25; 0.25 | 0.15; 0.13 | 0.08; 0.07 | 0.08; 0.08 | 0.17; 0.16 |
| LC/FC NinjaRS families; cut visits | 0.33; 0.33 | 0.33; 0.33 | 0.47; 0.42 | 0.32; 0.20 | 0.45; 0.32 | 0.32; 0.27 |
| KK gain on LC dressing proxy | 2.6× | 3.2× | 1.65× | 1.2× | 1.0× | 1.4× |

Notes:

- **Pure gluons.** LC = FC minus all quark-loop classes, so the ratio is exactly 1/(1 + number of quark mass classes) = 0.25 here. It is 0.33 if b is massless (`restrict_no_b_mass`). Closed fermion loops have no quartic vertices and ghosts are cheap, so the real ratio is probably 0.25–0.35 [I].
- **uu~→ggggg.**
  - 13,560 LC skeletons out of 101,520 FC skeletons. The FC set is 71,280 mixed or gluon-loop skeletons (the LC ones are a subset) plus 30,240 closed-quark-loop skeletons over three mass classes.
  - The 120 LC heptagon parents each have exactly one quark propagator, in the q̄|q gap.
  - LC removes 4,707 − 2,712 = 1,995 massless families (those where q̄ and q sit in non-adjacent blocks) and all 2 × 551 massive closed-loop families [V].
- **e⁺e⁻ (V) processes.** V must sit in a colourless gap (§3.2). The four LC words of uu~→e⁻e⁺dd~g are the two crossed flows (gluon on the u or d string) × V on the u or d line [V].
- **gg→tt~gg.** The LC gap propagator is a massive top, so LC families mix massless and top propagators. FC adds three closed-quark-loop mass classes and all mixed top-line primitives.

### 6.3 Expected overall LC/FC per point [I]

Assume dressing dominates (the toy model of `understand-algorithms.md` §8.1) and NinjaRS plus masters take 10–20 % in v2 (they were 0.4–2 % of the inherited runtime, assessment §1). Weighting with the table gives:

| Process | Expected LC/FC per point |
|---|---|
| gg→gggg, gg→ggggg | ≈ 0.26–0.35 |
| uu~→ggggg | ≈ 0.18–0.22 |
| gg→tt~gg | ≈ 0.19–0.22 |
| uu~→e⁻e⁺ggg | ≈ 0.10–0.12 |
| uu~→e⁻e⁺dd~g | ≈ 0.11–0.13 |

KK canonical blocks give a further 1.2–3.2× in both LC and FC. These ratios are consistent with BlackHat's measured ≈ 1/20 for W+3 jets (§0.1), given its literature LC definition (§1).

---

## 7. Reusing the tree engine

### 7.1 What exists [V]

| Concern | Where | What it does |
|---|---|---|
| LC sectors (flows) | `color/plan_build.py:17-129` (`build_color_plan`); `:226-290` (traces); `:255` (LC reflection folding); `:96` (block permutations only for NLC/full) | Enumerates LC words: traces, or open-line strings with adjoint allocations. Takes a `max_sectors` cap argument (`:21`, `:110-112`). |
| Sector semantics | `color/plan_types.py:69-240` (`LCColorSector`, `admissible_traversal_words` `:116`, `canonical_closure_traversal_word` `:144`) | Sector colour words, block traversals, canonical closure sink. |
| Recursion colour engine | `generation/dag_color.py:68` (`ColorEngine`); `:130-157` (`_shared_single_trace`, NLC/full gluons); `:158-169` (`_shared_lc_orderings`); `:186-243` (LC segment table); `:1041-1076` (segment admissibility) | Structural LC pruning of tree currents by contiguous word segments. |
| Fierz singlet at LC | `generation/dag_color.py:475-562` (weight `1.0 / 3.0` at `:539`, `:562`) | Numeric N_c = 3 Fierz-singlet piece in LC sectors (historical tree convention). |
| U(1) subtraction | `models/compiler_color_flow.py:26`, `:413-425` (`coefficient /= fundamental_dimension`) | Numeric 1/N_c auxiliary currents. |
| Contraction | `color/contraction.py:48-49` (LC → no plan; "historical Rusticol diagonal reduction path", docstring `:39-45`); `color/contraction_factors.py:86-125` (LC N_cⁿ diagonal at `:98-99`; NLC `:100-108`); `:242-260` (one open line); `:621` (N_c power terms) | Tree LC is diagonal; NLC/full use sparse exact matrices. |
| DDM Gram | `color/adjoint_kernel.py:1-9`, `:36-62`, `:64`, `:136`; `MAX_ADJOINT_GLUONS = 12` at `:20` | U(N_c) cycle-counting kernels with integer NLC truncation. |
| FFT basis choice | `color/fft_basis_strategy.py:38-116` | DDM S_{n−2} only with a model f-tree certificate; fundamental chains otherwise. |
| Runtime codec | `generation/recurrence_color.py:27-28`, `:447-451` | Encodes only NLC and full contraction plans; LC never needs one. |
| LC topology replay | `color/plan_replay.py:143` (safe groups), `:367-435` (`build_lc_topology_replay_plan`), `:437` (proof), `:671` (fermion sign) | One representative sector evaluated, the others replayed with permuted momenta, proof-gated by model-contract label equivariance. |
| Exact overlaps | `color/connections.py:492` (`_literal_tensor_overlap_nc_terms`), `:517` | Exact Laurent-in-N_c overlaps of trace/open-string tensors with contracted labels, through the repo's trace/Fierz reducer. |
| Inherited loop LC/NLC | `loops/color_projection.py:26-132`; `loops/physical_color.py:257-416` | Correct structural projection after computing everything; no pruning (measured LC ≈ FC). |

### 7.2 How the loop recursion should use it [I]

1. **Orderings.** Call the tree LC plan (`build_color_plan(..., color_accuracy="lc", fold_trace_reflections=True)`) and close each sector word cyclically. Place singlets in colourless gaps. LC loop orderings are in one-to-one correspondence with LC Born flows, up to reflection and singlet placement (§3.2). Never pass `max_sectors` (user rule: no caps).
   - The open-line plan enumerates every antifundamental pairing (`color/plan_build.py:75-116`). Pairings without a leading U(N_c) Born amplitude produce no LC loop ordering. An example is the uncrossed pairing of distinct flavours joined only by QCD. Detect such pairings structurally (no leading-flow closure in the Born DAG) and skip them [I].
2. **Open-loop colour state.** Reuse the `CurrentIndex` and `ColorEngine` machinery with two extra cut labels (`understand-tree-engine.md` recommends this extension). Extend the LC segment table with cut-label-terminated segments (§4.2). The current key additionally needs the denominator history (sibling §3.2).
3. **Blocks.** Blocks are the tree engine's LC currents from the same DAG, so the Born and the loop attachments share one table (assessment R5). Most LC blocks are contiguous segments of sector words and already exist. Blocks that wrap around a sector's sink label must be added on demand, with the same rules.
4. **Seeds.** Take the per-(sector, helicity) Born amplitudes from the runtime arenas (`understand-tree-engine.md` §0 (a)). Multiply by N_c^P κ. With KK blocks, form signed sums over at most 2^{|B|−1} sectors per block.
5. **FC/NLC Gram.** Loop skeleton colours are DDM rings or flow products. Compute their overlaps with the Born basis once, with `color/connections.py:492` (exact, ≈1.5 ms per 7-gluon ring [V]) or the cycle-counting kernels. The Born side can use the existing S_{n−2} FFT planner where `ddm_proven`. MASTER_PLAN §2.16's "Do not inherit tree DDM … certificates for loop amplitudes" (`MASTER_PLAN.md:609`) concerns loop FFT orbits. It does not forbid using the model's tree f-colour certificate for tree blocks (§4.4).
6. **Replay (optional, generation-side).**
   - All LC orderings with the same species pattern are relabellings of one program: one pattern for n gluons, one for q̄q + gluons. `plan_replay.py`'s proof gate already certifies this for trees.
   - Compiling one ordering and replaying it loses only 4.6–6.3 % of open-loop state sharing at 5–7 gluons (`lc/lc_replay.py`, re-run log `lcs/rerun_lc_replay.log`) [V].
   - Replay must write into global family accumulators after an affine loop-momentum shift (MASTER_PLAN §2.13), so cross-ordering family merging is kept.
   - Replay is in tension with KK canonical blocks, because canonical order is label-dependent. Choose KK for per-point speed, and replay only if generation time or artifact size becomes the bottleneck.

### 7.3 Pitfalls to remove when reusing it [V/I]

- **Fierz singlet and U(1).** The tree LC Fierz singlet (`dag_color.py:539`, `:562`) and the U(1) auxiliaries (`compiler_color_flow.py:419`) carry numeric 1/3. Under formal N_c they are degree −1 pieces, so the strict loop LC seed must exclude them or carry them as graded terms (the tree-engine report already flags this).
- **Two NLC definitions.** Tree NLC (`contraction_factors.py:100-108`) and loop NLC (`color_projection.py:34-50`) are different definitions. Reuse exact Laurent coefficients, not the tree truncation.
- **Caps.** `MAX_ADJOINT_GLUONS = 12` (`adjoint_kernel.py:20`) and `max_sectors` (`plan_build.py:21`) are arbitrary caps and must not reach the loop engine.
- **R₂ source.** The fixed-SU(3) four-gluon R₂ cannot be split into LC/NLC. Use the general-N_c asset.

### 7.4 What the production one-loop generators do [L/V]

- **OpenLoops 2** includes exact N_c dependence by default. "Dedicated process libraries with large-N_c expansions can be generated by the authors upon request", selected with `leading_colour = 1` at process registration (arXiv:1907.13071 §3.2) [L]. LC is therefore a *library-generation* restriction there too, which is the design recommended here.
- **Recola** works only with colour flows. Its currents carry "colour structures" (products of Kronecker deltas), and its `colour_optimization` levels are exact full-colour optimisations:
  - level 1 computes branches that differ only by a colour factor once ("mother/daughter" branches);
  - level 2 skips external U(1)-gluon flows and rebuilds dependent structures with the SU(N) projector.

  Sources: arXiv:1211.6316 §2.4; `recola2-2.3.0/src/core/skeleton_rcl.f90:815-819`, `currents_rcl.f90:1827-1880, 2365-2420`; `pyrecola.c:234-240` [V]. Recola has no LC mode.
- **NJet/BlackHat** compute primitives and assemble colour afterwards. They use LC and subleading colour as separately sampled MC contributions (§5).

---

## 8. Composition with the "family-tree" reduction idea

**Structure [V].**

- All LC families of one ordering are pinches of that ordering's n-gon: the 2ⁿ−2n−1 compositions of the cyclic word.
- A family appears in every ordering whose word contains its blocks contiguously and in its cyclic block order. At 7 gluons, 40,680 (ordering, family) pairs map onto 4,707 families, i.e. 8.6 orderings per family on average.
- At LC every skeleton of ordering σ carries the same seed w_σ(h). So the Born-weighted "parent integrand" of σ is w_σ(h) times the integrand of the LC primitive A_{n;1}(σ): the object unitarity codes reduce with one parent diagram (NJet arXiv:1209.0100 §2.1; EKMZ arXiv:1105.4319 §7) [L].

**Two ways to reduce** (7 gluons, LC, massless) [V counts]:

| Schedule | NinjaRS calls | Cut visits | Member expansions (callback work) | Cross-ordering merge | Precision |
|---|---|---|---|---|---|
| Family-major, Born-weighted before reduction (MASTER_PLAN §2.14) | 4,707 | 150,570 | 150,570 | yes, 8.6 orderings per family | separate reductions |
| Parent-major per ordering, cut-restricted lift (sibling §4.2) | 360 | 32,760 | 360 × 1,505 = **541,800** | no | ≈1.5 digits worse at n = 7 (sibling probe: 1–3×10⁻⁴ vs 4.5×10⁻⁶) |

Parent-major saves only NinjaRS's per-cut overhead, which is 4.6× fewer visits. With realistic callbacks it needs **3.6× more** expansion work, because it cannot merge a family's numerators across orderings before reducing. A global tensor lift is excluded by the rank bound (rank ≤ n+2; sibling §4.1) and by MASTER_PLAN §2.13 (`MASTER_PLAN.md:531`).

**Recommended use of the family tree:**

- **Generation.** Enumerate families and skeletons combinatorially per ordering (§6), with no diagrams; FeynKit stays the small-process oracle (`MASTER_PLAN.md:1496-1502`).
- **Reduction.** Family-major, Born-weighted and helicity-summed, with a per-point master cache. At 7 legs there are 1,407 distinct masters for 150,570 requests (sibling §4.4). At LC every master is a scalar integral over consecutive-momentum invariants of some ordering, so the cache key is purely combinatorial [I].
- **Verification.** An evaluation-first LC oracle reduces one ordering's parent through a NinjaRS callback (sibling §3.3). At LC its input is exactly the LC primitive, which unitarity codes and the analytic pole formula (§3.2) can cross-check.
- **KK interplay.** With canonical blocks, each canonical skeleton still belongs to exactly one family (its ring of block sets). The family count is unchanged (4,707). Each family now receives Π (|B_i|−1)! canonical numerators instead of Σ over orderings of Π |B_i|! [I].

---

## 9. Recommendations

1. **Make LC a generation-time restriction, not a projection.** Generate only face-rule skeletons from LC sector words. Seed them diagonally. Prune quark loops, A^R, mixed and U(1) classes before dressing. This turns the measured LC/FC ≈ 1 into the ratios of §6.
2. **Build one skeleton machinery for all colour modes.** LC, NLC and FC differ only in which skeleton classes are generated (§4.1) and in their seeds (diagonal, truncated, or full Gram).
3. **Adopt KK canonical pure-gluon blocks once the plain path is validated.** They give 2.6–3.2× on gluon-rich dressing in every colour mode. Validate them by exact equality against plain ordered blocks on gg→ggg and gg→gggg, and by the 2^{|B|−1}-term seed structure (§4.4).
4. **Keep family-major Born-weighted reduction and add the master cache.** Do not reduce per ordering parent.
5. **CTs at LC.**
   - Use the LC parts per the §4.3 table.
   - Use the general-N_c four-gluon R₂ asset.
   - Grade the Fierz/U(1) pieces.
   - Use an analytic Born-proportional UV term where no masses enter.
6. **Validation gates.**
   - LC (new) equals the inherited engine's LC projection on DY, uudd, qq→gg, gg→gg, uu~→ggg and uu~→e⁻e⁺gg, in all carriers.
   - LC poles equal the adjacent-dipole formula per ordering.
   - NLC and FC (new) equal the inherited projections on the same controls.
   - Pruning counts as in §6.
7. **Offer explicit opt-in MC modes** (not defaults): de-symmetrised LC, LC-ratio, and separate LC and FC−LC outputs for colour-expansion sampling.
8. **Expected gain** [I]. LC per point at about 0.25–0.35 of FC for the gluon processes and 0.1–0.2 for quark processes, times a further ≈ 1.2–3.2 from KK blocks. NLC costs ≈ FC; its value is accuracy, not speed.

### What to verify next

- **Real dressing costs per skeleton class.** Measure gluon ring vs fermion arc vs closed fermion loop on a v2 prototype, to replace the equal-weight proxy of §6.
- **The off-shell DDM decomposition of tree blocks** on actual tree-engine currents (numerical KK check per block), not only at the colour level.
- **The strict-LC Born seeds** from the runtime arenas, with the Fierz-singlet grading, on uu~→dd~ss~-type processes where U(1) pieces matter.
- **Mixed QCD/EW sectors.** Structural P per Born×loop pairing for gg→tt~e⁻e⁺g (QCD and EW), including EW loops whose LC set is "flow-preserving" rather than "face-closed".

---

## Appendix A. Reproduction

```sh
source /shared/external/pyLoopAmpliCol/workspace/env.sh
cd /shared/external/pyLoopAmpliCol/workspace/tmp/engine-v2/lcs
/shared/external/pyLoopAmpliCol/workspace/venv/bin/python -I lc_structure_counts.py   # 39 s; writes lc_structure_counts.json
PYTHONPATH=/shared/external/pyLoopAmpliCol/pyamplicol/src \
  /shared/external/pyLoopAmpliCol/workspace/venv/bin/python lc_kk_block_probe.py      # exact colour-level KK check
(cd ../lc && PYTHONPATH=/shared/external/pyLoopAmpliCol/pyamplicol/src \
  /shared/external/pyLoopAmpliCol/workspace/venv/bin/python lc_colour_probe.py out.json)   # exact degree classes
(cd ../lc && /shared/external/pyLoopAmpliCol/workspace/venv/bin/python lc_replay.py)    # replay overhead
```

Logs: `lcs/lc_structure_counts.log`, `lcs/lc_kk_block_probe.log`, `lcs/rerun_lc_colour_probe.log`, `lcs/rerun_lc_colour_probe2.log`, `lcs/rerun_lc_replay.log`.

## Appendix B. Sources read in this session

| Source | Sections |
|---|---|
| hep-ph/9403226 (Bern, Dixon, Dunbar, Kosower) | §2 colour decomposition; §7 COP formula |
| hep-ph/9409393 (Bern, Dixon, Kosower, q̄qggg) | §2 colour decomposition and interference formula; §3 primitives, L/R, reflection, vanishing fermion loops; §4 A_{n;1} |
| hep-ph/9708239 (BDK, e⁺e⁻→4 partons) | §2.2 colour decomposition with the vector boson; §2.3 primitives |
| hep-ph/9910563 (Del Duca, Dixon, Maltoni) | §2 LCA statement and Jacobi argument; §3 one-loop ring decompositions (gluons, q̄q) |
| hep-ph/0209271 (Maltoni, Paul, Stelzer, Willenbrock) | §3 leading term and even powers; U(1) gluons |
| 1105.4319 (Ellis, Kunszt, Melnikov, Zanderighi) | §6.1–6.2 colour (DDM one loop, n_f part); §6.4.2 primitive poles; §7 parents |
| 1111.4193 (Ita, Ozeren) | Abstract; §2.3–2.4 partial and primitive amplitudes, routing; §4.3 W+4 jets LC vs FC |
| 1310.0413 (Reuschle, Weinzierl) | §2.2 planar words on the annulus; §3 partial and primitive amplitudes; §4 U(1) operations (headings) |
| 1712.03539 (Kälin) | §3 minimal bases; §4.1 loop colour = closed tree colour |
| 1209.0100 (NJet) | §2.1 parents; §2.2 RREF colour, de-symmetrisation; §4.3 timings |
| 1312.7140 (NJet proceedings) | §2 LC/SLC separation for 5 jets |
| 0902.2760, 0907.1984 (BlackHat W+3j) | LC definition (0907.1984 §2.2); timings and colour-expansion sampling (§2.3, §2.6); subleading-colour terms (§8) |
| 1210.6684 (Ita et al., proceedings) | LC sampling motivation |
| 1112.3940 (BlackHat 4 jets) | Setup (full colour used) |
| 0901.4101 (Ellis, Melnikov, Zanderighi W+3j) | §4 LC p = 2 term, interference formula |
| 0911.1962 (Giele, Kunszt, Winter) | Motivation; one-loop flow decomposition with subleading Δ products |
| 0903.4665 (HELAC-1Loop) | §2 one-particle cutting, planar and non-planar colour connections, R₂ and CT as trees |
| 1907.13071 (OpenLoops 2) | §3.2 "Number of colours" (leading_colour=1 libraries); §4.5 LC colour-flow identities (eqs. 4.34–4.37) |
| 1211.6316 (Recola) and recola2-2.3.0 source | §2.4 structure-dressed colour; `colour_optimization` levels (`skeleton_rcl.f90:815-819`, `currents_rcl.f90:1827-1880`, `pyrecola.c:234-240`): exact full-colour optimisations, no LC mode |
| 1205.2096, 1010.4187, 1111.1733 (Becker, Reuschle, Weinzierl et al.) | One-loop colour-ordered recurrence after cutting the loop; LC e⁺e⁻→7 jets, n⁴ CPU scaling |
| hep-ph/0607057 (Duhr, Höche, Maltoni) | Adjoint (DDM) and colour-flow decompositions of currents; KK |
| 1404.4986 (Mafra, Schlotterer) | KK relations of BG currents [L\*, search summary only] |

Also used: `MASTER_PLAN.md` §§1.2, 2.11–2.17 and the formal-colour records (lines cited); the assessment §§1–2 and R1–R5; the journal colour/FFT reading; `workspace/reports/loop-general-nc-r2-input-audit.md`; and the sibling engine-v2 reports.
