# Engine v2: Recola, NinjaRS and the recursion–reduction interface

Date: 2026-10-20. Research report; the `pyamplicol` repository (branch `loop`, 4faafcad) was not modified.

**Question.** How can pyAmpliCol get the most out of recursion relations when NinjaRS needs, per reduction call, a numerator polynomial in q (and μ²) over one fixed denominator set?

**Evidence labels**

| Label | Meaning |
|---|---|
| [V] | Verified here: source or document read at the cited `file:line`. |
| [L] | Literature read here (PDF text extraction or arXiv LaTeX source; section or file:line given). |
| [M] | Measured here by an experiment in `workspace/tmp/engine-v2/rn2/` (or a retained log from the interrupted previous attempt of this task, said so). |
| [I] | Inference, model estimate or design judgement. |

**Paths**

- `recola2/` = `pyamplicol/FOR_REFERENCE_ONLY/recola2-2.3.0/src/`
- `ninjars/` = `workspace/cache/cargo/git/checkouts/ninjars-3842d69907c1a3a3/0d0d7e7/` (the revision pyAmpliCol pins; it is newer than `FOR_REFERENCE_ONLY/ninjars`, e4c16a7).
- Scratch: `workspace/tmp/engine-v2/rn2/` (this report), `workspace/tmp/engine-v2/recola-ninjars/` (Recola tensor-integral dumps and timing harness from the interrupted earlier attempt of this task, re-checked here).

**Companion reports** in this directory, which this one does not repeat: `understand-algorithms.md` (OpenLoops/Recola/MadLoop survey, Born-seeded schedule, toy cost model), `understand-recursion-literature.md` (van Hameren, HELAC, NJet, first lifting probe), `understand-tree-engine.md`, `understand-contracts.md`. Where this report corrects one of them, it says so (§0.7).

---

## 0. Answers first

### 0.1 The structural answer

Recursion shares work in four places. Only one of them is blocked by NinjaRS, and that one can be recovered at reduction time.

| Level | What is summed | Mechanism | Blocked by the fixed-denominator contract? |
|---|---|---|---|
| L1 tree blocks | All trees on a leg subset | pyAmpliCol's Berends–Giele currents, shared by Born, loop attachments and CT trees | No |
| L2 loop, same family | All diagrams with the same denominator history | Recola's `(ho, hm)` current key (forward), or OpenLoops-2 merging on (family, undressed suffix) after Born seeding and on-the-fly helicity sums (backward) | No. This is exactly the per-family numerator NinjaRS wants. |
| L3 loop, across pinch-related families | Diagrams of different families that share a parent n-gon | Impossible as one polynomial: different denominators. **Possible on every cut**, because there D_j − D_i is affine in q and μ²-free | Yes for the recursion; **no for the reducer**: NinjaRS samples the numerator only on cuts and tells every callback which cut it is on (§4.2) |
| L4 masters | Scalar integrals shared by many families | Per-event master cache (Collier's cache in Recola, the AvHOneLoop cache in original Ninja) | No. Missing in NinjaRS and in pyAmpliCol's adapter today |

The literature has two camps. Coefficient-first codes (Recola, OpenLoops, MadLoop, van Hameren) stop at L2. Evaluation-first codes (NJet, EGKM) get L3 inside the recursion, but pay with one recursion per cut. The design below gets L1, L2 and L4 in the recursion and the cache, and recovers L3 in the reducer by a **cut-restricted lift of pinched families into parent groups** (§4.2). That lift preserves the rank class and needs no NinjaRS API change; measured on synthetic numerators it halves the NinjaRS stage at 7 propagators [M].

### 0.2 Recola in brief [V]

- **Recursion.** Recola builds the loop as a tree with two extra legs, once per cut particle.
- **Current identity.** A loop current is interned only if particle, leg mask, **propagator history `ho`**, **mass history `hm`**, counterterm flag, coupling orders, U(1) flag and colour structure all match (`recola2/core/skeleton_rcl.f90:876-903`).
- **Tensor integrals.** A tensor integral (TI) is the `(ho, hm)` of the closing current (`recola2/core/currents_rcl.f90:3391-3416`). Its rank is the maximum over contributions. Different cut particles with equal masses share it.
- **Pinched topologies** stay separate TIs. Their sharing happens inside Collier, whose global cache identifies identical sub-integrals across all TIs of an event by call position [L 1604.06792 §4 "Structure of the library" and §5.5 "Using the cache system"].
- **Amplitude level.** Recola contracts every closing branch with its TI separately for each helicity configuration, cut polarisation and colour structure (`recola2/core/amplitude_rcl.f90:1470-1496`). This is possible because tensor integrals do not depend on the numerator.
- **Where the time goes** (measured on this host). Collier TIs take 2.4 % (gg→ggg, 411 ms/pt) to 29 % (uu~→e⁻e⁺dd~g, 41 ms/pt) of Recola's time. The tensor coefficients take the rest [M].

### 0.3 NinjaRS in brief [V]

- **One call.** A call reduces one integrand N/Π_{i∈P} D_i completely. It processes:
  - all C(n,4) box cuts;
  - C(n,3) triangles if rank ≥ n−3;
  - C(n,2) bubbles if rank ≥ n−2;
  - n tadpoles if rank ≥ n−1.

  Scaleless cuts are skipped, pentagons are never computed, and bubble/tadpole subtractions are coefficient-level corrections from the same call's triangles and bubbles (`ninjars/crates/ninjars/src/fast/engine.rs:228-333, 673-684, 1256-1257`).
- **Callbacks.** Every numerator callback receives the cut:
  - fast engine: `cut`, the sorted partition (`fast/numerator.rs:61-78`);
  - generic engine: `partition` in every request (`numerator.rs:4-9, 58-63`).

  Original Ninja states that an expansion "only needs to be valid for the cut specified" [L 1403.1229 §3.1].
- **Rank** ≤ n+2. Rank n+2 is 2.1–2.9× its n+1 twin and has extra conditioning issues. Over-declaring the rank costs up to 981× accuracy (`ninjars/docs/rank-n2.md:345-353, 825-829`).
- **Masters** are batched per call and type, with **no cache** (`fast/engine.rs:1746-1838`, `docs/benchmarks.md:65`). pyAmpliCol's wrapper only times them (`pyamplicol/rust/loop-reduction/src/lib.rs:241-262`).
- **Cost.** With masters cached, about 75 % of NinjaRS time is the numerator expansions (Horner sweeps of the dense tensor at every cut) [M].

### 0.4 Verdict on the structural options

Numbers are for gg→ggggg in full colour, per point, binary64 NinjaRS fast engine, one core [I: calibrated model, §6; ratios also measured, §4.2].

| Option | Exact? | Rank | NinjaRS reductions per event | NinjaRS stage | Verdict |
|---|---|---|---|---|---|
| P0: one reduction per family | yes | true | 9,414 | 1.0–2.3 s without master cache; 0.79–0.82 s with it | Baseline. **Add the master cache.** |
| (a) global D-dim lift into hosts | yes | +2 per lifted line, μ² terms | 2,225 (≤ n+1) / 1,072 (≤ n+2) | 0.80 s / 2.7 s | **Reject**: never cheaper, accuracy loss (n+2 conditioning; earlier probe errors 3e-4 / 8e-3 at n = 6/7). |
| (b1) cut-restricted lift into parent groups (no API change) | yes | class preserved (≤ n) | 720 groups | 0.30–0.40 s | **Adopt as opt-in stage 2.** Measured 1.5× (n=6) to 2.0× (n=7) faster than separate reductions, with errors at rotation-noise level. |
| (b2) parent groups with per-member division (needs NinjaRS API) | yes | true per member | 720 groups | about 0.7 s | Small gain (shares only geometry/subtraction/masters). Keep only as a precision fallback. |
| (c) on-the-fly reduction (OFR) | yes for N ≥ 4; integral-level only at N = 3 | ≤ 2 | 9,414 (closure; +178 extra families for uu~→e⁻e⁺dd~g) | about 0.01 s, but the work moves into the dressing | **Reject for production.** OpenLoops' own benchmark finds a-posteriori reduction faster; it brings Gram-type instabilities and private reduction maths. |
| (d) no Born weighting (amplitude level) | yes | true | 8.7×10⁸ | — | Born weighting before reduction is mandatory with integrand reduction. |
| (e1) per-event master cache | yes | — | unchanged | removes 0.23–1.5 s | **Adopt in stage 1.** 109× redundancy (308,448 requests, 2,822 distinct). |
| (e2) families (or groups) as lanes | yes | — | **6** physical calls (1 with groups) | small | **Adopt.** No API change; needs per-lane rank when ranks are mixed. |

### 0.5 Counts headline

| Process | Families FC (massless + massive) | Families LC | Maximal parents FC / LC | Reductions per event, P0 → (b1), FC | Same, LC | Physical calls, P0 → groups |
|---|---|---|---|---|---|---|
| gg→gggg | 1,100 (550 + 550) | 550 | 120 / 60 | 1,100 → 120 | 550 → 60 | 5 → 1 |
| gg→ggggg | 9,414 (4,707 + 4,707) | 4,707 | 720 / 360 | 9,414 → 720 | 4,707 → 360 | 6 → 1 |
| uu~→ggggg | 5,258 (4,707 + 551) | 2,712 | 420 / 120 | 5,258 → 420 (loose) / 971 (strict) | 2,712 → 120 / 671 | 11 → 3–6 |
| uu~→e⁻e⁺dd~g | 244 (231 + 13) | 115 | 39 / 4 | 244 → 39 / 66 | 115 → 4 / 83 | 10 → 5–6 |

The family enumerator is validated against Recola's tensor-integral tables (exact set equality on 5 processes, §5.1). uu~→e⁻e⁺dd~g families come directly from Recola. "Strict" and "loose" are the group-assignment policies of §4.2.4.

### 0.6 Recommendation

1. **Family identity** = Recola's `(ho, hm)`: a canonical cyclic block sequence of external legs plus the propagator mass pattern, up to rotation and reflection. Generate it from leg subsets and tree-current reachability, without diagrams. Use FeynKit `IntegralFamily` only as the small-process oracle.
2. **Numerator recursion** as in `understand-algorithms.md`:
   - Born-seeded, OpenLoops-2 schedule, over Berends–Giele blocks;
   - Recola keys;
   - output: one Born-weighted, helicity-summed tensor per family and channel, at the **true structural rank**.
3. **Stage 1 reduction:**
   - one NinjaRS reduction per family;
   - families with equal (n, rank) batched as lanes of one `FastAmplitude`;
   - a per-event master cache keyed by a generation-time master id, with all S-matrices filled from one canonical invariant table.
4. **Stage 2 reduction (opt-in, after physical validation):**
   - families grouped under maximal parents (LC: the planar n-gon of an ordering), with a balanced, rank-aware assignment;
   - one call per group, whose callback returns the per-reference lifted tensor T_{i(cut)} (§4.2);
   - per-lane fallback to stage 1 when NinjaRS flags instability.
5. **Do not:** globally lift, run OFR, or reduce at amplitude level.
6. **Ask the NinjaRS owner**, in priority order (§7.4):
   1. a caching master provider for both engines;
   2. a layout-aware `mul_affine` / lifted-tensor builder and a cut-reference numerator adapter;
   3. per-lane rank (already planned) and an explicit test that lanes may carry unrelated families;
   4. a multi-numerator (transposed) reduction for several channels on one geometry;
   5. optionally, a per-member-division group API as the exact fallback.

### 0.7 Corrections to earlier reports

- `understand-recursion-literature.md` §0.1(c) and §4.3–4.5 conclude that the cut-restricted lift ("variant A") saves no numerator work and that building the per-reference tensors costs more than it saves. That probe lifted all 127 subsets of a heptagon, including tadpoles and single-leg bubbles, with full-rank random numerators, and did not time the build.
  - With realistic groups (9–14 members per parent under a balanced assignment, physical sub-families only), the measured build is 28k–98k complex MACs per lane, not 3×10⁶.
  - The callback (expansion) work collapses to that of the parent alone.
  - Net: 1.5–2.0× faster NinjaRS stage, at rotation-noise accuracy [M, §4.2.3].

  The P1 policy proposed there is therefore worth more than "≈2× of the non-callback part". It is the main reduction-side optimisation.
- The same report's "Collier's cache shares pinched sub-integrals" was [L*]. It is now verified [L 1604.06792 §4: "a single user call of an N-point integral … leads to recursive internal calls of lower N′-point integrals … different user calls … involve identical tensor integrals … pairs (N_ext, id) with identical arguments are pointed to the same address in the cache"].

---

## 1. The constraint and its one exact loophole

**NinjaRS contract** [V].
- Per call: N(q, μ²)/Π_{i=0}^{n−1} D_i with D_i = (q+V_i)² − m_i² − μ², rank ≤ n+2, μ² of weight 2.
- Numerators come as dense tensors (`TensorNumerator`, C(R+4,4) coefficients in MadLoop/Ninja layout, `fast/tensor_layout.rs:1-12`) or μ²-blocked tensors (`MuTensorNumerator`).
- Alternatively they come as callbacks: `evaluate`, `mu_expansion`, `t3_expansion`, `t2_expansion` and their rank n+2 variants (`fast/numerator.rs:66-181`); in the generic engine, `evaluate_batch` / `expansion_batch` (`numerator.rs:72-85`).

**Why a coefficient-first recursion cannot sum across families.** Two loop currents with different denominator sets are different rational functions of q. Adding them as polynomials needs a common denominator, which multiplies numerators by the missing D_j. That raises the degree by 2 per factor and adds μ² (§4.1). This is the structural reason Recola keys loop currents by `ho/hm` [L 1211.6316 §2.2: "we are not allowed to sum them unless their denominators are equal … loop currents need an additional parameter, called sequence number"] and OpenLoops by topology.

**The exact loophole** (derived in the sibling report; made practical here).
- On any cut containing line i, D_i = 0, so D_j = D_j − D_i = 2q·(V_j − V_i) + (V_j² − m_j² − V_i² + m_i²) =: L_j^{(i)}(q). This is **affine in q and free of μ²**.
- Integrand reduction never evaluates the numerator off a cut:
  - box c₀ at the two 4-d quadruple-cut solutions;
  - box c₄ from the μ²→∞ expansion on the d-dimensional quadruple cut;
  - triangles and bubbles from large-t expansions on their cuts;
  - tadpoles from single-cut expansions [L 1403.1229 §2.2].
- So a numerator that depends on which cut it is called on, and is correct there, is a valid input. NinjaRS passes that cut to every callback (§3.2).

---

## 2. Recola 2.3.0, read from the source

### 2.1 The loop as a tree with two extra legs, one skeleton per cut particle [V]

- **Cut-particle loop.** `generate_currents` loops over the cut particle t (`cutsloop`, `recola2/core/currents_rcl.f90:842-845`; skipped if `loopCoef(t)=0`). It sets the extra legs `park(legs+1)=t` and `park(legs+2)=anti-t` (`:880-883`) and builds a full skeleton for the tree process A + B̄ + t + t̄ → 0 (`call makeSkeleton`, `:943`), followed by `filter_and_optimize` (`:947`, which calls `optimize_color`, `:2300`).
- **Shared fermion loops.** Fermion loops of different flavour but equal mass are computed once (`optimize_fermionloop`, `:2488+`; the multiplicity is folded into `loopCoef`).
- **Loop legs.** They get binaries 2^E and 2^{E+1} (`recola2/core/skeleton_rcl.f90:100-104`).
- **Currents.** Loop currents contain the loop binary and tree currents do not. Loop currents must have the quantum-field attribute and tree currents the background-field attribute (`:654-660`).
- **Order of construction.** Branches are generated in increasing order of the outgoing binary `eN` (`:141-177`) via `branch_partition` (`:199-229`). For 3- and 4-point vertices the first incoming binary of a loop branch is the loop current (`:386-398`).

### 2.2 Loop-current identity and the denominator history [V]

When a new branch is created, its outgoing current is looked up among existing currents with the same binary (`wloop`, `recola2/core/skeleton_rcl.f90:876-903`). It is reused only if **all** of the following match:

- antiparticle;
- binary;
- the full `hosT` vector;
- `hmaT`;
- `xxxT` (counterterm-insertion flag);
- `orderT` (coupling-order vector);
- `U1gT`;
- `csT` (colour structure);
- resonance data.

Otherwise a new current is registered (`:905-925`).

- **`ho` (propagator history).** For each external leg k, `ho(k)` is the index of the loop propagator after which leg k enters the loop.
  - The initial loop current has `ho = 0` and `hm = mass id of the cut particle` (`recola2/core/currents_rcl.f90:925-930`).
  - Each loop branch sets `hoOut = ho + vectorLeg(treeBins)·(maxval(ho)+1)`: the legs of the attached tree block enter at the next propagator (`skeleton_rcl.f90:587`).
  - The closing branch keeps `hoOut = ho` (`:585`).
  - So `ho` is exactly an **ordered block partition of the external legs**. Block j consists of the legs with `ho = j`, and block 0 enters at the closing vertex.
- **`hm` (mass history)** is a base-(n_masses+1) integer, extended by `sethm` (`:1483-1507`). Propagator extensions for massive vector bosons (R_ξ) are encoded by substituting the Goldstone mass (`:792-805, 842-850`). A comment notes that general R_ξ would need doubled propagators (`:588-593`).

### 2.3 Canonical cut, orientation and filters [V, L]

- **Cut rules** (`checkDoubleCounting`, `skeleton_rcl.f90:1388-1477`; rules 1–2 of [L 1211.6316 `recola.tex:2031-2038`]):
  1. only an offset containing leg 1 enters first;
  2. the three offsets with the smallest identifiers appear in increasing order.

  Each loop diagram is generated once, with a unique `ho`.
- **Tadpoles and external-leg self-energies** are rejected during skeleton construction (rules 3–4; `skeleton_rcl.f90:416-440, 269-277`).
- **Self-energy cut factor.** Self-energy cuts get a factor 1/2 [L `recola.tex:2040-2049`].
- **Residual scaleless TIs.** A few remain. For example, a massless tadpole from a quartic vertex with two non-external currents, or an external-leg bubble whose other side is a quartic vertex. They appear in the TI tables (§2.9) and are evaluated as zero by Collier [M].

### 2.4 Coefficient recursion [L, V]

A loop current after l loop branches is

w_l(q) = Σ_k d_{l,k}^{μ1…μk} q_{μ1}⋯q_{μk} / Π_{h=1}^{l} [(q+p_h)² − m_h²].

Each loop branch with a cubic vertex updates the coefficients as {d_l} = w_t({0, d_{l−1}} a^μ + {d_{l−1}, 0} b); quartic vertices use b only [L `recola.tex:2342-2405`]. The coefficients are symmetrised at each step. Rank increments come from the model vertex (`get_vertex_lorentz_rank_mdl`, `currents_rcl.f90:1129-1148`). Recola's rank is the maximum over contributions, not a structural bound per family.

### 2.5 How contributions with identical denominators become one tensor integral [V]

`set_tensor_parameter` (`currents_rcl.f90:3391-3416`) runs at every closing branch (`:1154-1158`):

1. it compares `(hosT, hmaT)` of the closing current with all registered TIs (`momsti`, `massti`);
2. on a match it records `tij(j)=k` and raises `rankti(k)` to the branch's output rank;
3. otherwise it registers a new TI with `legsti = maxval(ho)+1`, `momsti = ho`, `massti = hm`.

So:

- **Merging during the recursion.** Loop branches whose outgoing currents share `(ho, hm, …, colour)` are summed as currents. This is Berends–Giele with the history in the key.
- **Merging at registration.** All closing branches with the same `(ho, hm)` point to the same TI, across cut particles, because "not the particles but only their masses and momenta enter the sequence number" [L `recola.tex:2421-2435`].
- **The coefficients are not merged into one per-TI tensor.** Each closing branch is contracted with the TI immediately (§2.7).

### 2.6 Pinched topologies and the Collier cache [V, L]

- **No lifting.** Every pinched topology is its own TI.
- **Collier setup** (`recola2/api/collier_interface_rcl.f90:60-200`):
  - `Init_cll(nPropMax, rmax)` with the process maxima (`:93-96`);
  - in `cache_mode=1`, one cache per process and `InitCacheSystem_cll(nCacheTot, nPropMax)` with `SetCacheLevel_cll(n, nProp(pr))` (`:108-166`);
  - at runtime, TIs are evaluated in blocks with `InitEvent_cll(n)` (`recola2/core/amplitude_rcl.f90:771-778`);
  - `cache_mode=2` (global caches shared by processes) enforces an identical call sequence (`:632-680`).
- **How the cache works** [L 1604.06792 §4 and §5.5, `collier.tex:1525-1547, 2553-2592`]:
  - The reduction of one N-point TI calls lower-point integrals internally, so pinched sub-integrals are reached via several paths and from several TIs.
  - The cache compares arguments during the first phase-space point and then maps calls by position ("integrals are identified by their position in the sequence of user calls").
  - Every pinched sub-integral is therefore computed once per event, whether reached inside a parent's reduction or requested as its own TI.
- **Otter.** The newer backend registers all TIs up front (`register_integrals_otter`, `amplitude_rcl.f90:1698-1807`; used for `reduction_mode≥4`, `:731-767`).

### 2.7 Per-event flow and where the time goes [V, M]

**Flow** (`compute_amplitude`, `recola2/core/amplitude_rcl.f90:452-1645`):

1. Momenta and invariants.
2. **All TIs once**: `TNten_cll` per TI, giving tensor components to rank R (`:771-861`).
3. Tree branches per helicity configuration (`:941-1263`).
4. Loop branches, per c = (cut particle t, cut-line fermion helicity fh, first-offset class ih1) (`:1272-1279`; ih1 bounds memory, `currents_rcl.f90:1567-1585`).
   - Inside that: per `lmu` (cut-line spinor or polarisation index, 4 for vectors and fermions, `currents_rcl.f90:1241-1264`), per helicity configuration (`configloop1`), per branch.
   - At each closing branch: `wwTI = Σ_k ww1out(lmu,k)·TIri(k,ti)` is added to `matrix(cs, gs, i)` (`amplitude_rcl.f90:1470-1496`).
5. Dependent colour structures are rebuilt from primary ones (`:1589-1614`).
6. Born interference and helicity sum (`compute_squared_amplitude`).

**Measured** with the retained harness `recola-ninjars/recola-timing/time_rcl.f90`: Recola 2.3.0 library (it prints "Version 2.2.5") plus Collier 1.2.8 from `workspace/reference/sources`, RAMBO points at √s = 500 GeV, one core of this host.

| Process | ms/pt | TIs (Collier) | Tensor coefficients | Helicity sum | Generation |
|---|---|---|---|---|---|
| gg→gg | 11.1 | 0.57 (5 %) | 10.2 | 0.21 | 0.14 s [M, retained log `t_gg_gg/run.log`] |
| gg→ggg | 411 | 9.9 (2.4 %) | 389 | 11.5 | 6.6 s [M] |
| uu~→e⁻e⁺dd~g | 41.3 | 12.0 (29 %) | 29.2 | 0.04 | 6.6 s [M] |

**Reading.** Recola's tensor-integral layer is cheap because it is numerator-independent (one TI per family per event) and cached. Recola's cost is the helicity-explicit, colour-structure-dressed coefficient recursion. For the 4-quark + V process the TI layer is about 30 %. That is the realistic benchmark for pyAmpliCol's NinjaRS stage on quark-heavy processes (§6).

### 2.8 Colour and helicity in Recola [V, L]

- **Colour.** Currents carry explicit colour-flow structures (`csT`) [L `recola.tex:2497-3075`].
  - `optimize_color` marks branches that are identical up to a colour factor as daughter × factor of a mother branch, and computes them once (`currents_rcl.f90:2365-2485`).
  - With `colour_optimization=2` (default, `globals_rcl.f90.in:250`) U(1)-gluon flows are skipped (`skeleton_rcl.f90:815-820`) and dependent colour structures are reconstructed (`amplitude_rcl.f90:1589-1614`).
- **Helicities.** These are explicit configurations, 32 for gg→ggg ("Helicities = 32" in the summary).

### 2.9 Counts: tensor integrals versus diagrams versus loop currents [M, V]

Recola generation summaries and TI dumps (`recola-ninjars/recola-count/*/output.rcl`, `tis.txt`), classified with `analyse_tis.py`:

| Process | Recola TIs (all / non-scaleless) | Loop diagrams (source) | Recola loop currents / branches | Helicity configurations | Colour flows (computed) |
|---|---|---|---|---|---|
| gg→gg | 29 / 24 | 186 (FeynKit, pyAmpliCol filters; journal) | 1,786 / 5,930 | 16 | 24 (9) |
| gg→ggg | 160 / 154 | 2,075 (same) | 40,350 / 165,630 | 32 | 120 (44) |
| uu~→gg | 17 / 14 | — | 240 / 662 | 8 | 6 (3) |
| uu~→ggg | 94 / 90 | 556 (same) | 4,136 / 16,330 | 16 | 24 (11) |
| uu~→gggg | 633 / 628 | — | 80,768 / 363,406 | 32 | 120 (53) |
| uu~→e⁻e⁺dd~g | 246 / 244 | 1,668 (MadLoop, assessment) | 7,836 / 25,624 | 40 | 6 (4) |
| gg→gggg | not generated in 600 s (previous attempt) | 26,975 (MadLoop) | — | 64 | — |

- **Diagrams per family.** Typically 5–30, growing with multiplicity. For comparison, MadLoop+Ninja's loop groups: gg→tt̄ggg has 82,470 diagrams in 9,940 groups; uu~→tt̄bb̄dd̄g has 99,981 in 18,469 [L 1604.01363 `timing_table.tex`].
- **Recola keeps more loop objects than diagrams** (40,350 loop currents for 2,075 diagrams in gg→ggg). Loop currents are split by colour structure and cut particle, and then evaluated per helicity configuration and cut polarisation. This is why Recola is slow for many-gluon processes.

### 2.10 What to copy from Recola, and what not [I]

- **Copy:**
  - the current key (history, masses, counterterm flag, orders, colour state);
  - the TI-identity rule (masses and momenta only, so species share families);
  - the canonical cut rule (or a merging-friendly replacement, see the sibling's largest-block-first rule);
  - filtering tadpoles and external self-energies during generation;
  - a per-event cache for everything numerator-independent.
- **Do not copy:**
  - the amplitude-level contraction per helicity × cut polarisation × colour structure;
  - per-helicity loop currents.

  With integrand reduction the reduction cost is per numerator, so the numerator must be Born-weighted and helicity-summed first (§4.4).

---

## 3. Ninja and NinjaRS

### 3.1 The Laurent-expansion method [L 1403.1229 §2.2]

| Cut | What is computed | Subtractions |
|---|---|---|
| Pentagon | skipped ("the subtraction terms coming from five-point residues always vanish in the asymptotic limits") | — |
| Box | c₀ = ½[N/ΠD(q₊) + N/ΠD(q₋)] at the 4-d quadruple-cut solutions; c₄ = leading term of the μ²→∞ expansion on the d-dim cut (only if rank ≥ n) | none |
| Triangle | Large-t expansions on the two parametrisations q± = a + t e₃ + (β+μ²)/(2t e₃·e₄) e₄ (and swapped) | none; boxes and pentagons vanish in the average |
| Bubble | Expansion in t with free x, μ² | Coefficient-level corrections from the triangles containing it |
| Tadpole | Expansion in t | Corrections from bubbles and triangles |

The Laurent expansion of N/Π_{uncut} D is a **polynomial division** of the numerator expansion by the uncut denominators' expansions [L §3.2].

### 3.2 Interfaces [V]

- **Dense tensors.**
  - `TensorNumerator`: rank R, C(R+4,4) coefficients per lane in Ninja/MadLoop order. "Every expansion is a single transposed Horner sweep … each coefficient of the tensor is touched once per expansion" (`fast/tensor.rs:1-22`).
  - `MuTensorNumerator`: blocks T_k(q)μ^{2k} (`fast/tensor_mu.rs`).
- **Callbacks** (`FastNumerator`). All methods receive `cut` (the sorted cut partition) and `lanes`. "Implementations must not assume any relation between consecutive calls" (`fast/numerator.rs:61-66`).
- **Generic `Numerator<T>`.** Requests carry `partition`. "A callback cache must include lane, partition, expansion kind and sample" (`numerator.rs:4-9, 58-71`).
- **Original Ninja's contract** [L 1403.1229 §3.1]: "The returned expansion only needs to be valid for the cut specified." Samurai already had cut-dependent integrands (`xnum(icut,q,mu2)`, "the value of icut yields a selective access to the proper integrand") [L 1006.0710 `samurai.tex:1077-1100`].
- **Masters.** `FastMasters::evaluate(family, args, out, ok)` with family ∈ {A0, BubbleRank2/3/4, C0, D0} and floating arguments only (`fast/masters.rs:12-64`).

### 3.3 What one NinjaRS call does [V]

- `FastAmplitude::new(n, rank)` rejects rank > n+2 (`fast/engine.rs:624-632`).
- `Topology::new` lists all boxes, triangles and bubbles, and the subtraction parents within the same call (`:228-333`).
- Cut levels run only if the rank allows them (`any_triangle: rank−n+3 ≥ 0`, `any_bubble: +2`, `any_tadpole: +1`, `:673-684`). A heptagon of rank 3 therefore needs boxes only.
- Scaleless bubbles (massless lines, |s| < 10⁻⁹) are skipped before the callback (`:1256-1257`).
- All master requests of all cuts and lanes are collected and evaluated with one provider call per master type, then contracted (`:1746-1838`). Lanes are independent phase-space rows with their own shifts, masses and S-matrix (`:789-843`).
- **Original Ninja cached masters by argument** in its OneLOop wrapper and advised clearing the cache once per point [L 1403.1229 §3.3, App. C.1]. NinjaRS's providers have no result cache (`docs/benchmarks.md:65`).

### 3.4 Rank domain and accuracy cost [V]

- **Rank class.** NinjaRS reduces any n with rank ≤ n+2 in binary64, DoubleFloat and Float. Original Ninja stops at n+1 [L 1403.1229 App. B].
- **Speed** (NinjaRS benchmark, `docs/benchmark/data/results.json`): rank n+1 costs about 2× rank n (syn-n5: 192 vs 89 µs; syn-n6: 743 vs 416 µs). Rank n+2 costs 2.1–2.9× its n+1 twin (`docs/rank-n2.md:345-348`).
- **Accuracy.**
  - The n+2 terms divide by up to (e₃·k)⁵ (`:350-353`).
  - Rank n+2 pieces at n=7 lose accuracy like original Ninja (5.8e-9 to 1.6e-8 against ≤ 1.5e-12 generic) (`:891-897`).
  - Declaring rank n+2 for a rank n+1 numerator costs the fast engine up to 981× (`:825-829`).
  - Both binary64 engines lose digits on ill-conditioned n=7 lanes (`:898-903`).

### 3.5 Cost anatomy [M]

**Calibrated model** (`rn2/calib.py`). It is fitted to NinjaRS's ten synthetic rank ≤ n cases (EPYC 9754, width 32):

- numerator expansions = 1.354 ns × C(r+4,4) × (jet components per cut: boxes 2–2.25, triangles 2×t3_len(r−n+3), bubbles t2_len(r−n+2), tadpoles t3_len(r−n+1)); fits all ten cases within ±7 %;
- reduction bookkeeping = 40 ns per unit (cut visit plus uncut divisions × jet length); within about ±30 %.

**Probe** (`rn2/master_probe`). It is built against the pinned NinjaRS 0d0d7e7 with oneloopmaster native masters: 32 lanes, synthetic dense numerators of full rank, sub-families of one n-gon.

All 113 physical sub-families of a massless heptagon, separate reductions (µs per lane):

| Variant | Total | Callbacks | Masters | Rest |
|---|---|---|---|---|
| No cache | 2,847 | 1,895 | 440 | 512 |
| Naive bit-keyed master cache | 2,629 | 1,895 | 223 | 511 |

- The cache hit 38,528 of 41,216 master requests: 2,688 distinct.
- The naive `HashMap<Vec<u64>>` key costs about 0.17 µs per lookup. Integer master ids would make the cached cost negligible.

**Conclusion.** With masters cached, the numerator expansions (callbacks) are about 72–75 % of NinjaRS time. Geometry, division and subtraction are about 15–20 %. Any further gain must reduce the number of (numerator, cut) expansions.

---

## 4. Structural options

### 4.1 (a) Lifting pinched topologies into a parent by multiplying the missing denominators

**Algebra.** For a member family F ⊂ P with k = |P| − |F| missing lines:

N_F/Π_F D = [N_F Π_{j∈P∖F} D_j] / Π_P D,

where D_j = (q+V_j)² − m_j² − μ² is quadratic in q and linear in μ². The lifted numerator has weighted rank r_F + 2k and μ² powers up to k. With r_F ≤ |F| (renormalisable, Feynman gauge), the excess is (r_F − |F|) + k ≤ k. Within NinjaRS's n+2 bound, k ≤ 2. Using four-dimensional D_j would be wrong: it drops μ² N_F Π(…) terms that integrate to spurious rational parts.

**Counts and cost** [M/I, `rn2/lift_global.py`; greedy host cover]:

| Process (FC) | Hosts, rank ≤ n+1 | Model cost | Hosts, rank ≤ n+2 | Model cost | P0 model cost |
|---|---|---|---|---|---|
| gg→gggg | 301 | 47 ms | 158 | 158 ms | 39 ms |
| gg→ggggg | 2,225 | 0.80 s | 1,072 (954 at n+2) | 2.7 s | 0.79 s |

Masters are excluded from this table.

**Accuracy.** The earlier probe found 2.8e-4 (n=6) and 8.3e-3 (n=7) scaled errors, against 1e-7 to 5e-6 for separate reductions (`understand-recursion-literature.md` §4.3). Rank n+2 also brings the conditioning issues of §3.4.

**Literature.**
- Samurai's authors offered the same grouping but noted it "is not guaranteed to speed up the calculation as one moves the complexity … to higher values of n and at the same time allows the rank of the numerator to increase … process dependent and heuristic" [L 1006.0710 `samurai.tex:1829-1850`, commented-out text in the arXiv source].

**Verdict: reject.**

### 4.2 (b) Family-tree / integrand-sum reduction of a parent and its pinched families in one pass

#### 4.2.1 Exact algebra

Take a parent P (lines 0..N_P−1, shifts V_i, masses m_i) and members m with line sets S_m ⊆ P. Each member's numerator N_m(q) is expressed in P's loop-momentum frame: shift q → q+c and, if needed, reflection q → −q. Both are cheap if the recursion builds the member in P's frame (§7.2).

I_P = Σ_m N_m/Π_{j∈S_m} D_j = Σ_{C⊆P, |C|≤5} Δ_C/Π_{j∈C} D_j,   with Δ_C = Σ_{m: S_m⊇C} Δ_C^{(m)}.

- **Residues are linear.** On cut C, members with S_m ⊉ C vanish. The Laurent coefficients are Σ_{m: S_m⊇C} [expansion(N_m) ÷ Π_{j∈S_m∖C} D_j].
- **Subtractions are linear too.** A bubble subtracts the triangles containing it, Δ_{C∪{j}}/D_j. For members without line j, Δ_{C∪{j}}^{(m)} = 0, so the parent-level subtraction with summed residues equals the sum of member-level subtractions.
- **Masters and geometry** of cut C depend only on the cut, so they are shared.

**The answer to "does Ninja's method support division by each member's own extra uncut denominators?"**: yes, mathematically, and exactly. It needs either an API that divides each member separately (b2), or the identity below (b1), which turns the per-member division into a per-cut lifted numerator.

#### 4.2.2 Two realisations

- **(b2) Per-member division.** NinjaRS would take members {(S_m, rank_m, numerator_m)}. Per parent cut it would:
  1. build the geometry once;
  2. expand the uncut denominators of P∖C once;
  3. expand each member containing C, and divide it by its own S_m∖C;
  4. sum;
  5. apply subtractions once, call the masters once, integrate once.

  Numerically it is the same as separate reductions, except that the cut basis is chosen with the parent's uncut lines. The cost is the same as separate reductions minus the shared share of "rest", which is ≤ 15–20 % of NinjaRS time (§3.5).
  - Needs a NinjaRS API.
  - Gains little.
  - Useful as a **precision fallback**.

- **(b1) Cut-restricted lift with per-reference tensors.** For each reference line i ∈ P,

  T_i(q) = Σ_{m: i∈S_m} N_m(q) Π_{j∈P∖S_m} L_j^{(i)}(q),   L_j^{(i)} = 2q·(V_j−V_i) + V_j² − m_j² − V_i² + m_i².

  - deg T_i = max_m (r_m + k_m) ≤ N_P in a renormalisable theory, so **rank class n+0 is preserved**. T_i contains no μ² beyond what the N_m carry.
  - On every cut C ∋ i, T_i equals the lifted numerator Σ_m N_m Π_{P∖S_m} D_j. A callback answering each request with T_{i(cut)}, for some i(cut) ∈ cut, is therefore a valid numerator for the parent. NinjaRS returns Σ_m (reduction of member m).
  - **Box μ²-expansion check.** On the d-dimensional quadruple cut each L_j^{(i)} is O(t) (t = √μ²). The leading coefficient of T_i is the product of leading coefficients, and dividing by the leading coefficients of the N_P−4 uncut parent denominators gives Σ_m N_m[t^{r_m}]/Π_{S_m∖C} d_j. That is the correct c₄ (only full-rank members contribute).
  - **No API change:**
    - fast engine: dispatch on `cut[0]`;
    - generic engine: dispatch on `partition`.
  - **The reference must be a line of the cut.** A fixed reference is wrong by O(1) to O(10⁵) (sibling probe, variant A0).
  - **Cost:**
    - Callbacks become those of the parent alone. This removes the dominant 75 % share for all non-parent members.
    - Plus the per-event build of the T_i: Σ_m |refs ∩ S_m| Σ_{s<k_m} 5·C(r_m+s+4,4) complex MACs per lane.
    - Only a vertex cover of the cuts NinjaRS actually processes is needed as reference set. For a massless heptagon with single-leg blocks, 5 of 7 lines suffice (adjacent pairs are scaleless).

#### 4.2.3 Measurement [M, `rn2/master_probe/group32.log`]

Set-up:
- binary64 fast engine, oneloopmaster native masters, 32 lanes;
- RAMBO kinematics, synthetic dense full-rank numerators;
- one parent group with the **average composition** of the balanced assignment of §4.2.4 (n=6: 1+3+3+2 members of sizes 6/5/4/3; n=7: 1+4+5+3+1 of sizes 7/6/5/4/3);
- the T_i build is timed, using a naive dense `affine_mul` over the NinjaRS monomial layout;
- error = max over lanes of |x − Σ_m ref_m| / Σ_m |ref_m| (finite and 1/ε);
- rotation noise = the same for separate reductions with each family's propagators rotated by one.

| Case | Separate (cached masters) µs/lane | (b1) one call + build µs/lane | Speed-up | (b1) error | Rotation noise |
|---|---|---|---|---|---|
| n=6 massless, 9 members (2 seeds) | 295 / 294 | 143 + 57 / 143 + 54 | 1.5× | 1.1e-6 / 6.7e-7 | 2.1e-6 / 2.8e-7 |
| n=6 massive | 493 / 492 | 262 + 55 / 260 + 55 | 1.55× | 2.5e-6 / 4.3e-11 | 1.8e-6 / 8.7e-11 |
| n=7 massless, 14 members | 1,114 / 1,106 | 380 + 188 / 382 + 171 | 2.0× | 4.3e-6 / 0.52* | 6.7e-6 / 0.24* |
| n=7 massive | 1,629 / 1,625 | 650 + 172 / 649 + 171 | 2.0× | 3.1e-6 / 1.3e-8 | 9.4e-6 / 9.4e-9 |

\* One random lane is ill-conditioned for every method: the rotated separate reductions disagree by 24 % themselves. A production stability trigger must catch such lanes whatever the strategy.

Inside the (b1) call at n=7 massless: callbacks 276 µs, masters 66 µs, rest 38 µs, against 838 / 105 / 171 µs for the separate reductions. The build (98,175 complex MACs per lane) runs at about 1.8 ns per complex MAC in this naive form. A layout-aware SIMD build should reach about 0.6 ns.

#### 4.2.4 Policy [I]

- **Parents.** Maximal families (no strict superset in the family set). In LC these are the planar n-gons of the orderings.
- **Assignment.** Process families in decreasing N. Assign each to the admissible maximal ancestor with the fewest members so far. The "first ancestor" rule gives one parent 113 members at n=7 and wrecks balance.
- **Admissibility:**
  - *strict*: r_m + (N_P − N_m) ≤ r_P, so the parent's rank class is not raised. Use this for quark processes, where mixed quark-gluon loops have rank N−1 or N−2.
  - *loose*: ≤ N_P.

  Non-admissible families are reduced on their own (stage 1).
- **Fallback.** Lanes flagged `Unstable` by NinjaRS are re-reduced family by family (stage 1), then in DoubleFloat.

#### 4.2.5 Literature precedents [L]

- **CutTools.** "One does not need to repeat the work for all Feynman diagrams, provided their sum is known" (sampling-based OPP) [0711.3596 `main.tex:327-331`].
- **GoSam 1.0** partitions diagrams into groups with a maximal element under the "appears in the reduction of" pre-order (S-matrix row/column deletion = pinching): "while carrying out the tensor reduction for the diagram max V_i, all other diagrams in the same cell V_i are reduced with virtually no additional computational cost" (OPP sampling and tensor reduction) [1111.2034 `diagen.tex:86-120`].
- **GoSam 2.0** keeps that grouping "when Samurai or Golem95C are used" and adds `diagsum` (identical propagator sets). A sentence in the arXiv source, commented out, says it "is not implemented for the new integrand reduction method based on the Laurent expansion" [1404.7096 `newfeatures.tex:23-35`].
- **MadLoop** groups only loops with the same ordered propagator list [1604.01363 `main.tex:920`].

The cut-restricted lift is the form of GoSam-1.0 grouping that is compatible with Ninja's polynomial expansions.

### 4.3 (c) OpenLoops on-the-fly reduction (OFR)

**Algebra** [L 1710.11452 §4.1, `OL_OFRED_paper_epjc.tex:1771-2038`]. The del Aguila–Pittau identity is

q^μq^ν = [A_{−1} + A_0 D_0] + [B_{−1,λ} + Σ_{j=0}^{3} B_{j,λ} D_j] q^λ.

- A and B depend only on p₁, p₂, p₃ of the first three segments.
- In D dimensions, D_j → D̄_j − q̃² adds −q̃²A_0.
- It holds at the integrand level for N ≥ 4 and only after integration for N = 3.
- Each step turns a rank-R open loop into a rank-(R−1) N-point object plus four rank-(R−1) (N−1)-point pinched objects. These are merged into existing objects when their undressed segments coincide, otherwise after extra dressing steps (§4.2).
- What remains: N ≥ 5 at rank ≤ 1, N = 4, 3 at rank 1, N = 2 at rank ≤ 2.
- The direction is parent → child (it **pinches**); it never lifts.

**Can it only keep ranks low and still feed NinjaRS per-family numerators?**
- Yes for the N ≥ 4 steps: outputs are polynomials (with μ² terms, so `MuTensorNumerator`) over subsets of a family's denominators.
- No for N = 3: the identity holds only after integration. NinjaRS would get rank ≤ n−3 numerators for N ≥ 5, which need boxes only.

**Why not:**
1. OpenLoops' own benchmark: "using OpenLoops2+Collier, i.e. restricting the on-the-fly approach to helicity sums plus diagram merging and reducing tensor integrals with Collier, can result in a further significant speed-up". OFR is kept for stability [L §6.2.1, `:3810-3850`].
2. The Gram-type coefficients need expansions and hybrid precision [L §5.4].
3. For quark processes the pinch closure adds families that no diagram produces: uu~→e⁻e⁺dd~g goes from 244 to 422 [M, `rn2/required_analysis.json`].
4. It would be a private reimplementation of reduction mathematics next to NinjaRS (user rule).
5. After (b1) plus the master cache, NinjaRS is not the bottleneck.

**Niche use.** A few integrand-level OFR steps could bring a rank > n+2 numerator (EFT, general R_ξ) into NinjaRS's domain.

### 4.4 (d) Born-dual weighting, helicity sums and multiple numerators per family

- **Born weighting before reduction is mandatory with integrand reduction.** Without it the count is families × helicities × colour structures: 9,414 × 128 × 720 ≈ 8.7×10⁸ for gg→ggggg, and 244 × 40 × 4 ≈ 3.9×10⁴ for uu~→e⁻e⁺dd~g. Recola can afford amplitude-level outputs only because tensor integrals do not depend on the numerator (§2.7).
- **Helicity sums on the fly.** U_k = Σ_{h_k} U_{k−1} S_k(h_k) [L 1710.11452 §3.1] makes the family numerator helicity-summed before closure, so reductions do not scale with helicities.
- **Several numerators per family** (coupling-order sectors, ε-sectors in direct R2, directed colour/spin correlations, simultaneous LC/NLC/FC) multiply the NinjaRS work. Two remedies:
  1. extra lanes, which duplicates only cheap geometry;
  2. a **transposed reduction**: NinjaRS computes once per family the linear functional (in effect the family's tensor integrals in the monomial basis) and contracts it with every numerator at C(R+4,4) cost each. The NinjaRS owner's n3/r3 symbolic study finds the transposed graph cheaper than forward from H > 1.85 (factored) or H > 3.97 (fused) numerators sharing one geometry (`ninjars/docs/audits/transpose-probe.md`).

### 4.5 (e) Other structural ideas

| Idea | Value | Evidence |
|---|---|---|
| **Per-event master cache** keyed by a generation-time master id; the cut's canonical invariants and masses, with S-matrices from one invariant table | Removes 85–109× redundant master calls at 7 legs (8–33× at 6) | [M] requests/distinct: gg→ggggg FC 308,448 / 2,822; uu~→ggggg FC 146,576 / 1,724; uu~→e⁻e⁺dd~g FC 3,433 / 296 |
| **Families (or groups) as lanes.** One `FastAmplitude` per (n, rank); lanes = (family, point, channel), since shifts, masses and S-matrix are per lane | Physical calls per event fall from #families to 5–11 (P0) or 1–6 (groups); larger batches | [V] `fast/engine.rs:789-843`; [M] `rn2/batch_classes.py` |
| **One canonical invariant table per event** shared by all families | Makes master keys bit-exact. Replaces per-family re-parsing of exact kinematics (7.8 ms/pt for 94 families today, `understand-contracts.md` K4) | [I] |
| **Declare the true structural rank per family** | Low-rank families skip whole cut levels (r < n−3: boxes only). Avoids the over-declaration accuracy loss | [V] `fast/engine.rs:673-684`; `docs/rank-n2.md:825-829` |
| **LC planar parents** | LC families = coarsenings of the planar parents of the LC orderings. Groups = orderings (gg→ggggg: 360; uu~→ggggg: 120; uu~→e⁻e⁺dd~g: 4) | [M] §5 |
| **Reference cover** for (b1) | 5 instead of 7 lifted tensors per massless heptagon | [I] |
| **Not pursued: evaluation-first callbacks** (recursion on Laurent jets per cut) | Full L3 sharing inside the recursion, but about 98 cuts × jet arithmetic per primitive; 50–100× costlier at n ≤ 7 | sibling `understand-recursion-literature.md` §3.3 |
| **Not pursued: cross-family cut-geometry cache** | Geometry, division and subtraction are 15 % of separate, 10 % of grouped NinjaRS time | [M] §3.5 |
| **Not pursued: supersymmetric/second-order rank lowering** | Needs per-primitive helicity amplitudes; conflicts with Born weighting before reduction | [I] |

---

## 5. Counts for the required processes

### 5.1 Method and validation

- **Families** are cyclic sequences of blocks (sets of external legs attached at one loop vertex) with a mass label per propagator, up to rotation and reflection. Tadpoles and external single-leg bubbles are excluded, as are scaleless massless families.
- **The enumerator** (`rn2/families.py`, from the earlier attempt) reproduces **exactly** the non-scaleless Recola TI sets [M, `validate_enum.py`]:

  | Process | Recola non-scaleless TIs = enumerated families |
  |---|---|
  | gg→gg | 24 |
  | gg→ggg | 154 |
  | uu~→gg | 14 |
  | uu~→ggg | 90 |
  | uu~→gggg | 628 |

- **Ranks:**
  - gluon, ghost and closed-quark loops: N;
  - mixed quark-gluon loops (q and q̄ in different blocks): N−1. This matches Recola's maximal ranks on 4 processes (`check_ranks.py`);
  - uu~→e⁻e⁺dd~g: Recola's per-TI ranks (N−2: 100 families; N−1: 118; N: 26).
- **LC sets:**
  - gg→nG: all massless families (closed quark loops are N_f/N_c, top loops dropped).
  - uu~→ggggg: q and q̄ in the same or cyclically adjacent blocks. This is identical to "coarsenings of the 120 planar parents (q, q̄, σ)" [M, `lc_check.py`], the BDK A^L structure.
  - uu~→e⁻e⁺dd~g: coarsenings of the four planar parents (u –[g?]– d –[V?]– d̄ –[g?]– ū –[V?]– u), with g on a gluon arc and V on a quark arc, intersected with Recola's families.
    - Why g sits on a gluon arc: a gluon emitted inside the loop from a quark arc is 1/N_c-suppressed. Example: the abelian q q̄ g vertex has colour C_F − C_A/2 = −1/(2N_c) [I, from the colour algebra].

### 5.2 Families, parents and per-option NinjaRS reductions per event [M]

Sources: `rn2/required_analysis.json`, `groups.json`, `lift_global.py` output.

| Process | Colour | Families by N (2,3,4,5,6,7), massless + massive | Maximal parents | (a) hosts ≤ n+1 / ≤ n+2 | (b) groups, loose (strict = groups + alone) | (c) OFR pinch closure | (d) no Born weighting | Physical calls P0 → groups |
|---|---|---|---|---|---|---|---|---|
| gg→gggg | FC | 1,100 = 2×(25, 90, 195, 180, 60) | 120 (2×60 hexagons) | 301 / 158 | 120 (median 9 members) | 1,100 | 8.4×10⁶ | 5 → 1 |
| gg→gggg | LC | 550 | 60 | 152 / 80 | 60 | 550 | 4.2×10⁶ | 5 → 1 |
| gg→ggggg | FC | 9,414 = 2×(56, 301, 1,050, 1,680, 1,260, 360) | 720 (2×360 heptagons) | 2,225 / 1,072 | 720 (median 13) | 9,414 | 8.7×10⁸ | 6 → 1 |
| gg→ggggg | LC | 4,707 | 360 | 1,113 / 537 | 360 | 4,707 | 4.3×10⁸ | 6 → 1 |
| uu~→ggggg | FC | 5,258 = massless (56, 301, 1,050, 1,680, 1,260, 360) + top (26, 90, 195, 180, 60) | 420 (360 heptagons + 60 top hexagons) | 780 / 506 | 420 (strict: 420 + 551 = 971) | 5,258 | 4.0×10⁷ | 11 → 3 (strict 6) |
| uu~→ggggg | LC | 2,712 = (56, 301, 765, 930, 540, 120) | 120 | 361 / 189 | 120 (strict: 120 + 551 = 671) | 2,712 | 2.1×10⁷ | 11 → 1 (strict 6) |
| uu~→e⁻e⁺dd~g | FC | 244 = massless (17, 39, 83, 72, 20) + top (5, 5, 3) | 39 (20 hexagons, 16 pentagons, 3 top boxes) | 49 / 40 | 39 (strict: 39 + 27 = 66) | 422 (+178) | 3.9×10⁴ | 10 → 5 (strict 6) |
| uu~→e⁻e⁺dd~g | LC | 115 = (17, 36, 38, 20, 4) | 4 hexagons | 17 / 9 | 4 (strict: 4 + 79 = 83) | 161 (+46) | 1.8×10⁴ | 9 → 1 (strict 7) |

Notes:
- (d) counts helicity configurations × a representative colour-basis size: 2^n (64 for massless qq̄ lines); (n_g−1)! or (n_g+n_q)! colour flows; 40 helicities and 4 flows from Recola for uu~→e⁻e⁺dd~g. It is illustrative only.
- Physical calls assume families-as-lanes batching by (N, declared rank).

### 5.3 Master redundancy (rank-aware, scaleless skipped) [M]

| Process | Per-family master requests FC (LC) | Distinct masters FC (LC) | Redundancy FC |
|---|---|---|---|
| gg→gggg | 20,800 (8,410) | 627 (310) | 33× |
| gg→ggggg | 308,448 (134,183) | 2,822 (1,407) | 109× |
| uu~→ggggg | 146,576 (62,673) | 1,724 (1,122) | 85× |
| uu~→e⁻e⁺dd~g | 3,433 (1,077) | 296 (133) | 12× |

---

## 6. Estimated NinjaRS stage per phase-space point [I]

Assumptions:
- binary64 fast engine, one core of this host class;
- calibrated model (§3.5) for expansions and bookkeeping;
- masters 'lo–hi' = (D0 2–15, C0 0.5–2, bubble 0.1–0.3, A0 0.1) µs. The probe measured about 0.8 µs (massless) and 2.7 µs (massive) per computed master, so massless processes sit near 'lo';
- (b1) build at 0.6–1.8 ns per complex MAC.

Uncertainty is about ×2. The ratios between columns are more reliable: (b1)/separate is directly measured at 1.5–2.0×.

| Process | Colour | P0, no master cache | P0 + master cache | (b1) groups + cache | (a) ≤ n+2 lift |
|---|---|---|---|---|---|
| gg→gggg | FC | 51–116 ms | 40–45 ms | 19–29 ms | ≈160 ms |
| gg→gggg | LC | 23–55 ms | 17–20 ms | 9–14 ms | — |
| gg→ggggg | FC | 1.0–2.3 s | 0.79–0.82 s | 0.30–0.40 s | ≈2.7 s |
| gg→ggggg | LC | 0.48–1.1 s | 0.36–0.38 s | 0.14–0.19 s | — |
| uu~→ggggg | FC | 0.31–0.97 s | 0.19–0.21 s | 0.11–0.16 s (strict 0.09–0.13) | — |
| uu~→ggggg | LC | 0.13–0.40 s | 79–90 ms | 43–64 ms | — |
| uu~→e⁻e⁺dd~g | FC | 3.9–16 ms | 1.9–4.2 ms | 1.4–3.9 ms | — |
| uu~→e⁻e⁺dd~g | LC | 1.1–4.4 ms | 0.5–1.4 ms | 0.4–1.3 ms | — |

**Context.**
- Recola spends 12 ms per point on tensor integrals for uu~→e⁻e⁺dd~g, and 41 ms in total, on this host [M]. Staying within 1.3× of Recola leaves about 54 ms for everything. The NinjaRS stage (2–4 ms with the cache) fits comfortably; without the cache it can take up to 30 % of that budget.
- For 7 gluons the NinjaRS stage at 0.3–0.4 s is small against any realistic total. OpenLoops 2 needs 21 s for gg→tt̄ggg on a 2014 CPU [L 1907.13071 Table 8]. MadLoop+Ninja needs 0.78 s of Ninja time for gg→tt̄ggg's 9,940 groups [L 1604.01363 App. B].
- The numerator recursion remains the main cost (sibling toy model: 3.3×10⁹ MAC for the 7-gluon gluon loop).

---

## 7. Recommended combination and implementation sketch

### 7.1 The combination

1. **Stage 1 (with the new engine):**
   - Recola-keyed, Born-seeded OpenLoops-2 recursion over Berends–Giele blocks (sibling design);
   - one Born-weighted, helicity-summed tensor per (family, channel) at true rank;
   - NinjaRS fast engine, families batched as lanes by (N, rank);
   - per-event master cache;
   - stage 1 is the reference and fallback.
2. **Stage 2 (opt-in, then default if validated on physical numerators):**
   - parent groups with the cut-restricted lift (b1), strict policy for quark processes and loose for pure-gluon ones;
   - LC uses planar parents;
   - per-lane fallback to stage 1, then to DoubleFloat.
3. **Never:** global lift, OFR, or amplitude-level reduction.

### 7.2 Data structures [I]

```text
FamilyCatalogue                      (generation; replaces FeynKit enumeration for large processes)
  families[f] = {
    key:   canonical (cyclic block sequence of leg masks, propagator mass ids)   # Recola (ho, hm)
    lines: [{shift: leg-mask, mass: mass_id}]  in canonical order (= NinjaRS shift order)
    rank:  structural max over closing contributions (Recola rankti analogue)
    lc:    bool (member of an LC planar parent's coarsenings)
    group: (parent g, line_map S_m -> parent lines, frame: shift mask + reflection flag) | None
  }
  groups[g] = { parent f_P, members [f], refs: vertex cover of processed cuts, rank_class }
  buckets[b] = { n, declared rank, entries: [family | group] }      # one FastAmplitude per bucket
  masters: canonical cut -> master id   (cut = ordered invariant masks + mass ids, type)
  invariants: leg-mask -> slot          (one table per process)

Recursion plan (sibling design): open-loop objects keyed by
  (family or prefix history, undressed suffix, loop-line state, seed colour component, orders, CT flag),
  closing into family accumulators acc[f][channel][monomial], built in the group-parent frame
  (frame shift folded into segment constants).
```

### 7.3 Per-event flow [I]

1. Exact external momenta give the invariant table s[mask], the per-family shifts V_i and the S-matrices, all from one table and rounded once to the carrier.
2. Trees: Berends–Giele currents with helicity selection. Born, then Born-dual weights in the colour basis, then seeds.
3. Loop recursion: dress, merge, sum helicities on the fly. The output is acc[f][ch] dense tensors at true rank.
4. Stage 2 only: for each group and each reference i, T_i = Σ_{m∋i} acc[m] Π_{j∈P∖S_m} L_j^{(i)} (dense affine products).
5. NinjaRS, one call per bucket:
   - lanes = (family or group) × point × channel;
   - numerator: `TensorNumerator` (stage 1) or `CutReference{T_i}` (stage 2);
   - masters: the cached provider over oneloopmaster.
6. Per point: sum the lanes, add UV/R2 counterterms, normalise, run the IR-pole check (free trigger). Unstable lanes go to stage 1 per family, then to the generic engine in DoubleFloat.

### 7.4 What to ask the NinjaRS owner, in priority order

| # | Request | Why | Acceptance |
|---|---|---|---|
| 1 | `CachingMasters<M: FastMasters>` and `CachingMasterProvider<V>`: per-event cache with explicit `clear()`, keyed by an optional caller-supplied integer master id per request row, falling back to exact argument bits | 85–109× redundancy at 7 legs. Original Ninja shipped this (AvHOneLoop). Collier's cache is the Recola analogue. | Bitwise-identical results with and without the cache. Hit rate reported. Per-lookup cost ≤ 20 ns. |
| 2 | Layout-aware `TensorNumerator::add_affine_product(src, rank, affine_factors[])`, possibly μ²-blocked, plus a `CutReferenceNumerator` adapter (dispatch by cut to per-reference tensors; reference = best-conditioned line of the cut) — or natively `FastAmplitude::evaluate_group(parent_input, members)` that builds T_i lazily | The (b1) group reduction without pyAmpliCol reimplementing the tensor layout (user rule). Measured 1.5–2× at n = 6–7. | Equals the sum of separate reductions within rotation noise on the probe suite and on physical gg→ggg and uu~→e⁻e⁺gg groups. |
| 3 | Per-lane rank (`lane_rank`, `lower_rank`, already on the integration branch) and a test that one `FastAmplitude` call may carry lanes of unrelated families (different shifts, masses, S-matrices) | Families-as-lanes batching. Mixed ranks without over-declaration (981× risk). | Batched equals unbatched, bitwise. |
| 4 | Multi-numerator (vector or transposed) reduction: H numerators sharing one geometry and master set | Coupling-order sectors, directed correlations, simultaneous LC/NLC/FC. Transposition pays from H ≥ 2–4 (owner's own study). | H separate calls equal one vector call. Timing reported against H. |
| 5 | Optional exact `GroupNumerator` with per-member division (b2) | Precision fallback for groups whose lift is ill-conditioned | Equals separate reductions to rounding |
| 6 | Per-cut conditioning diagnostics exposed per lane (minimal uncut-denominator leading coefficient, Gram determinant) | Decide group or separate per lane, cheaply | — |

### 7.5 Validation and next measurements

1. **Catalogue.** Extend the set-equality check with Recola to gg→gggg; run generation in the background, since it exceeded 600 s. Also check against FeynKit `IntegralFamily` for the small controls.
2. **Physical numerators.** Repeat §4.2.3 with physical Born-weighted family numerators: the inherited per-diagram path for gg→ggg, uu~→e⁻e⁺gg and the 7-leg multiquark control. The synthetic tests may overstate or understate cancellations.
3. **Master cache.** Integer ids; measure hit rates and times on the 6-leg controls.
4. **Batching.** Families-as-lanes widths (32 points × 13 families) against throughput.
5. **Rescue rate.** The stability-trigger rate for groups against separate reductions on 10⁴ points.

---

## Appendix A. Reproduction

```sh
source /shared/vhirschi/t3code/pyamplicol/workspace/env.sh
cd /shared/vhirschi/t3code/pyamplicol/workspace/tmp/engine-v2/rn2
PY=/shared/vhirschi/t3code/pyamplicol/workspace/venv/bin/python
$PY -I calib.py $PYLOOP_WORKSPACE/cache/cargo/git/checkouts/ninjars-3842d69907c1a3a3/0d0d7e7/docs/benchmark/data/results.json
$PY -I analyse.py          # families, parents, options, P0 cost   -> required_analysis.json
$PY -I groups.py           # (b1) assignment policies and costs   -> groups.json
$PY -I lift_global.py      # (a) global-lift hosts and costs
$PY -I batch_classes.py    # physical calls with families-as-lanes
$PY -I lc_check.py         # LC rule == coarsenings of planar parents (uu~->ggggg)
(cd ../recola-ninjars && $PY -I validate_enum.py)   # enumerator == Recola TI sets
export CARGO_TARGET_DIR=$PWD/master_probe/target
(cd master_probe && cargo build --release --offline && SYMBOLICA_LICENSE=... LANES=32 ./target/release/master_probe)
# Recola timing (harness from the earlier attempt):
#   printf "g g -> g g g\n3 5 5\n" + rambo.py 3 20 7 500.0  |  recola-ninjars/recola-timing/time_rcl
```

Logs:
- `rn2/master_probe/run32.log`: master cache on all 113 sub-families;
- `rn2/master_probe/group32.log`: (b1) against separate reductions;
- `rn2/timing/{gg_ggg,uu_eeddg}/run.log`: Recola timings.

## Appendix B. Sources read

| Source | Parts |
|---|---|
| `recola_1.pdf` = arXiv:1605.01090 | Whole text (pdftotext per page, `recola-ninjars/pdf/recola_1.txt`): §2 (TIs/TCs, structure-dressed amplitudes, squared amplitudes), §4 API, App. B checks |
| `recola_2.pdf` = arXiv:1711.07388 | Whole text: §2 (renormalisation, BFM "10–15 % faster … generation 50 % longer"), §3–4 API |
| arXiv:1211.6316 (Recola algorithm) | §2.2 one-loop recursion and sequence numbers, §2.3 R₂/CT, §2.4 colour, §3.1.5 accuracy and timing (`recola.tex`) |
| Recola2 2.3.0 source | `skeleton_rcl.f90`, `currents_rcl.f90`, `amplitude_rcl.f90`, `collier_interface_rcl.f90`, `globals_rcl.f90.in` (lines cited) |
| arXiv:1604.06792 (Collier) | §4 (structure, cache system), §5.5 (using the cache system) |
| `ninja.pdf` = arXiv:1403.1229 | Whole text: §2 method, §3 interface and division, §3.3/App. C cache, App. B higher rank |
| NinjaRS 0d0d7e7 | README; `numerator.rs`; `fast/{numerator,tensor,tensor_layout,masters,engine}.rs`; `docs/{rank-n2,benchmarks,denominator-reuse,reduction-kernel-profile}.md`; `docs/audits/{transpose-probe,integrated-weight-fusion}.md`; `docs/benchmark/data/results.json` |
| arXiv:1710.11452 (OpenLoops OFR) | §4 (identity, master formula, pinch merging), §6.2.1 speed |
| `openloops2.pdf` = arXiv:1907.13071 | Table 8 timings |
| arXiv:1604.01363 (MadLoop+Ninja) | Loop groups, timing tables |
| arXiv:1404.7096, 1111.2034 (GoSam 2.0, 1.0) | Grouping and `diagsum`; pinch pre-order and groups |
| arXiv:1006.0710 (Samurai) | `xnum(icut,…)`; Born-contracted subamplitudes; optional pinched grouping |
| arXiv:0711.3596 (CutTools) | §2 "at the amplitude level … provided their sum is known" |
| Local | Assessment §1–2, §4 R1–R5; MASTER_PLAN §2.11–2.18; workflow journal entries (generation, dependencies); sibling engine-v2 reports |
