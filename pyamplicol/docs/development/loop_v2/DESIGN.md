# pyAmpliCol one-loop engine v2: final design

**Date** 2026-10-20, revision 2 (same day; supersedes revision 1). **Host** branch `loop` @ 4faafcad (read only for this design). **File** `workspace/reports/engine-v2/loop-engine-v2-design.md`.

This is the implementation design of the shared-recursion one-loop engine ("v2") that replaces the inherited per-Feynman-diagram engine. It consolidates the seven `understand-*.md` reports, the three candidate designs and the three judgements in this directory, and adds checks made for this document: code and NinjaRS source re-reads, provider timings, and a literature pass. Where it departs from a candidate design or from `MASTER_PLAN.md`, it says so.

**Revision 2** resolves the two adversarial reviews of revision 1 (literature; NinjaRS implementation). Each issue and its resolution is listed in the final section, "Review resolution". The revision adds new measurements, all under `workspace/tmp/engine-v2/revision/`:

- production-mode provider timings on this host;
- a Rust segment-kernel probe;
- schedule-B counts and dressing work in the providers' configuration (massless b), with a dense-kernel upper bound from the same model;
- gg→tt̄e⁻e⁺g counts;
- a FeynKit family check.

**Evidence labels**

| Label | Meaning |
|---|---|
| [V] | Verified for this document: `file:line` read or command run. |
| [V-s] | Verified by a sibling report or judge in this workflow (location given), not re-read here. |
| [L] | Literature read in source in this workflow (arXiv id, section, label or line). |
| [L*] | Literature known from abstract or search summary only. |
| [M] | Counted or modelled by workflow scripts (Appendix B). |
| [I] | Inference, estimate or design judgement. |

**Paths.**

- Python: `src/pyamplicol/…`; Rust: `rust/…`; both relative to `pyamplicol/`.
- `ninjars:` is the checkout `workspace/cache/cargo/git/checkouts/ninjars-3842d69907c1a3a3/0d0d7e7/crates/ninjars/src/`, where the [V] line references were verified. The pinned revision is now aa6a6307 (`…/aa6a630/`, a fast-forward); line numbers there may differ.
- `revision/` is `workspace/tmp/engine-v2/revision/`.
- "This host" is the AMD EPYC 9754 used for all local measurements, one pinned idle core.

---

## 1. Executive summary

### 1.1 The engine in one paragraph

For every denominator family *f*, squared-order sector *s* and requested observable *o*, v2 produces the four-dimensional numerator N_{f,s,o}(q) = Σ_α c_α q^α that NinjaRS reduces. The numerator is Born-weighted and helicity-summed. v2 never enumerates a loop or counterterm Feynman diagram. Three recursions share all the work:

1. **Trees.** pyAmpliCol's colour-ordered Berends–Giele (BG) recursion computes, once per phase-space point:
   - every off-shell current over every subset of external legs that a loop vertex can absorb (the *blocks*);
   - the Born colour components;
   - from wave 2, every tree that contains exactly one counterterm vertex.
2. **Topologies.** A Recola-type recursion over *loop prefixes* (ordered leg sets with propagator species and masses) produces the denominator families as closed histories. Per family it produces the admissible colour-ordered *skeletons*: rings of blocks, each block summing all trees on its leg subset. Skeletons are never stored as a list.
3. **Loop numerators.** An OpenLoops-2-type open-loop recursion dresses skeletons one segment at a time with polynomial-valued objects. The colour-summed Born interference is the starting value (the *seed*). The helicities of each block are summed as soon as it is attached. All partial results that share their key are added.

Two production schedules run on the one generator:

- **B** keys partial results by family and undressed remainder (a per-family suffix trie). It is generic and is the default for n ≤ 7 until gate G1.
- **M** puts Born seeds per colour ordering at the cut and shares prefixes across families. It is asymptotically better, generic at leading colour, and valid in full colour for certified colour classes.

G1 chooses between them per process class by measurement.

Families close directly into NinjaRS `TensorNumerator` lanes. Families with equal (propagators, rank) are reduced together as lanes of one call. Master integrals are cached per event. Every family is reduced a second time in a relabelled frame as the default accuracy check.

Colour lives only in exact seed tables (Laurent polynomials in N_c). Leading colour (LC), next-to-leading colour (NLC) and full colour (FC) therefore differ only in which skeleton classes are generated and how seed cells are truncated. At LC each colour ordering is seeded by its own Born amplitude, and skeletons below the leading N_c degree are never generated.

### 1.2 What was adopted from where

Two of three judges chose Design 2 as the base. The physics judge chose Design 1 but required Design 2's generic colour seeds, which makes the three verdicts compatible.

| Element | Source | Why |
|---|---|---|
| Per-family Born-seeded suffix-trie executor (schedule **B**); per-family cost-optimal cut; colour only in seeds; generic colour-flow seed classes; families as threads and as streaming units | Design 2 | Correct for any colour-flow-lowerable model without decomposition theorems. About 1–2 MB live state per lane. Families are independent (parallelism, streaming, targeted rescue). Runs on today's NinjaRS API. |
| Forward Recola-type schedule **F** on the same generator (MASTER_PLAN §2.14 schedule 1) | Design 1 | Baseline and oracle required by MASTER_PLAN §2.14. |
| Merged primitive-seeded schedule **M**, as the planned asymptotic production path | Design 1 | Best asymptotic class of the three schedules (§1.4). |
| Counterterm pseudo-order in the tree recursion with a unary node; witness table with power-counting completeness; direct-R2 model certifier; chiral and mixing δZ as CT sources | Design 1 | Counterterms produced by the recursion (user rule); physics completeness. |
| C-parity-gated orientation folding; primitive seed templates with relabelling; `loop-segment` kernel contract with a generic raise over the NinjaRS layout; a first measurement gate | Design 1 | Physics completeness and a measured start. |
| Per-(cut, lane) diagnostics | Design 3 | Correlations and rescue. |
| The process-level cut lattice as the end state of reduction-side sharing, fed by family tensors through NinjaRS's own jets | Design 3 | No private jet arithmetic in pyAmpliCol. |
| Per-family random-q oracle; "one contraction per channel" accounting for correlations | Design 3 | Oracles and observables. |
| Master-integral cache, families as lanes, true structural rank, no lifting, no on-the-fly reduction | recola-ninjars report, all designs | 85–109× redundant master requests at 7 legs [M]. |
| LC face rule, diagonal LC seeds, NLC = FC skeleton set with truncated cells | leading-colour report | Structural LC pruning before numerics. |
| Fixes for every flaw raised by the judges (§2.4) and by the two reviews of revision 1 (final section) | This document | |

### 1.3 Headline numbers

**Configuration.**

- Binary64, one thread, at least 16 SIMD lanes per batch (§5.4).
- The providers' model restriction `loop_sm-no_b_mass`. With a massless b, the closed-quark mass classes are {u,d,s,c,b} and {t} [V `required/*/madloop-export.generation.json`; `tools/loop_reference/gluon_scattering_qcd.json` has `mb = 0`].
- "Families" equals NinjaRS reductions per event per squared-order sector (stage 1), before the re-test.

**Every v2 time below is a target until gates G0 (kernels on real tries) and G1 (end to end) measure it.** G0 has measured the dressing term on real tries (§5.4, "Gate G0 on real tries"): the last column of Table 1a replaces the modelled dressing by the measured one and keeps the modelled NinjaRS stage and outer factor.

**Table 1a. v2 structure and targets.**

| Process | Colour | Families | Skeletons | B-trie nodes | Dressing work, complex MAC/pt (structured–dense kernels) | NinjaRS stage, cached masters | **Target s/pt (with re-test)** | **G0 s/pt: finite only / with re-test** |
|---|---|---|---|---|---|---|---|---|
| gg→gggg | FC | 1,100 | 9,180 | 7,619 | 1.3–2.4×10⁸ | 44–51 ms | **0.17–0.47** | **0.23–0.25 / 0.26–0.30** |
| gg→gggg | LC | 550 | 3,060 | 2,723 | 4.7–7.9×10⁷ | 22–26 ms | **0.07–0.17** | **0.070–0.078 / 0.088–0.106** |
| gg→ggggg | FC | 9,414 | 122,040 | 82,738 | 2.6–4.6×10⁹ | 0.85–1.0 s | **3.4–9.0** | 4.4–4.8 / 5.1–5.9 [I] |
| gg→ggggg | LC | 4,707 | 40,680 | 29,155 | 0.9–1.5×10⁹ | 0.43–0.50 s | **1.4–3.4** | 1.4–1.8 / 1.7–2.4 [I] |
| uu~→ggggg | FC | 5,258 | 91,440 | 56,861 | 0.86–1.6×10⁹ | 0.22–0.29 s | **1.0–3.0** | 1.2–1.5 / 1.4–1.9 [I] |
| uu~→ggggg | LC | 2,712 | 13,560 | 15,526 | 1.6–2.6×10⁸ | 92–124 ms | **0.28–0.66** | 0.28–0.36 / 0.35–0.49 [I] |
| uu~→e⁻e⁺dd~g | FC | 242 | 2,484 | 1,375 | 4.9–8.7×10⁶ | 1.8–3.3 ms | **0.007–0.020** | **0.0064–0.0083 / 0.0078–0.0118** |
| uu~→e⁻e⁺dd~g | LC | 114 | 144 | 419 | 0.9–1.4×10⁶ | 0.5–0.9 ms | **0.0016–0.0041** | 0.0014–0.0023 / 0.0018–0.0033 [I] |
| uu~→e⁻e⁺ggg | FC | 463 | 3,990 | 2,774 | 2.1–3.6×10⁷ | 9–13 ms | **0.032–0.082** | **0.035–0.040 / 0.042–0.054** |
| uu~→e⁻e⁺ggg | LC | 164 | 306 | 666 | 2.6–4.1×10⁶ | 1.5–2.1 ms | **0.0045–0.011** | 0.0046–0.0059 / 0.0058–0.0082 [I] |
| gg→tt̄gg | FC | 1,086 | 6,888 | 5,659 | 4.9–8.5×10⁷ | 19–27 ms | **0.070–0.19** | **0.078–0.090 / 0.093–0.119** |
| gg→tt̄gg | LC | 375 | 1,224 | 1,702 | 1.3–2.2×10⁷ | 5–7 ms | **0.019–0.048** | 0.021–0.026 / 0.025–0.034 [I] |
| gg→tt̄e⁻e⁺g (QCD) | FC | 579 | 3,990 | 2,840 | 2.4–4.1×10⁷ | 10–14 ms | **0.035–0.092** | **0.038–0.044 / 0.046–0.060** |
| gg→tt̄e⁻e⁺g (QCD) | LC | 164 | 306 | 666 | 3.6–5.9×10⁶ | 1.5–2.1 ms | **0.005–0.013** | 0.0057–0.0074 / 0.0068–0.0096 [I] |
| gg→tt̄e⁻e⁺g (EW) | FC | ≈1,450 [I] | gate E0 | gate E0 | ≈2–3.5× QCD [I] | ≈25–50 ms [I] | **0.07–0.32 [I]** | gate E0 |

G0 column: bold rows are measured programs (production code generation, 16 lanes; §5.4, "Gate G0 on real tries"): the measured dressing plus the modelled NinjaRS stage (×1 finite only, ×1.75–2.0 with the re-test), times the outer factor 1.05–1.10. Rows marked [I] scale the measured dressing of their class (gg→gggg for the all-gluon rows, uu~→e⁻e⁺ggg for uu~→ggggg, otherwise the row's own FC program) by the Appendix B structured work; their upper end uses the measured class with the most time per unit of that work.

**Table 1b. Providers on this host (s/pt, one pinned core) and the speed gate.** "Production" is each provider's fastest default output: finite part, with I-operator poles (OpenLoops) or no poles (Recola). "Protocol" is the required-references campaign's mode, which computes all actual poles (MASTER_PLAN §5.8 item 6).

| Process | OpenLoops production | Recola production | MadLoop, production build | OpenLoops protocol: campaign / same core | Recola protocol: campaign / same core | Gate = 1.3 × best production | Best production ÷ v2 target |
|---|---|---|---|---|---|---|---|
| gg→gggg | 2.92–2.96 | ≈19 [I] (not re-run: 2,035 s generation) | re-timing in progress (52–53 with a non-production build) | 9.5–10.1 / 8.6–8.7 | 57.2–57.6 / — | 3.8 | 6.2–17× |
| uu~→e⁻e⁺dd~g | 0.053–0.055 | **0.035** | 0.143 | 0.23–0.47 / 0.160 | 0.20–0.21 / 0.185 | **0.045** | **1.7–5.1×** |
| uu~→e⁻e⁺ggg | 0.348–0.353 | **0.246–0.259** | 1.41 | 1.16–1.24 / 1.04–1.06 | 1.13–1.24 / 1.02 | **0.32** | **3.0–7.7×** |
| gg→tt̄gg | **0.487–0.492** | 2.02–2.05 | 5.22–5.29 | 1.52–1.65 / 1.43–1.44 | 6.98–7.11 / 6.29–6.48 | 0.63 | 2.6–7.0× |
| gg→tt̄e⁻e⁺g (QCD) | **0.425–0.433** | 0.59–0.67 | 3.03–3.04 | 1.39–1.51 / 1.25–1.28 | 2.27–2.32 / 2.28–2.31 | 0.55 | 4.6–12× |
| gg→tt̄e⁻e⁺g (EW) | 0.96–1.12 | **0.95–1.02** | 6.90–7.09 | 3.07–3.57 / 2.75–3.27 | 5.45–5.52 / 5.33–5.57 | 1.24 | 3.0–14× [I] |
| uu~→ggggg | no public library | ≈33–61 [I] from the protocol run | export in progress | — | 180–184 (after 10,220 s of generation) / — | — | ≥ 11× [I] |
| gg→ggggg | no public library | protocol run still going after 4.5 h | still in diagram generation after 5 h | — | — | — | NJet: 171 s for two evaluations, scaling test included, 2012 Xeon [L] |

How the columns were obtained:

- **Counts** are exact enumerations by the generic skeleton enumerator [M `revision/mb0_all.log`; `lcs/lc_structure_counts.py`].
- **Dressing work** is schedule B on the same tries with two sets of kernel constants:
  - structured (lower): fused 200·X, split vertex 300;
  - dense (upper): fused 320·X, split vertex 1,300.
  
  The two are 1.7–1.8× apart [M `revision/b_dense.py`, `mb0_all.py`]. Revision 1 mixed schedule B in one model with schedule M in another; that is withdrawn.
- **NinjaRS stage** is the calibrated per-family model of the recola-ninjars report (constants `rn2/analyse.py`) applied to the family (N, R) histogram [M `revision/ninjars_stage.py`]. It is 5–30 % above the mass-aware model, which skips scaleless cuts (§11.4).
- **One recipe for every row:**

  t = [D × (1.20 … 1.35) / ((1.8 … 1.0)×10⁹ complex MAC/s) + R × (1.75 … 2.0)] × (1.05 … 1.10)

  - D: dressing work. Its factor covers quartic segments and ghost loops; ghost loops are ≈10–20 % of gluon-loop work.
  - Throughput: the Rust probe measured 1.3–1.8 G complex MAC/s at 16 lanes (§5.4). The lower bound allows 45 % for dispatch and gathers.
  - R: NinjaRS stage. Its factor is the default re-test of every family (§10.2): relabelling with cached masters costs 0.75 R; rescaling costs 1.0 R.
  - The outer factor covers trees, seeds, counterterms and checks.
  
  [M `revision/estimate_table.log`]
- **Massive b** (the `loop_sm` default), same recipe [M `revision/estimate_massive_b.log`]:
  - gg→gggg FC 0.25–0.64 s;
  - gg→ggggg FC 4.8–12.4 s;
  - uu~→ggggg FC 1.1–3.2 s;
  - 1,650 / 14,121 / 5,809 families respectively.
- **gg→tt̄e⁻e⁺g (EW)** is estimated, not counted [I]:
  - families scaled from QCD by MadLoop's EW/QCD loop-group ratio on this host (2,958/1,191 = 2.5) [V `required/gg_ttxemepg_{ew,qcd}/madloop-export/SubProcesses/P0_gg_ttxemepg/loop_matrix.f`, `NLOOPGROUPS`];
  - time scaled by the providers' measured EW/QCD ratios (1.4–3.2), and by one extra propagator wherever the lepton line runs through the loop (heptagons).
  
  Gate E0 (§14, WP3.2) replaces this with generator counts before any EW timing work.
- **Provider production columns** were measured for this revision on idle pinned cores. Points, binaries and settings are the campaign's except:
  - OpenLoops: `truepoles 0`, which keeps the Collier cache on, returns I-operator poles and uses the default stability system [V `OpenLoops-2.1.5/lib_src/openloops/src/init_ui.F90:455-463`];
  - Recola: `compute_ir_poles 0` with the Collier cache on. The steady state is quoted; the first timed point pays the cache set-up.
  
  Accuracy of the production runs [V `revision/ol_modes_*.json`, `revision/rcl_modes_*.json`, driver `revision/recola_driver_modes.f90`]:
  - OpenLoops finite parts equal the protocol runs bitwise.
  - Recola's cached results equal the uncached protocol runs to ≤ 1×10⁻¹² on these points. The campaign ran Recola uncached at 7 legs because of a COLLIER 1.2.9/1.2.10 cache defect at N = 7 (`tools/loop_reference/references.py:43`); the linked COLLIER is 1.2.8.
- **Recola gg→gggg and uu~→ggggg production values** are inferred [I]:
  - the protocol computes three amplitudes per point (`recola2-2.3.0/src/api/process_computation_rcl.f90:152-190` [V]), so divide by three;
  - for uu~→ggggg, also apply the cached/uncached ratio measured at 7 legs (0.55–0.86).
- **MadLoop** column: exports rebuilt with MG5's production flags (`GLOBAL_FLAG -O`), stability status 216 [V `required/*/madloop-production-result.json`, rewritten by the references session during this revision]. The gg→gggg production-flag timing is still running. Its earlier 52–53 s came from an unoptimised driver [V `required/gg_gggg/pre-production-flags/madloop-production-control-result.json`].
- **Speed gate** (MASTER_PLAN §5.8): t_v2 < t_MadLoop and t_v2 ≤ 1.30 × t_Recola, 1.30 × t_OpenLoops.
  - This design targets the stricter production baselines.
  - §5.8 item 6 requires v2's own timing to include all actual poles. v2 gets them in one pass: NinjaRS returns the full Laurent expansion, and the CT columns carry their ε powers.
  - The providers need extra evaluations for actual poles: 2.85–3.03× (OpenLoops) and 3.1–5.9× (Recola, steady state) on these processes.
  - The choice of baseline for formal acceptance is decision 7 (§1.6).

**Assumptions behind the FC targets.**

1. Certified colour compression for the gluon-rich rows (DDM adjoint ring, fundamental trace, q̄q ring; WP2.4 schedules these certificates first). With generic seeds, gg→ggggg FC needs a further 0.7–2.4×10⁹ complex MAC, i.e. +0.4–2.4 s (§7.2).
2. C-parity folding of closed quark loops with QCD attachments, which is certified. Closed loops with Z attachments keep both orientations. Without any folding, FC dressing of the gluon processes would be ≈1.75× (§7.2).
3. At least 16 lanes per batch. At 8 lanes the probe measured half the throughput.

**Ratios** (providers' configuration, same recipe).

- **LC/FC per point:**
  - 0.37–0.42 for the all-gluon processes (0.27–0.30 with a massive b);
  - 0.22–0.28 for uu~→ggggg;
  - 0.13–0.23 for the V+jets processes;
  - 0.26–0.27 for gg→tt̄gg;
  - 0.14–0.15 for gg→tt̄e⁻e⁺g (QCD).
  
  LC removes about two thirds of the gluon-process dressing but only half of the families, so the NinjaRS share grows at LC.
- **NLC/FC:** 0.9–1.0.
- **Parity halving** (optional, §8.3) is valid only for gg→gggg, gg→ggggg, uu~→ggggg and gg→tt̄gg: vector-like couplings, real masses, zero widths. It gives ×1.6–2 on dressing, i.e. ×1.25–1.6 per point [I].

### 1.4 Asymptotic complexity

Definitions:

- n external legs, worst case all gluons.
- H = 2ⁿ helicity rows.
- F_n ≈ ½(n−1)!·(1/ln 2)ⁿ families per mass class: 12, 77, 550, 4,707 and 47,348 for n = 4…8 [M].
- S_n = ½(n−1)!(2ⁿ−2n−1) gluon-loop skeletons per class: 21, 252, 3,060, 40,680 and 602,280 for n = 4…8 [M].
- Both counts exclude the certified-scaleless rings of §3.3.

| Stage | Full colour | Leading colour |
|---|---|---|
| Tree blocks, generation and per event | Θ(n·n!) colour-ordered currents; Θ(n!·2ⁿ) current-helicity rows per event; below 1 % of runtime at n = 7 [I] | Same class, restricted to contiguous segments of leading words |
| Loop-program generation | Θ(S_n) trie leaves and Θ(n·F_n) family-recursion states; linear in the instructions executed | Gluons: S_n per class without closed quark loops. One quark line: Θ((n−2)!·2ⁿ) |
| Seed tables, generation | Θ(#classes × #Born basis) exact overlaps. With adjoint primitives this is Θ((n−1)!·(n−2)!): 360 × 120 per class at n = 7 | Diagonal: Θ((n−1)!) |
| Seeds per event | Θ((n−1)!·(n−2)!·2ⁿ) direct; symmetric-group FFT where certified; 1–3 % of runtime for n ≤ 8 with compression [I] | Θ((n−1)!·2ⁿ) |
| **Dressing, schedule B** (default for n ≤ 7 until G1) | Asymptotically Θ((n−1)!·4ⁿ): the seed-injection term, ≈½(n−1)!·2ⁿ leaves × up to 2ⁿ rows. For n ≤ 8 that term is 2–10 % of the work; the trie dressing dominates, with model growth ×14, ×17, ×19, ×22 for n = 4→5→6→7→8 [M `otf/scaling8.log`] | Same class. Gluons: FC/(1 + number of closed-quark mass classes). One quark line: ≈ FC/(n−1) |
| **Dressing, schedule M** (planned production path, WP2.8) | Θ(n!·2ⁿ·n⁴): seeds per colour ordering at the cut ((n−1)!/2 orderings × 2ⁿ rows), prefixes shared across families. Same-model growth ×15, ×16, ×18 for n = 4→7, against B ×14, ×17, ×20 [M `otf/schedule_cost.log`] | Same class; valid generically at LC |
| Dressing, schedule F (baseline) | Θ(n!·(2/ln 2)ⁿ·n⁴); ≈29× M at n = 7 in the Design 1 model [M `cft/schedules_gluon_n8.log`] | — |
| NinjaRS | Θ(F_n) reductions (×2 with the re-test); Θ(F_n·n⁸) work (≈ n⁴ cuts × C(R+4,4) coefficients per family); Θ(4ⁿ) distinct masters after caching | ×½ (gluons; massless b), ×½ (one quark line) |
| Counterterm trees | Θ(#CT vertex kinds × tree cost), plus a Born-Gram contraction Θ(((n−2)!)²·2ⁿ) | Diagonal |

**Reference classes** (per event, exact helicity sums, full colour):

- **Per-diagram, per-helicity numerators** (MadLoop): Θ(D_n·2ⁿ·n⁴). D_n grows super-exponentially: gg→gggg has 26,975 loop diagrams and uu~→ggggg 110,475 [V `required/*/madloop-export.generation.json`].
- **OpenLoops 2:** Born-seeded per diagram, so a Θ(D_n·2ⁿ) seed term plus dressing over more objects than v2. Its measured cost grows ×27–35 per added gluon (gg→tt̄ + N g) [L 1907.13071 `OL2.tex` tab:perfnew].
- **Van Hameren's tensor recursion:** O(n⁴2ⁿ) per primitive and helicity [L 0905.1005 `introduction.tex:56-99`, `results.tex:142`], hence Θ((n−1)!·4ⁿ·n⁴) with explicit helicity sums.
- **Unitarity codes with exact helicity sums** (NGluon, NJet): Θ((n−1)!·2ⁿ·n^{6…9}) [L-s 1011.2900, 1209.0100].

**Where B and M stand.**

- B removes the explicit helicity factor from van Hameren's class and replaces diagrams by BG blocks. It is therefore at least as good as the best open-loop and tensor recursions.
- B's Θ((n−1)!·4ⁿ) seed term is asymptotically worse than the unitarity class Θ((n−1)!·2ⁿ·poly(n)). The crossover, 2ⁿ ≈ n^{6…9} times the ratio of constants, lies near n ≈ 30–50, far beyond the required set.
- M, at Θ(n!·2ⁿ·n⁴) = Θ((n−1)!·2ⁿ·n⁵), matches or beats the unitarity class.
- M is therefore the planned asymptotic production path (WP2.8), not an option. At n ≤ 7 the two schedules are within 10 % in the same model, so G1 decides by measured operation counts and kernel throughput (§2.2).

### 1.5 Why this beats a diagrammatic engine

1. **Against per-diagram, per-helicity numerators (MadLoop).** The MadLoop production builds take 5.2 s for gg→tt̄gg and 1.41 s for uu~→e⁻e⁺ggg. The v2 targets are 0.07–0.19 s and 0.032–0.082 s. The difference comes from three sources:
   - helicities summed while dressing: an object carries rows only for its undressed legs, not 2ⁿ;
   - trees computed once per point, not once per diagram (the inherited engine duplicates 54 % of loop-node and 95 % of CT-node evaluations, assessment W4 [V-s]);
   - families instead of diagrams.
2. **Against OpenLoops 2** (diagrams with on-the-fly helicity sums and merging). Object counts alone give little: at gg→gggg, v2 has 9,180 skeletons and 7,619 trie nodes against 26,975 loop diagrams (MadLoop's count). The expected 2.6–17× (Table 1b) has to come from four structural differences. G1 measures the total.
   - **Cut choice and sharing:** per-family optimal cut plus the suffix trie is 2.1× (n = 6) and 2.7× (n = 7) less dressing work than OpenLoops' pinch-invariant cut on the same skeletons [M `otf/scaling8.log`].
   - **BG blocks:** each position sums all trees on its leg subset once per point, whereas OpenLoops attaches cached subtrees per diagram [L 1907.13071 `OL2.tex:556-583`]. The ratio of objects grows with n, because blocks absorb the factorially many tree topologies.
   - **Colour:** exact seed-table lookups, with no per-diagram colour factors and no colour seed per diagram. OpenLoops 2 loses parent–child recycling because the seed carries the colour of the whole diagram [L 1710.11452 l.1405-1411].
   - **Reduction:** integrand reduction of Born-weighted family numerators with cached masters, plus the per-family re-test.
3. **Colour is a seed lookup.** There are no per-diagram tensor overlaps. Assessment W3 counts ≈5×10⁷ Born and ≈10⁹ loop overlaps at 7 gluons in the inherited engine.
4. **Generation is linear in the instructions.** It takes minutes at 7 legs. For comparison on this host [V `required/STATUS.md`, `*.generation.json`, `*-result.json`]:
   - gg→gggg: MadLoop needed 285 s of graph generation, 8,354 s of Fortran export and 20.6 s of compilation. Recola needed 2,035 s of generation.
   - uu~→ggggg: MadLoop graph generation took 5,793 s, followed by a colour step that MG5 itself estimates at 14 h. Recola needed 10,220 s of generation, then 180–184 s per point.
   - gg→ggggg: MadLoop is still in diagram generation after 5 h, and Recola is still running after 4.5 h.
   - The inherited engine was projected at more than 10⁶ s for a 7-leg process (assessment §3).

Revision 1's "20–50× below OpenLoops, 100–250× below MadLoop" compared against protocol-mode providers and is withdrawn. The defensible statement is the production comparison in Table 1b: targets 1.7–17× below the best production provider, measured at G1.

### 1.6 Decisions that need the user

1. **Stage-2 reduction sharing** (§11.3). By default it is the MASTER_PLAN-§2.13-compliant process-level lattice or per-member division. The cut-restricted lift is 1.5–2× faster on the NinjaRS stage. It multiplies pinched numerators by cut-reduced denominators, so it needs an explicit amendment of §2.13. Either option is pursued only if NinjaRS is measured above 30 % of the total.
2. **"Hybrid-reduction" rescue tier**: re-reducing binary64 coefficients in DoubleFloat. It contradicts the strict reading of §2.18 ("recompute failing lanes from original inputs"). It is off by default and, if accepted, must carry labelled statuses (§10.5).
3. **Resolved colour components.** The anchor becomes a Born colour-basis element, so the `kind` is renamed (contracts §3.12).
4. **Loop-level certificates** for primitive seeds, schedule M in FC/NLC, Kleiss–Kuijf (KK) blocks and parity halving. They are re-proved for loops and never inherited from tree certificates (MASTER_PLAN §2.16).
5. **Schedule adoption.**
   - F (schedule 1) ships first as baseline and oracle.
   - B becomes the default after exact equality with F and gate G1.
   - M is a planned production path (WP2.8). It is adopted per process class (LC; certified FC classes) when G1's measured operation counts × G0 throughput show at least a 15 % gain, and by default for n ≥ 8.
6. **Single-level rescue policies.** Today they cannot rescue (`rust/loop-reduction/src/verification.rs:346-351` [V]). Either require at least two levels or define single-level verification explicitly.
7. **Speed-gate baseline.** MASTER_PLAN §5.8 does not say in which mode the providers run. This design recommends accepting against the providers' production mode (OpenLoops `truepoles 0`; Recola `compute_ir_poles 0` with the Collier cache on) and also reporting the actual-pole protocol columns. Production mode is stricter by ≈3–6× (Table 1b).
8. **Default accuracy re-test** of every family of every point (§10.2). It costs +0.75–1.0 NinjaRS pass, i.e. +12–25 % per point. It may be relaxed only after WP2.3 measures its false-negative rate.

### 1.7 Main risks

These are detailed in §15.

1. **Real kernel throughput and per-node dispatch** on real tries. The probe measured 1.3–1.8 G complex MAC/s at 16 lanes on synthetic shapes and 0.4–0.9 G at 8 lanes. The thinnest production margin is uu~→e⁻e⁺dd~g: 1.7× at the upper end, against Recola's 35 ms. Gates G0 and G1 measure this before any 7-leg work.
2. **Correctness of generic colour seeds** for multi-quark, U(1) and electroweak classes, and of the Born–loop convention seam: phases, i-factors, symmetry factors, fermion signs, traversal orientation.
3. **NinjaRS stability and masters:** heptagon stability in binary64, the cost of rescue, and the known complex-mass master defects.
4. **Counterterm completeness** for general and electroweak models, and the availability of a formal-N_c continuation for LC/NLC.
5. **Electroweak family proliferation** for gg→tt̄e⁻e⁺g (EW), checked by gate E0.
6. **Observables whose cost grows with the number of seed columns** (directed correlations, resolved helicities), until the transposed reduction exists.

---

## 2. Architecture

### 2.1 Layers and dataflow

```text
                         GENERATION (once per process)
 UFO loop model --> prepare_loop_request_model (kept) --> order plan (kept, MadLoop joint selection)
        |                         |                                   |
        v                         v                                   v
 loop kernel catalogue   tree program (tree engine, loop mode)   CT rules, field rules (kept)
 (segment, propagator-   all-subset blocks, Born export,          |
  numerator, closure,    CT=1 currents (wave 2)                   |
  CT kernels; transverse/      |                                  |
  full YM variants)            |                                  |
        \_____________________ v _________________________________/
                 family recursion over loop prefixes (Recola keys, canonical cut)
                               |
             per family: contents, seed classes, LC/NLC pruning, rank, fermion signs,
             certified-scaleless drop, cost-optimal frame, suffix trie (B), prefix graph (F),
             merged prefix states (M, where certified)
                               |
             seed tables K[c,b] (exact Laurent), degrees P per sector,
             CT admission and completeness, observables  ---> plan v4 (PACBIN)

                         EXECUTION (per batch of points, native Rust, >= 16 lanes)
 exact inputs -> invariant table s[mask] -> tree runtime (blocks J, Born A0, CT amplitudes)
   -> seeds U0 = A0^dagger K -> per bucket (N,R): dress families (threads x SIMD lanes; B or M)
   -> TensorNumerator lanes -> NinjaRS (cached masters) -> Laurent per family
   -> relabelled re-reduction per family (accuracy estimate)
   -> 2Re(i/16pi^2 sum) + UV_local + R2 + fields -> checks -> rescue -> CompleteLoopResult
```

### 2.2 One generator, three schedules

The generator emits one family and skeleton description. The three schedules are three key projections of it, which makes them mutual oracles.

| | F: forward (MASTER_PLAN §2.14 schedule 1) | **B: backward suffix trie (schedule 2; default)** | **M: merged, primitive-seeded (schedule 2 variant; planned production path)** |
|---|---|---|---|
| State key | cut species, dressed positions with contents, open species, orders | family, cut species, depth k, undressed contents and gap species | routing class, dressed position sets and species, undressed colour word, open species, orders |
| Rows | Helicities of attached legs | Helicities of undressed legs | Helicities of undressed legs |
| Born weighting | At closure | At the leaves (seeds) | At the cut (one seed per colour ordering) |
| What is shared | Prefixes across all families (OpenLoops-1 parent–child recycling) | All prefixes with the same family and remainder | Prefixes across families and compositions |
| Colour requirement | None | None: one seed per skeleton class | Seed may depend only on the ordering: generic at LC; FC/NLC only for certified primitive classes |
| Gluon loop, 7 gluons, otf model | — | 8.98–9.23×10⁸ complex MAC [M `otf/schedule_cost.log`, `scaling8.log`] | 8.45×10⁸ (0.92–0.94× B) [M same log] |
| Gluon loop, 7 gluons, Design 1 model | F/M = 29 [M `cft/schedules_gluon_n8.log`] | — | 2.69×10⁹ (dense kernels, rank = step) |
| Memory per lane, gg→ggggg FC | High (helicity-explicit prefixes) | 1–2 MB live objects plus streamed buckets | ≈85 MB: 65 MB family accumulators + ≈20 MB live states, chunked by the last two letters of the cut word [M `cft/memory_gluon.py 7`] |
| Role | Baseline, oracle, helicity-resolved and many-observable outputs with a transposed reduction | Default until G1; generic fallback for uncertified classes | Production where certified and measured better (G1); default for n ≥ 8 |

**Why B is the default until G1, and why M is planned rather than optional.**

- Only one cost model compares B and M directly, and at n = 7 it puts M at 0.92–0.94× B. The choice is therefore made by measurement. The Python reference executor (WP1.5) runs F, B and M with exact operation counters per kernel form. G1 multiplies these counts by G0's measured kernel throughput.
- M needs certified colour decompositions in FC/NLC: DDM for gluon and ghost loops, fundamental trace for closed quark loops, the q̄q ring for one quark line. Generic multi-quark classes, e.g. uu~→e⁻e⁺dd~g, stay on B.
- M keeps family accumulators live: ≈85 MB per lane at gg→ggggg FC, ≈1.4 GB at 16 lanes. That is affordable on this host and is not a cap. B has 1–2 MB per lane and per-family locality (threads, streaming into NinjaRS buckets, per-family rescue).
- M has the better asymptotic class (§1.4) and is the recursion-native structure the user asked for: prefixes shared across families, seeds per ordering. It is therefore a planned work package (WP2.8) with a measured adoption rule, not a wave-4 option.

### 2.3 Where the recursion shares work

| Level | What is summed | Mechanism in v2 | Effect of NinjaRS's fixed-denominator contract |
|---|---|---|---|
| L1 trees | All trees on a leg subset | BG blocks, shared by Born, loop attachments and CT trees | None |
| L2 loop, same denominators | All skeletons with the same family and undressed remainder (B); also across families with equal prefixes (M) | B-trie merging; M prefix sharing (F: prefix sharing) | None. This is exactly the per-family numerator |
| L3 across pinch-related families | Pinched sub-topologies of a parent | Not in the recursion (would need lifting, §11.3); optional at reduction time (stage 2, gated) | Blocked for polynomials; possible on cuts |
| L4 masters | Scalar integrals common to many families | Per-event scalar-master cache | None. It shares scalar masters only; Collier's cache also shares tensor sub-reductions (§11.3) |

### 2.4 Resolution of the judges' findings

The two adversarial reviews of revision 1 are resolved in the final section. This table covers the earlier judges' findings and is updated where revision 2 changed the answer.

| # | Finding (raised by) | Resolution | § |
|---|---|---|---|
| 1 | Massive quartic tadpoles dropped (Design 1; physics, scalability) | N = 1 families are retained, as in `loops/graphs.py:117, 328-331` [V]. Only VEV, massless and external-leg tadpoles are excluded; scaleless families are dropped only with a structural certificate. | 3.3, 4.3 |
| 2 | Merging across compositions rests on unproven primitive decompositions (Design 1) | B needs no decomposition. Primitive classes compress seed tables and enable M, with loop-level certificates. | 3.4, 7.2 |
| 3 | KK pre-merge partly inferred (Design 1) | Optional: only pure-gluon, CT-free, source-free cubic segments, behind a numerical identity gate. | 16 (WP4.2) |
| 4 | A rescue tier promotes binary64 values (Design 1) | Rescue recomputes failing lanes from original inputs; the per-family tier recomputes that family in DoubleFloat. "Hybrid" exists only as a labelled opt-in after sign-off. | 10 |
| 5 | 85 MB per lane (Design 1) | B: per-family tries and bucket streaming. M: 85 MB per lane (65 MB accumulators + 20 MB chunked states), accepted where M wins. | 2.2, 5.3 |
| 6 | NinjaRS asymptotics understated (Design 1) | Corrected to Θ(F_n·n⁸), plus the re-test pass. | 1.4, 11 |
| 7 | Stage-2 lift conflicts with MASTER_PLAN §2.13 (Designs 1, 2) | The default stage 2 is the compliant lattice / per-member division; the lift needs sign-off. | 11.3 |
| 8 | CT completeness criterion wrong (Design 2) | Power counting R ≥ 2N − 4 per divergent witness, plus certified zeros; bounds derived from the model. | 6.3 |
| 9 | CT-site symmetry unspecified (Design 2) | Production CTs come from the tree recursion, whose vertex rules fix the symmetry. The site closure is kept as the oracle, over unordered partitions with symmetrised vertices. | 6.1, 6.6 |
| 10 | CT sites not produced by the recursion (Design 2) | CT pseudo-order inside the tree recursion. | 6.1 |
| 11 | Closed-fermion orientation folding without a C-symmetry caveat (Design 2) | Folding requires a C-parity certificate of the attached vertices; otherwise both orientations are kept. | 7.2 |
| 12 | External fields scalar only (Designs 2, 3) | Chiral and mixing δZ become CT-tagged external sources. | 6.5 |
| 13 | Family numerator store omitted from the memory budget (Design 2) | Bucket streaming. | 5.3 |
| 14 | No prefix sharing across families (Design 2) | F has it; M is the planned production path with it. | 2.2 |
| 15 | O(n!·2ⁿ·n²) asserted but not derived (Design 2) | Classes derived in §1.4, including the 4ⁿ seed term and its consequence. | 1.4 |
| 16 | Correlations ×42 at 7 legs (Design 2) | Observable axis, or schedule 3 (F plus transposed reduction), chosen by a generation-time cost model. | 7.7, 11.5 |
| 17 | Schedule 1 not built (Design 2) | F is built first. | 2.2, WP1.5 |
| 18 | Production depends on a non-existent NinjaRS API; private jets (Design 3) | Not the base. The lattice is only a gated stage 2. It needs a public jets API: `fast/tensor_jets.rs` is a private module (`ninjars:fast/mod.rs:46` [V]); owner request 7. | 11.3, 11.5 |
| 19 | No stored coefficients for cheap rescue (Design 3) | B stores per-family polynomials; they also feed the default re-test. | 10 |
| 20 | X1: LC/NLC counterterms need a formal N_c continuation | LC/NLC are refused without a source-owned continuation; FC always works. | 7.4 |
| 21 | X2: the Ward test is trivial with symmetric sources | Asymmetric validation mode: physical Born bra, loop leg ε → p. | 8.4 |
| 22 | X4: signs and symmetry factors only by oracle | Explicit rules: automorphism weight, fermion sign (§3.3), closed-loop sign, i-factors. FeynKit oracle; comparison with Recola's 1-/2-point factor convention. | 3.3, 14 |
| 23 | Transverse Yang–Mills shortcut | Kept where exact (blocks with transverse sources). Full kernels for loop segments, CT = 1 currents, Ward and non-transverse sources. | 3.2 |
| 24 | Bubble ½ and double counting of two-block rings in the models | Automorphism rule, checked by FeynKit. | 3.3 |
| 25 | Cost models disagree by 2–4× | One model for both ends (structured vs dense kernel constants); kernel probe; gates G0 and G1 before any timing claim. | 1.3, 5.4 |
| 26 | Invasive tree-engine CT changes (engineering) | Staged: site closure in the slice, recursion in wave 2, CT colour in a new module. | 16 |

---

## 3. Data model

### 3.1 Identities

MASTER_PLAN §2.11 asks to keep graph, current and reduction-family identity distinct.

| Identity | Object | Key | Owner |
|---|---|---|---|
| Tree current | Block | Full `CurrentIndex` (`src/pyamplicol/generation/dag_types.py:121-176` [V]), including the full `ColorState` (accuracy, `sector_id`, `line_groups`, `basis_key`; `dag_types.py:66-90` [V]). Rust: `CurrentCoreKey` (`rust/crates/rusticol-core/src/recurrence/layout.rs:497-510` [V-s]). CT order added in wave 2 | Tree engine |
| Loop prefix (generation) | Family-recursion state | (cut species, ordered position leg sets, propagator species and masses, open species, orders): Recola's (ho, hm) [V-s recola2 `skeleton_rcl.f90:876-905`] | `loops/v2/families.py` |
| **Family** (reduction identity) | Denominator set | Canonical cyclic sequence of (position leg mask, mass id), minimised over the 2N rotations and reflections | `loops/v2/families.py` |
| Skeleton | Colour-ordered ring of blocks | (family, cut species, content per position, gap species) | Implicit (trie leaves / M closures) |
| Open-loop state | B node / F prefix / M state | §3.5 | Executor |
| Seed class | Colour vector | Exact colour-flow vector (generic) or primitive id (certified) | `loops/v2/colour.py` |
| Channel | NinjaRS lane | (family, point, sector, observable) | Reduction adapter |
| Graph | Feynman diagram | FeynKit | **Oracle only** |

MASTER_PLAN §2.11 also says "use FeynKit for canonicalization, routing, signs and symmetry factors". The 2026-10-20 user clarification (`MASTER_PLAN.md:1496-1501` [V]) supersedes this for production: families, insertions, colour and signs come from the recursion. FeynKit `IntegralFamily` and graph overall factors stay the small-process oracle (§14).

### 3.2 Blocks: the tree engine in loop-request mode

Blocks are the tree engine's colour-ordered off-shell currents. A loop request changes the following:

1. **Every needed subset; loop-aware reachability.** The Python builder prunes in three layers, each derived from the Born's closure splits (`generation/dag_compiler.py:131-133` flags, `:341-378` use [V]):
   - closure-side mask pruning;
   - colour-order mask pruning;
   - species reachability (the "useful states" pass).
   
   All three would drop currents a loop vertex needs. Loop mode replaces them with one loop-aware reachability test. A current (leg set B, field, colour state) is kept iff either:
   - some catalogue loop vertex accepts its field and (B, field) occurs as the content of an admissible position of some family-recursion state (leg-set, order and colour-class caps applied); or
   - it is needed by the Born or CT = 1 roots.
   
   The test is evaluated from the family recursion's position table, not from a list of skeletons.
2. **Kernels: transverse where exact, full where needed.**
   - The transverse massless Yang–Mills replacement is chosen per vertex term at model compile time (`models/compiler_entry.py:726-733` [V]), applied at `models/compiler_kernels.py:326-335` and defined at `:398-425` [V].
   - It drops only terms proportional to the input currents' self-contractions p_i·J_i. Berends–Giele currents built from transverse on-shell sources are conserved, so the replacement is exact for them. Plain blocks and the Born therefore keep it.
   - Full kernels are required wherever an input current is not conserved: every loop segment (the open-loop object is not transverse), CT = 1 currents (wave 2), and blocks fed by Ward (ε → p) or non-transverse spin sources.
   - WP1.1 adds the full-kernel catalogue flag. The loop-mode builder selects the kernel variant per current class.
   - The replacement exists only for massless adjoint currents, so electroweak blocks use full kernels anyway.
3. **Block identity and exports.** Blocks are keyed by the full `CurrentIndex`: leg mask, field, coupling orders and the full `ColorState`. For multi-quark FC, colour word and field alone do not identify a block. Exports:
   - J[block id][h_B]: block currents per helicity row;
   - A0[b][h]: the Born per colour-basis element and helicity, which already exist internally in `recurrence/direct_plan.rs:256-263` and `direct_runtime.rs:1167` [V-s];
   - Born roots bucketed by exact Born order (needed for mixed orders).
4. **Graded colour pieces.** The U(1)-subtraction auxiliaries and the LC Fierz singlet are separate, graded currents instead of numeric 1/3 factors (`models/compiler_color_flow.py:417-419`, `generation/dag_color.py:539, 562` [V]). This is needed for LC/NLC; FC at N_c = 3 is unaffected.
5. **Counterterms (wave 2).** A CT pseudo-order capped at 1, and a unary node for two-point insertions (§6.1). `InteractionNode` is binary today (`dag_types.py:313-325` [V]).
6. **Electroweak (wave 3).** Massive vectors in Feynman gauge with explicit Goldstones. Today a massive vector is unitary (`models/external_evaluation.py:186-192`) and the Feynman branch divides by p² only (`:309-318`) [V]. Ghosts appear only as loop lines, never as blocks.
7. **Genericity.** The hard-coded gluon gate `abs(pdg) == 21` (`generation/service.py:8467` [V]) is replaced by a model predicate.

### 3.3 Families

- **Positions.** A family is a cyclic sequence of N ≥ 1 positions. Position i has a leg set B_i; the leg sets are non-empty, pairwise disjoint, and their union is all legs. A quartic position carries two currents with B_i = B′ ∪ B″. Higher-arity loop vertices of a general model carry more currents.
- **Propagators.** Propagator i follows position i and has a mass id m_i: the exact model mass expression including its width. Two ids are equal only if the expressions are identical, so species of equal mass share families: gluon, ghost and massless quarks; at ξ = 1 a vector and its Goldstone.
- **Frame.** The cut sits before position 1 and the loop momentum q flows through propagator N.
  - The shifts are V_i = Σ_{j≤i} p(B_j), stored as integer leg masks, with D_i = (q+V_i)² − m_i² (−μ² in d dimensions). This is the contract convention (`loops/kinematics.py:72-101` [V-s contracts K4]).
  - Each schedule publishes its own frame; oracles map frames with exact affine transformations (`loops/v2/poly_affine.py`).
- **Canonical key.** The lexicographic minimum of ((mask, mass id)_i) over the 2N rotations and reflections. It equals Recola's tensor-integral identity and was validated as equal sets on 5 processes [M recola-ninjars §5.1].
- **Selection rule.** These are the semantics of `GraphSelectionConvention` and `_physical_rejection` (`loops/graphs.py:109-121, 312-335` [V]), applied structurally. A ring is never generated if it is one of:
  - an external wavefunction correction: N ≤ 2 positions with exactly two attachments, one of them a single external leg (LSZ; `graphs.py:321-327`);
  - a VEV tadpole: N = 1 with at most one attachment;
  - a colour zero: a ring with exactly one coloured attachment.
  
  Every other ring is generated. That includes two-position rings with three or more attachments, i.e. a single-leg position whose partner position is quartic. Internal self-energies and bridges are retained and live inside blocks, so nothing is restricted to 1PI (MASTER_PLAN §2.10). Massive N = 1 tadpoles with two or more attachments are retained.
- **Certified scaleless families.** A generated family is dropped only with a structural certificate:
  - the condition is that all its propagator mass ids are zero and every entry of its S-matrix is structurally zero after on-shell bindings;
  - examples: N = 1 massless tadpoles, and N = 2 massless bubbles whose single invariant is p_i² = 0 of an on-shell massless leg;
  - the certificate uses mass ids and the on-shell binding table, never numerics.
  
  For massless QCD this removes exactly the gluon bubbles {k², (k−p_i)²} with a quartic partner position. FeynKit gives 52 gg→gg families in the control model: 4 mass classes × 12, plus 4 such bubbles, each from 3 quartic-vertex diagrams [V `revision/fam_gg_classes.py`; the control model has massive c, b, t]. The QCD counts of this document (Appendix B) are therefore unchanged. With massive loop lines and quartic vertices (electroweak, BSM) these families are non-zero and kept. Oracles O2 and O7 compare modulo certified-zero families.
- **Rank.** R_f is the max-plus sum over contents of Δ = vertex q-degree + propagator-numerator q-degree. It is ≤ N in renormalisable Feynman gauge. R_f > N + 2 fails explicitly; this is NinjaRS's domain (`ninjars:fast/engine.rs:620-632` [V]), not an arbitrary cap.
- **Leaf weight** of every skeleton, folded into its seed (judges X4, 24):

  w_s = (automorphism weight) × (fermion sign ε_s) × (product of the kernels' i-factors).

  - The automorphism weight is 1/|Stab(s)| under the ring's dihedral group restricted to maps that preserve contents and species. It is ½ for a two-position bubble with identical self-conjugate propagators, and 1 for oriented fermion or ghost loops and for N ≥ 3 rings of distinct blocks.
  - Closed ghost loops carry −1 inside ε_s, like closed fermion loops.
  - These rules are checked against FeynKit overall factors (`overall_factor_expression`) on the oracle set, and against Recola's separate factor for 1- and 2-point loops (`amplitude_rcl.f90:1471-1474` [V-s physics judge]).
- **Fermion sign ε_s.** Every skeleton carries the sign of its Wick contraction relative to the Born. It is computed with the tree engine's own conventions, so that blocks, Born and loop agree:
  1. Signs inside blocks are already in the block currents:
     - colour-singlet fermions: Koszul signs of binary merges, with source ranks = source slots (`generation/fermion_ordering.py:58-79` support sign, `:81-110` orientation sign, `:112-130` ranks, `:163` `merge_sign` [V]);
     - coloured open lines: the colour-flow construction, which owns their ordering (`fermion_ordering.py:1-16` docstring [V]).
  2. **Ring sign.** Cut the loop before position 1 and treat the skeleton as the (n+2)-point tree whose two extra legs are the cut pair. The field entering position 1 gets source rank n+1 and the field leaving position N gets rank n+2, both after all external legs. At every ring step, apply the tree engine's `merge_sign`: Koszul sign plus the orientation sign of the two-fermion kernel, merging the open-loop object's support (which includes leg n+1) with the block's support, exactly as for a tree merge.
  3. **Coloured lines.** For coloured fermion lines, multiply by the parity of the external-fermion pairing per (role, canonical id) class, as the colour plan does for identical fermions (`color/plan_replay.py:671-701` [V]), with the cut pair ranked n+1, n+2.
  4. **Sewing.** Closing the cut pair gives −1 if the cut line belongs to a closed fermion or ghost loop and +1 otherwise. No further permutation arises, because the cut pair is adjacent and ranked last.
  5. ε_s ∈ {±1} multiplies w_s. Both traversal orientations of a fermion line through the loop carry their own ε_s.
  
  `loops/v2/conventions.py` calls the tree engine's functions; there is no private re-implementation. Oracles (O13): uu~→uu~ and uu~→uu~g (identical-flavour pairings with relative signs), and ee→μμ and e⁺e⁻→e⁺e⁻ at NLO EW (lepton lines inside the loop), on top of the uudd, uu~→dd~g and uu~→dd~ss~ controls.
- **Majorana fermions** are rejected by the tree engine (`fermion_ordering.py:100-104`: "requires non-self-conjugate model orientations" [V]). v2 inherits this limit and states it in its scope check.
- **Per-family data:**
  - on-shell bindings (S-matrix entries replaced by exact external masses, `loops/kinematics.py:150-194` [V-s]);
  - active sectors;
  - bucket (N, R_f);
  - master keys: canonical (cut lines, invariant masks, mass ids) per cut with at most 4 lines;
  - the re-test relabelling k_f (§10.2).

### 3.4 Skeleton contents and seed classes

**Content of a position:** (vertex template, attached block current or currents with their full colour states, outgoing gap species, traversal orientation). Fermion and ghost lines are traversed in either direction with the matching oriented kernels.

**Certified planar mode.** When a colour class is certified (DDM for adjoint rings, or LC), the cubic and quartic contributions of a position with word π fold into one segment:

X(π) = V₃·J(π) + Σ_{π=π′π″} V₄(J(π′), J(π″))  (Design 1).

The selection rule (§3.3) still applies to the cubic and quartic parts separately: for a two-position ring with a single-leg partner, only the quartic part is generated. Generic mode keeps quartic splits, and blocks on the non-planar side of the colour double line, as separate contents with their own ordered-vertex kernels.

**Seed class of a skeleton.** This is the colour vector obtained by closing the cut ports of the skeleton's colour flow. It is the colour factor of the (n+2)-point tree with the cut pair closed [L-s Kälin 1712.03539 §4.1].

- **Generic** (any model whose vertices lower to colour flow): κ_s = Σ_t c_{s,t}|t⟩. The structures t are single traces, products of traces, open strings, and graded U(1) pieces. Blocks on the inner side of a gluon double line are merged into the planar skeleton only where the tree engine's reflection certificate applies (`models/external_catalog.py` `adjoint_current_reflection_phase` [V-s tree report]); otherwise they are separate contents.
- **Certified compressions.** Each is re-certified at loop level per model and order, never inherited from tree certificates (MASTER_PLAN §2.16):
  - adjoint ring Tr(F^σ) for gluon and ghost loops [L hep-ph/9910563 §3 `NewLoopColor`; verified at `fcolor.tex:845-852` by the physics judge];
  - fundamental trace Tr(T^σ) for closed quark loops (automatic in colour flow);
  - q̄q ring with left/right primitives for one quark line [L hep-ph/9910563 `LoopColorqqNew`].

  Compression changes only the seed-table size, never the dressing. It is also what makes M valid in FC/NLC.

### 3.5 Open-loop objects

All payloads use raw monomial coefficients in the NinjaRS layout. `ninjars:fast/tensor_layout.rs` is a public module with `ncoeff`, `block_offset` and `homogeneous_index` [V]. Rank r has C(r+4,4) coefficients, there are no hidden factorials (MASTER_PLAN §2.12), and the raise tables are built once from `homogeneous_index`.

- **B node** v = (f, s_cut, k, σ_{k+1..N}), where σ_j = (content of position j, gap species s_j).
  - Payload O_v[o][h][α][β][m]:
    - o: observable column;
    - h: helicity row over the undressed legs;
    - α: cut-line index (4 for vector or Dirac lines, 1 for scalars and ghosts);
    - β: current-end index;
    - m: monomials of degree ≤ r_k.
  - Leaves (k = 0) are skeletons; their seed object is w_s·U0[c(s)][o][h]·δ_{αβ}.
- **F state** p = (s_cut, (content_1…content_k), species, orders). Payload O_p[h_att][α][β][m]; observables enter at closure.
- **M state** = (routing class ρ, (B_1,s_1)…(B_k,s_k), undressed word u, open species, orders). Payload as in B; seeds per (σ, ρ, rotation) at the cut. Closures scatter into per-family accumulators. Live states are chunked by the last letters of the cut word, which loses no merge (`cft/memory_gluon.py` [M]).

### 3.6 Axes

- **Helicities.** The Born catalogue (`loops/physical_states.py:23-29, 87-89` [V-s]) with a selection mask (§8.1). B and M carry rows of undressed legs; F carries rows of attached legs.
- **Observables.** Columns o = (Born-order component a) × (inclusive | directed history and direction | spin-source class | resolved component). The closure of a skeleton of loop order b lands in sector a + b.
- **Orders.** Exact per-current and per-content order vectors. The joint MadLoop-style eligibility of `loops/orders.py` is kept.
- **ε and μ².** Production numerators are four-dimensional and μ²-free, which `TensorNumerator` covers (MASTER_PLAN §2.12; `physical_process.py:495-503` [V-s]). The monomial label (a0..a3, k_μ², e) is retained in the plan schema for the direct-R2 bonus (§6.9).

### 3.7 Kernel contracts

These are new prepared-kernel kinds in `models/prepared_catalog.py` (`PreparedKernelSpec` at `:269-310` [V-s]).

| Kind | Content | Inputs |
|---|---|---|
| `loop-segment` (split) | Vertex tensor Y_g, Z_g (and higher q-degree terms if a model has them) for (vertex template, loop-in leg, loop-out leg, attached legs, orientation), including the outgoing loop-propagator numerator; independent of the block current | P_in, block momenta, couplings, masses |
| `loop-segment` (fused) | X(J) = Y(J) + Z(J)·q for one block current | Same, plus J |
| `loop-propagator-numerator` | Numerator only, per species and orientation (denominators belong to NinjaRS) | Momentum |
| `loop-closure` | Trace over the cut index; no sign (the closed fermion- and ghost-loop −1 enters only through ε_s, §3.3) | — |
| `ct-vertex` (wave 2) | CT vertex as a tree vertex with (kind, ε) slots | As tree vertices |
| `ct-site` (slice, oracle) | CT vertex applied to m block currents at the root | Currents, momenta |

The catalogue carries a **full-kernel flag** (WP1.1): every tree kernel exists in its compile-time variant (transverse where `compiler_entry.py` chooses it) and in its full Feynman-rule variant, and §3.2 item 2 selects between them.

All kinds are compiled from the UFO Lorentz structures by Symbolica coefficient extraction in q. That works because the kernel component expressions are plain Symbolica expressions. The probe found q-degree ≤ 1 for G G G and ghost–gluon kernels and 0 for quark–gluon kernels (`probe_kernel.py`) [V-s tree report §6.2].

- **Evaluators.** Binary64 uses SymJIT; DoubleFloat and arbitrary precision use exact `ExpressionEvaluator<Complex<Rational>>` states.
- **Polynomial multiply.** O′ = O·Y + raise(O·Z_μ, q_μ) is one generic Rust routine over T ∈ {f64, DoubleFloat, Float}, with no per-rank kernels and no rank cap. Lanes are the innermost (structure-of-arrays) index, and the batch carries at least 16 lanes (§5.4).

### 3.8 Plan v4 and batch state

```text
LoopProgramV4 {
  process:   legs (StateLeg), helicity catalogue + selection mask, normalization divisor,
             order layout, ModelBinding::Source, flavour members
  tree:      reference to the tree recurrence plan (existing PACBIN member), block export table
             (block id -> full CurrentIndex incl. ColorState, rows, kernel variant),
             Born export (basis b x helicity), CT=1 export (slots)
  kernels:   references into the prepared model bundle (SymJIT applications, exact states),
             transverse and full variants
  families:  [key, positions (leg masks), mass ids, frame, rank, bucket, on-shell bindings,
              active sectors, master keys, re-test relabelling]; certified-zero log
  schedules: B: per family {nodes (depth, species, row legs), edges (child|leaf, content, group),
                kernel groups, roots per cut species, leaf weights incl. fermion signs}
             F: prefix graph {states, edges, closures (family, class, weight)}
             M: where adopted {chunked states, edges, seed slots (ordering, routing class),
                closures (family, weight)}
  colour:    Born basis, seed classes (generic terms | primitive ids + certificates),
             exact Laurent tables K[c,b], degrees P per sector, accuracy, FFT certificates
  counterterms: CT=1 export map, kinds/slots, Laurent coefficient columns, field rules, witness log
  observables:  sectors and pairs, directed histories, spin classes, resolved axes
  reduction:    buckets (N,R) -> families, master canonicalization rules, engine policy
  stability:    re-test mode, error budget, trigger and escalation policy defaults
}

LoopBatchStateV4<T> {
  exact inputs, invariant table s[mask], model snapshot (SourceOwner), tree arena,
  seeds U0[class][obs][h], per-thread trie workspaces (B) / chunk workspaces (M),
  bucket TensorNumerators, Laurent results and re-test results per (family, lane),
  CT amplitudes, statuses, rescue queues
}
```

---

## 4. Generation algorithm

### 4.1 Pipeline

| Step | Work | Reused code |
|---|---|---|
| G0 | Model, restrictions, scope, CT and field rules, scalar ownership | `loops/compiler.py:91-155` `prepare_loop_request_model`; `loops/counterterms.py:161` `prepare_counterterm_model`; `loops/wavefunction.py` `prepare_external_field_rules` [V/V-s] |
| G1 | Loop kernel catalogue (transverse and full variants), cached per (model, gauge, restriction) | `models/compiler_kernels.py:223` oriented kernels, `prepared_catalog.py`, `prepared_compile.py` (SymJIT), Symbolica, Idenso |
| G2 | Tree program in loop mode (loop-aware reachability) | `GenericDAGCompiler` (slice); Rust recurrence builder (wave 3) |
| G3 | Families by loop-prefix recursion; certified-scaleless drop | New `loops/v2/families.py` |
| G4 | Contents, seed classes, LC/NLC pruning, rank, leaf weights and fermion signs, frame, B trie, F graph, M states | New `loops/v2/skeletons.py`, `schedules.py`, `colour.py`, `conventions.py` |
| G5 | Seed tables and degrees | `color/connections.py:492`, `color/adjoint_kernel.py:36-62` (cycle counting), `color/plan_build.py:17` (Born basis, never with `max_sectors`) [V-s] |
| G6 | CT program, admission, completeness | Tree recursion (wave 2) or site closure (slice); new `loops/v2/counterterms.py` |
| G7 | Lowering to plan v4 | `generation/loop_artifact.py`, `rust/crates/rusticol-core/src/loop_runtime.rs` (schema v4) |

### 4.2 Pseudo-code

```text
generate_loop_program_v2(request) -> LoopProgramV4:
    prep = prepare_loop_request_model(...)                 # kept
    plan = loop_order_plan(request, prep)                   # kept: joint selection, pairs, sectors
    kcat = loop_kernel_catalogue(prep)                      # cached per model; transverse + full variants
    pos  = position_table(prep, plan, kcat)                 # (leg set, field, colour state) a loop vertex accepts
    tree = tree_program(process, prep, plan, keep=loop_reachable(pos), kernel_variant=per_class,
                        ct_insertion_order=1,               # wave 2; the slice uses CT site closure
                        export=("blocks", "born", "ct1"))

    # (1) families: recursion over loop prefixes built from leg subsets; no diagrams
    families = {}
    work = [Prefix.root(s) for s in loop_species(prep, plan)]       # incl. ghosts, Goldstones
    while work:
        p = work.pop()
        for B in next_position_sets(p, tree):     # disjoint from p; F-frame rule: leg 1 in position 1,
                                                  # direction by the smaller neighbouring minimal leg
            for v in loop_vertices(p.open_species, tree.fields_over(B), kcat, plan):  # cubic, quartic split, ...
                q = p.extend(B, v)                                   # appends (B, v.out species, mass)
                if q.covers_all_legs():
                    if v.out == p.cut_species and convention_admits(q):            # section 3.3 rule
                        families.setdefault(canonical_key(q), Family(q)).add_routing(q)
                elif q.can_still_close(tree, kcat, plan):
                    work.append(q)
    families = {k: f for k, f in families.items() if not certified_scaleless(f)}   # logged

    # (2) per family
    for f in families.values():
        f.contents = position_contents(f, tree, kcat)     # full colour states available in the tree program
        f.classes  = seed_classes(f, accuracy)            # generic kappa or certified primitive id;
                                                          # LC/NLC: drop classes below the kept degrees
        if f.is_empty(): drop(f); continue
        f.rank     = max_plus_rank(f)                      # > N+2 -> explicit failure
        f.weights  = leaf_weights(f)                       # automorphism, fermion sign, i-factors
        f.frame    = argmin over 2N (cut, direction) of cost_model(f)          # schedule B
        f.trie     = suffix_trie(f)                        # nodes keyed by undressed contents
        f.retest   = relabelling(f)                        # k_f != 0, section 10.2
        f.bucket, f.masters = (f.N, f.rank), master_keys(f)
    fwd = prefix_graph(families)                            # schedule F, canonical frames
    mrg = merged_states(families, certified_classes)        # schedule M where adopted (G1 policy)

    # (3) colour, counterterms, observables
    K   = seed_tables(all_classes, tree.born_basis, plan.sectors)   # exact Laurent in N_c
    P   = structural_degrees(K, ct_cells, field_cells)               # one P per virtual sector
    K   = project(K, P, accuracy)                                    # LC: P; NLC: P..P-2; FC: all
    CT  = counterterm_program(tree, families, prep, plan, P)         # section 6
    OBS = observable_program(request, K, tree)
    return lower_to_plan_v4(tree, kcat, families, fwd, mrg, K, P, CT, OBS)
```

The suffix trie of a family is built backwards from the last dressed position. The nodes at depth k are the distinct (contents, species) of positions k+1..N, restricted to contents admissible with every vertex, species transfer, order cap and (at LC/NLC) colour class. Its leaves are the skeletons, which appear only as step-1 edges.

### 4.3 Local rules applied during generation

All of these are structural; none depends on a process or a multiplicity.

- **Vertex existence and species transfer.** Every position must be a model vertex (s_in, block field(s), s_out). Species routing is consistent around the ring. Closed fermion and ghost loops carry orientation.
- **Orders and eligibility.** Coupling-order caps and full-set eligibility come from `loops/orders.py:37-66` (`LoopEligibility`) [V-s]. Every circulating species must be eligible for a selected correction, and a selected order must occur at a loop vertex (MASTER_PLAN §2.2).
- **Conventions.** As in §3.3: the selection rule, then the certified-scaleless drop.
- **Colour zeros.** A single coloured attachment is excluded. At LC/NLC, classes whose exact maximal degree is below the kept degrees are dropped (§7.5, §7.6).
- **Massless closed-quark multiplicity.** A single class with an integer multiplicity is used only when every loop vertex coupling and mass of the merged flavours is structurally identical (MASTER_PLAN §2.15). The multiplicity has N_c degree 0.

### 4.4 Generation cost

The asymptotic class is Θ(n·n!) for trees plus Θ(S_n) for the loop program plus the seed tables (§1.4). Per process [I, from the measured tree-DAG builds in the tree-engine report §2.2 and the Design 2 model timings]:

| Process (FC) | Tree program (Python builder) | Loop program (Python / Rust) | Seed tables | Total | Peak memory | Plan size |
|---|---|---|---|---|---|---|
| gg→gggg | 10–60 s (FC Born DAG alone: 8.9 s [V-s]) | 2 s / <1 s | seconds | < 2 min | < 2 GB | ≈5 MB |
| gg→ggggg | 3–10 min (LC Born DAG: 81 s [V-s]) | 33 s / < 5 s | < 1 min with primitives | 5–20 min Python, < 5 min Rust | 2–8 GB | 20–80 MB |
| uu~→ggggg | 1–3 min (Born DAG: 17–20 s [V-s]) | 20 s / < 5 s | < 1 min | 2–10 min | < 4 GB | 10–40 MB |

For comparison:

- the inherited engine's artifact is 394 MB at 6 legs (assessment §2.1);
- Recola needs 2,035 s (gg→gggg) and 10,220 s (uu~→ggggg) to generate on this host;
- MadLoop needs 285 + 8,354 + 20.6 s for gg→gggg (§1.5).

---

## 5. Per-event execution

### 5.1 Pseudo-code

```text
evaluate(batch, prog) -> CompleteLoopResult:            # lanes = points (SoA, >= 16); T in {f64, DD, Float}
    inv       = invariant_table(batch.exact_inputs)      # s[mask] for every leg subset; exact -> T once
    J, A0, C1 = tree_runtime(prog.tree, batch)           # blocks, Born per (b,h), CT=1 per (b|t, h, slot)
    U0        = seeds(prog.colour, A0, prog.observables) # U0[c][o][h] = sum_b K[c,b] conj(A0_b(h)) / norm
                                                         # LC: diagonal; FC/NLC: direct or certified FFT
    for bucket in prog.buckets:                          # (N, R); chunked to bound memory
        num = TensorNumerator(R, lanes = |chunk| x |points| x |columns|)
        parallel for f in chunk:                         # threads over families, SIMD over lanes
            acc = dress_family(prog.trie[f], U0, J, inv)     # or: M chunks scatter into acc
            scatter(acc, num, f)                         # SoA family accumulator -> lane-major NinjaRS
        Omega[bucket]  = reduce(bucket, num, inv)         # fast engine (f64) | generic (DD, Float);
                                                          # caching master provider
        Omega2[bucket] = reduce(bucket, relabel(num, prog.retest), relabel(inv))   # re-test, cached masters
    ct  = contract_counterterms(C1, A0, prog.counterterms)   # UV_local, R2, fields per sector, pole
    res = assemble(born(A0), bare = 2 Re(i/(16 pi^2) sum_f Omega_f), ct)   # 6 components, 19 totals
    return verify_and_rescue(res, Omega, Omega2, batch)  # section 10: error budget per family

dress_family(trie, U0, J, inv):
    acc = 0
    for root in trie.roots:                              # one per cut species
        acc += Tr_alpha(dress(root))                     # closure: cut-propagator numerator in segment N
    return acc

dress(v):                                               # post-order DFS: O(depth) live objects
    T = {g: 0 for g in v.kernel_groups}                  # T[g][o][h][alpha][beta][nu][m]
    for e in v.incoming:                                 # one edge per content of position k
        O = w(e.leaf) * seed(U0, e.leaf) if v.depth == 1 else dress(e.child)
        for h in rows(v), hB in rows(e.block):           # helicity sum of the attached block
            T[e.group][h] += O[hB (+) h] (x) J[e.block][hB]      # split form, or fused (chosen per node)
        free(O)
    return sum over g of apply_segment(Y_g(inv), Z_g(inv), T[g])  # rank r -> r + Delta
```

Schedule F (baseline and oracle) runs over the prefix graph in increasing depth, with helicity-explicit rows of attached legs. At each closure it adds Σ_h U0[class][o][h]·w·Tr O[h] into acc[f][o]. Schedule M runs its chunked state graph in increasing depth with seeds injected at the cut, and scatters closures into the family accumulators. All three use the same kernels and the same seeds.

### 5.2 Kernel organisation

A segment is linear in the attached current J. Design 2's split form exploits that:

1. First accumulate T = Σ_{edges, h_B} O ⊗ J. This costs d_α·d_β·d_ν = 64 complex MACs per (input row, coefficient) for vector lines.
2. Then apply each vertex tensor once per group of edges sharing a vertex template. This costs ≈300 per (output row, coefficient) when structured (Symbolica common-subexpression elimination) and ≈1,300 when dense.

The fused form (X(J) per block row) costs 200 (structured) to 320 (dense) per (input row, coefficient). The generator picks the cheaper form per node.

Two properties hold for every node:

- Execution is the same instruction stream for every lane (lockstep). Unstable lanes are compacted, never branched.
- Every step writes in the NinjaRS monomial layout, so closure scatters coefficients without reordering. Only a transpose from structure-of-arrays to NinjaRS's lane-major storage is needed (`ninjars:fast/tensor.rs:30-90` [V]).

### 5.3 Memory and threads

- **Live open-loop objects (B).** O(depth) per family in the post-order traversal, about 1–2 MB per lane at 7 legs [M Design 2]. Families run on independent threads.
- **M.** ≈85 MB per lane at gg→ggggg FC (65 MB accumulators + ≈20 MB live states chunked by the last two letters of the cut word) [M `cft/memory_gluon.py 7`]. That is ≈1.4 GB at 16 lanes, which is acceptable on this host and is not a cap. Streaming accumulators into buckets is possible when chunk order is family-aligned; WP2.8 measures it.
- **Tree arena.** All words of all subsets with their helicity rows: about 3–15 MB per lane at 7 gluons [I], i.e. 0.05–0.24 GB at 16 lanes.
- **Seeds.** #classes × #rows: about 3 MB per lane for gg→ggggg FC with primitives [I].
- **Family numerators.** These are never all live. Buckets are dressed and reduced in chunks, which bounds the peak at chunk × lanes × C(R+4,4). Example: 512 heptagon families × 16 lanes × 330 coefficients ≈ 43 MB.
- **Batch size** is a workload consequence, not a cap. At least 16 lanes are needed for kernel throughput (§5.4).

### 5.4 Cost model, kernel probe and calibration

**One model for both ends.** Per node the cost is rows_out × columns × C(r+4,4) × min(c_f·X, 64·X + c_V), where X = Σ_edges (block rows):

- structured constants (c_f, c_V) = (200, 300) give the lower end;
- dense constants (320, 1,300) give the upper end, on the same tries, ranks and rows. The dense work is 1.7–1.8× the structured work for every process [M `revision/b_dense.log`, `mb0_all.log`].

Revision 1 instead took its upper end from the Design 1 model's merged schedule (MK). That is a different schedule with a different rank assignment, and the comparison is withdrawn. For reference, the two models disagree by 3.2× on schedule M at n = 7 (2.69×10⁹ vs 8.45×10⁸).

**Rank profile at 7 legs:** the dressing cost splits 10, 19, 19, 19, 18, 11 and 3 % over input ranks 0–6 [M `otf/rank_profile.py`]. Neither the early (row-heavy) nor the late (rank-heavy) steps dominate.

**Kernel probe** [M `revision/kbench/` (std-only Rust, `-C target-cpu=native`, f64 structure-of-arrays lanes), log `kbench_all.log`]. The probe implements the design's generic, rank-agnostic kernels with the generic raise table on synthetic node shapes that span a 7-leg trie, from early to late steps. Work is counted per lane as in the cost model. Units are G complex MAC/s per core (AMD EPYC 9754 at 3.1 GHz, idle pinned core).

| Node shape (rows_out, X, input rank, coefficients) | split, dense vertex: 8 / 16 lanes | fused: 8 / 16 lanes | fused, register-blocked: 8 / 16 lanes |
|---|---|---|---|
| (64, 4, 0, 1) | 0.76 / 1.54 | 0.84 / 1.44 | 0.79 / 1.69 |
| (32, 12, 1, 5) | 0.89 / 1.53 | 0.90 / 1.40 | 0.77 / 1.60 |
| (16, 16, 2, 15) | 0.65 / 1.46 | 0.91 / 1.43 | 0.79 / 1.80 |
| (8, 32, 3, 35) | 0.49 / 1.54 | 0.90 / 1.41 | 0.79 / 1.82 |
| (4, 48, 4, 70) | 0.46 / 1.64 | 0.87 / 1.35 | 0.78 / 1.80 |
| (2, 64, 5, 126) | 0.40 / 1.43 | 0.85 / 1.52 | 0.79 / 1.65 |
| (1, 64, 6, 210) | 0.39 / 1.50 | 0.84 / 1.33 | 0.77 / 1.70 |

What the probe shows:

- At 16 lanes, every form reaches 1.3–1.8 G complex MAC/s, flat in rank and rows. At 8 lanes it reaches only 0.4–0.9 G, so the batch default is at least 16 lanes.
- The theoretical peak is ≈6.2 G complex MAC/s per core (8 binary64 FMAs per cycle at 3.1 GHz, 4 FMAs per complex MAC). The probe reaches 21–29 % of it, which leaves room for specialised kernels (WP4.6).
- The probe does not measure per-node dispatch, gathers of row maps from real tries, or cache misses on large objects. The recipe's lower bound of 1.0 G allows 45 % for these.

**Gate G0** (WP0.3, wave 0/1, before WP2.1 commits the kernel form):

- run the same kernels on real trie shapes exported by WP1.3 for gg→gggg and uu~→e⁻e⁺dd~g: split, fused and register-blocked forms; 8, 16 and 32 lanes; per-node dispatch overhead;
- recalibrate the recipe;
- G0 passes when the recalibrated targets of every required QCD process are within their gates with ×1.3 headroom. Otherwise kernel specialisation (WP4.6) moves into wave 2.

**Gate G1** (WP2.7, wave 2) measures:

- complex MAC/s per kernel form;
- per-node dispatch overhead;
- end-to-end timings for gg→gg, gg→ggg, uu~→ggg, uu~→dd~g, gg→gggg, **uu~→e⁻e⁺dd~g and uu~→e⁻e⁺ggg** (the V+jets processes are the speed-critical set);
- the providers re-measured in production and protocol modes under one protocol;
- the B-vs-M decision from WP1.5's exact operation counts × measured throughput.

Every estimate in this document is then recalibrated. The low-multiplicity check is gg→gg: the recipe gives 0.6–1.7 ms at mb = 0, against MadLoop's 2.07–2.59 ms on this host. The inherited engine takes 14.3 ms, of which numerators are 9.2 ms and NinjaRS with masters 0.16 ms (`docs/performance_reports/loop/data/measurements.json` [V]).

**2→6 stretch.** Eight gluons cost 2.0×10¹⁰ complex MAC for the gluon loop alone [M `otf/scaling8.log`], so tens of seconds per point in FC. M is the lever there, and is the default at n ≥ 8.

**Gate G0 on real tries (results, 2026-10-21; WP0.3, second half).**

*Method.*

- `tools/loop_v2_costmodel.py --export-programs DIR` writes the schedule-B program that the Python reference executor runs (`ReferenceExecutor.run_b` over the WP1.3 families and schedules and the WP1.1 kernel catalogue), in the configuration of `tools/loop_v2_operation_counts.py`: loop_sm with mb = 0, synthetic block chirality rows (as WP1.5), cost-optimal frames, inclusive seeds, LC by the all-gluon face rule. A program lists every dressing step with its edges (child step or leaf seed, content with its catalogue entries, helicity row links) and split vertex groups. Its per-form counts equal the reference executor's exactly (`tests/unit/test_loop_v2_costmodel.py`; for gg→gggg, uu~→e⁻e⁺dd~g and uu~→e⁻e⁺ggg also `operation_counts.json`).
- `benches/loop_v2_kernels.rs --program FILE` runs each program with the WP0.3 kernels as the executor will: each family trie depth-first, one output and one set of split accumulators per depth, fused factors X(J) built per (content, frame position) inside the timed run, leaf-seeded steps on a collapsed segment. Before any timing all five forms must give the same family numerators: the largest deviation is 1.4×10⁻¹⁵ of the largest component, over 6 programs × 3 lane counts × 3 code generations.
- Per program it times every form, a walk without kernels, the per-step fastest form (oracle) and a per-step form rule, best of 3 after a warm-up, at 8, 16 and 32 lanes, in three code generations (baseline x86-64 = production, x86-64-v3, native AVX-512), each pinned to an idle core of its own CCX. The host was otherwise loaded: the synthetic probe measured 1.30 G at 16 lanes in this run (1.42 G in the kernel-gate run), and repeated runs spread by about ±10 %. Reports: `workspace/tmp/g0-real-tries/run2/` (`bench-*.json`, `costmodel.log`).

*Real tries, production code generation, 16 lanes.* Counted work is the reference executor's per-step cheaper form (the unit of D); "÷ D" divides it by the Appendix B structured work. Times are dressing per point, fused-factor builds included.

| Program | Steps | Edges | Row links | Counted work, 10⁶ CMAC/pt (÷ D) | Rule, ms/pt | Oracle | Best single form | G CMAC/s (counted) | Walk share | AVX2 / AVX-512 rule, ms/pt |
|---|---|---|---|---|---|---|---|---|---|---|
| gg→gggg FC | 18,712 | 36,330 | 1,498,140 | 235.6 (1.76) | 171 | 172 | 222 | 1.37 | 3.0 % | 170 / 122 |
| gg→gggg LC | 8,024 | 15,570 | 642,060 | 60.3 (1.28) | 45.3 | 44.8 | 50.5 | 1.33 | 2.6 % | 44.1 / 32.7 |
| uu~→e⁻e⁺dd~g FC | 1,969 | 5,942 | 76,352 | 5.5 (1.12) | 4.23 | 4.22 | 6.38 | 1.29 | 4.4 % | 4.06 / 3.14 |
| uu~→e⁻e⁺ggg FC | 4,616 | 11,636 | 277,493 | 32.6 (1.55) | 23.8 | 23.7 | 35.1 | 1.37 | 3.8 % | 21.9 / 17.5 |
| gg→tt̄gg FC | 9,168 | 21,833 | 995,360 | 75.1 (1.52) | 54.6 | 55.2 | 68.8 | 1.37 | 3.2 % | 57.4 / 49.1 |
| gg→tt̄e⁻e⁺g FC | 4,220 | 10,433 | 481,888 | 35.6 (1.49) | 26.2 | 26.2 | 42.4 | 1.36 | 3.3 % | 27.4 / 18.9 |

*What G0 shows.*

1. **Throughput holds on real tries.** The kernels run 1.29–1.37 G counted CMAC/s per core in the production code generation (AVX2 1.30–1.48, AVX-512 1.53–1.93), the synthetic probe's rate on the same host state. The 45 % allowance for gathers and cache misses is not needed. The counted work is 1.12–1.76× the Appendix B D: ghost loops and both orientations of closed loops are separate skeletons (no C-parity folding yet, §7.2), and kernels carry the catalogue's entries; the measured times include all of it, so Table 1a's G0 column drops the quartic factor.
2. **Dispatch is not the limiter.** The walk without kernels (traversal, segment set-up, output and accumulator clears) costs 1.5–4.3 µs per step, 2.6–4.4 % of the dressing. Step models fitted on the in-situ step times give 0 per step and 4.5–7.8 ns per inner-loop iteration for the fused form, 0–3.7 µs per step for the split form's vertex groups, and 11.7–14.8 ns per iteration for the register-blocked form; kernel work is 97 % of the modelled time. These replace the probe's empty-call lower bounds.
3. **Lane count.** Geometric-mean time over the six programs relative to 16 lanes is 1.05 / 1.05 / 1.16 at 8 lanes and 1.02 / 0.97 / 1.01 at 32 lanes (baseline / AVX2 / AVX-512). **The executor's default is 16 lanes**: the smallest count within 5 % of the fastest in every code generation (`program_lanes`). 32 lanes gains nothing measurable and doubles the live objects and fused factors.
4. **Kernel-form selection rule.** Per step, take the form f of {fused-structured, split-structured, fused-blocked} with the least t_f = c_f + i_f·n_f + W_f·L/r_f. Here W_f is the step's complex MACs per lane in form f, n_f its inner-loop iterations (W_f without the factor of input coefficients) and L the lane count. There is one constant table per code generation and lane count (`selection_rules`, fitted on gg→gggg FC). At 16 lanes, production code generation, (c, i, r) is fused (0, 4.5 ns, 1.39 G/s), split (0, 0, 1.28 G/s) and blocked (0, 11.7 ns, 1.43 G/s); under AVX-512 the blocked form reaches 3.6 G/s.
   - Transferred to the other programs, the rule's step time is within 0.2–0.9 % of the per-step oracle (AVX2 0.5–2.5 %, AVX-512 2.3–4.4 %). Choosing the form with less counted work is within 0.2–12 %; the best single form is 7.5–70 % slower.
   - In practice, leaf-seeded steps take the split form and rank ≥ 1 steps the fused form (the blocked form at high rank under AVX-512).
   - This replaces the kernel-gate rule "split for X ≥ 4": on real tries the discriminant is the input rank, not X.
5. **Verdict.** Upper end of the finite-only target (no checks) against the gate, headroom per code generation (baseline / AVX2 / AVX-512):
   - gg→gggg 0.245 s against 3.8 s: ×15.5 / 15.6 / 19.9;
   - uu~→e⁻e⁺ggg 0.040 s against 0.32 s: ×7.9 / 8.4 / 9.6;
   - uu~→e⁻e⁺dd~g 8.3 ms against 45 ms: ×5.5 / 5.6 / 6.4;
   - gg→tt̄gg 0.090 s against 0.63 s: ×7.1 / 6.8 / 7.6;
   - gg→tt̄e⁻e⁺g (QCD) 0.044 s against 0.55 s: ×12.5 / 12.1 / 15.2.

   **G0 passes** at 16 lanes in the production code generation: the thinnest headroom is ×5.5 against the required ×1.3 (the probe recipe gave ×2.15). WP4.6 is not promoted and stays a wave-4 option; runtime AVX2/AVX-512 kernel selection is worth 1.0–1.4× on dressing.
6. **For WP2.1.**
   - Fused factors X(J) per (content, frame position) are the largest per-batch state: gg→gggg FC has 6,839 of them, 2.2 M lane vectors (0.55 GB at 16 lanes), each read by 5.3 edges on average, and building them costs 0.06–0.54 ms/pt (0.3–1.5 % of the dressing). At 7 legs (10–15× the contents) they should be built per family bucket or per edge, not all at once.
   - C-parity folding of closed loops and ghost-loop compression are the largest structural levers left for the gluon processes (counted work 1.5–1.76× Appendix B).
   - Steps of input rank 1–4 in the fused form carry 60–80 % of the time, leaf-seeded split steps 10–20 %.

*Not measured by G0* (recipe terms kept): the NinjaRS stage (§1.3 model), trees, seeds and counterterms (outer factor 1.05–1.10), real tree-block chirality rows (synthetic, as WP1.5), and LC for processes with quark lines (scaled from their FC class). G1 measures these end to end.

*Reproduction* (bench with `CARGO_TARGET_DIR` in a scratch tree; add `RUSTFLAGS="-C target-cpu=..."` for the other code generations):

```text
python tools/loop_v2_costmodel.py --export-programs programs/
cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_kernels -- \
    --program programs/gg_gggg-FC.json ... --json bench-baseline.json
python tools/loop_v2_costmodel.py bench-baseline.json bench-x86-64-v3.json bench-native.json
```

---

## 6. Counterterms, R2, UV and external fields

### 6.1 Production form: counterterms inside the tree recursion (wave 2)

**Rules are kept.** Every prepared `CountertermRule` (`loops/counterterms.py:52`, built by `prepare_counterterm_model` at `:161` [V]) carries:

- its kind (R2, UVmass, UVloop or UVtree);
- Laurent coefficients for ε powers −2..0;
- loop-species alternatives and loop group;
- an order already tagged with the internal order `PYAMPLICOL_CT_INSERTION` (`:47` [V]).

The rules are lowered into the tree model as ordinary vertices.

- **Lorentz.** The same normalisation and oriented-kernel path as tree vertices (`models/tensors.py:72`, `compiler_kernels.py:223` [V-s]), always in the full-kernel variant. Contact CT vertices of any arity use the existing contact decomposition.
- **Order.** The pseudo-order is capped at 1. `CurrentIndex.coupling_orders` then interns CT = 0 and CT = 1 currents in one recursion; CT = 1 roots are the CT amplitudes.
- **Slots.** Kind × ε power become coefficient columns on CT = 1 currents: at most 12, sparse. UV_local, R2 and the fields therefore stay separable, and the coefficients are runtime scalars from the `counterterm_scale` source view (contracts K1).
- **Two-point insertions on internal lines** use a new unary node, J′(S) = Π(S)·V₂(S)·J(S) (Rust: a finalization row). Insertions on external legs are excluded and replaced by field terms.
- **Colour.** CT colour structures are lowered with Idenso into colour-flow components in a new module, `models/ct_color_flow.py`.
  - Besides singlet, identity, T and f, this covers the d-symbol and δδ forms. Today the recurrence compiler accepts only singlet, identity, T and f (`compiler_color_flow.py:180-280` [V-s]).
  - δδ and d·d produce a new colour-state kind, open string ⊗ closed trace. The Rust state already has `Trace` components (`recurrence/color.rs:16` [V-s]).
  - Before any LC/NLC split, the fixed-SU(3) four-gluon R2 is replaced by the general-N_c HV asset: single trace × (r_CI V_I + r_CII V_II) + T_F δδ × (−C_A/(3N_c)) [L-s 2007.03713 eqs. 5.30–5.38].

Counterterm amplitudes then sit in the Born colour sectors (plus the few multi-trace CT states) and are contracted with the Born like the Born itself. This costs about #CT kinds × tree cost and needs no per-site colour seed. That is why the recursion beats the site closure in FC at 7 legs: site seeds would cost #sites × #Born × #h ≈ 3×10⁹ complex MAC per point [I].

### 6.2 Admission: the witness table

The witness table is built at generation from the **divergent witness families**: families whose structural rank satisfies R_f ≥ 2N − 4. The bound is derived from the generated families' ranks, not fixed. For renormalisable Feynman-gauge models, R_f ≤ N already implies N ≤ 4.

A witness signature is (attached currents' fields and leg masks, loop species set, loop orders). Its number of attachments is bounded by the largest CT-vertex arity k_max among the model's CT rules (4 for renormalisable models).

A CT vertex instance of rule r at a node of the tree recursion (incoming currents plus the outgoing complement) is admitted as follows. These are the semantics of `match_counterterms` (`counterterms.py:756-914` [V-s contracts §3.6]):

| Kind | Admission |
|---|---|
| R2, UVmass | A witness with the same signature exists, and its loop species belongs to the rule's alternatives |
| UVloop | A witness exists; the alternatives act only as forbidden-species filters |
| UVtree | No witness needed |
| Any kind on an external two-point site | Rejected (LSZ) |

For QCD, admission is almost field-level; leg masks enter only through LSZ, order caps and eligibility. A precomputed predicate is therefore consulted during the recursion, and no site is ever enumerated.

### 6.3 Completeness

1. **Coverage rule.** Every divergent witness family must be covered by an admitted rule for each eligible loop-species group, or by a certified zero. A divergent witness family has structural rank R ≥ 2N − 4, i.e. it is superficially UV divergent, which is the only case where UV poles and R2 occur in a renormalisable theory. Otherwise generation fails and names the uncovered signature.
   - This replaces the incorrect "every N ≤ 4 family" rule of Design 2.
   - It removes the silent-drop failure behind the sgluon bug (assessment R2).
   - For a model whose ranks make the divergent set unbounded (Δ ≥ 2 per propagator, i.e. effective theories), generation refuses unless the model declares its counterterm completeness.
2. **Certified zeros.**
   - *Cheap test:* the summed 1PI numerator, including both fermion orientations, vanishes exactly at random exact points (Furry-type zeros).
   - *Thorough test:* the **model certifier** built on the retained direct-R2 path (`loops/evanescent.py`, `evanescent_gauge.py`, `direct_kernels.py`). Once per model, it computes R2 and UV poles of every admitted 1PI function with k ≤ k_max per loop species, and compares them with the UFO rules, both non-zero values and zeros. This is also the structural replacement for the string whitelist (assessment R6).

### 6.4 Contraction and component split

- **Contraction.** CT = 1 amplitudes are contracted with the Born exactly like the Born square: A0† G A_CT per sector, plus rectangular overlaps for CT-only multi-trace states.
- **Kinds stay separate** through slots. This reproduces the kept component vector [full, bare, UV, R2, UV_local, UV_fields] (`rust/crates/rusticol-core/src/loop_execution.rs:188` `CompleteLoopResult` [V]) and the 19 totals, with full = bare + UV_local + R2 + fields and UV = UV_local + fields.
- **No extra loop factor.** CT couplings already contain the loop factor, so no i/(16π²) is applied (`physical_counterterms.py:848-849` [V-s]).
- **Allowed exact refactorings** (contracts §2.5): fold currents per (site class, tensor) and drop identically zero currents.

### 6.5 External fields

- **Scalar δZ:** for each external leg and correction, δZ/2 × the order-masked Born interference, keeping the semantics of `physical_counterterms.py:877-905` [V-s].
- **Chiral δZ^{L,R} and mixing** (δZ_{Zγ}, CKM): CT-tagged modified external sources in the tree recursion (CT = 1 sources). The field term is then the Born dual times that amplitude, which is generic for the electroweak members.
- **Tadpole CTs.** The VEV-subtracted scheme has no tadpole CTs; one-point rules raise at preparation, as today (`counterterms.py:259-260` [V-s]).

### 6.6 Slice form and permanent oracle: CT site closure

In wave 1, before §6.1 exists, CT contributions are computed as **sites**:

- A CT vertex V_r is applied at the root to m ∈ {2, …, k_max} CT = 0 block currents whose leg sets partition the legs (k_max from the model's CT rules).
- Partitions are unordered and assigned to the vertex's field multiset. The UFO vertex is fully symmetrised, so each assignment counts once, which is the BG rule.
- Internal two-point insertions are two-block sites with both blocks of at least two legs.
- Colour comes from exact closure of the vertex colour with the block words.

Every one-CT tree has exactly one CT vertex and can be rooted there, so the site sum is exact and complete. It stays as the oracle for §6.1 (physics-judge graft 7) and for small processes.

### 6.7 Counterterms at LC and NLC

The loop_sm CT colour origins and their projection [V-s leading-colour report §4.3; `workspace/reports/loop-general-nc-r2-input-audit.md`]:

| CT family | Colour origin | LC keeps | NLC adds |
|---|---|---|---|
| G_UVg (β₀) | 11 C_A | C_A part | — |
| Quark-loop UV (per flavour), heavy-quark decoupling logs | 4 T_F | — | All (P−1) |
| Quark field, mass and Yukawa CTs | C_F | N_c/2 part | −1/(2N_c) part (P−2) |
| R2 3-gluon and 2-gluon, gluon loop | explicit N_c | All | — |
| R2 GQQ, QQq | C_F | N_c/2 part | Rest |
| R2 with a closed fundamental loop | T_F | — | All (P−1) |
| Four-gluon R2 (general-N_c asset) | single trace × C_A; T_F N_f; δδ | C_A single-trace part, colour-ordered | T_F N_f (P−1); δδ (P−2) |
| ggV, ggS, ggVV, ggSS | δ^{ab}/T_F | — | All (P−1) |

LC/NLC need these named invariants, which a UFO with numeric SU(3) colour does not provide (`loops/model.py:1-7` [V]). LC/NLC are therefore refused unless the model supplies a source-owned continuation: `QCDR2Continuation` (`loops/qcd_r2_color.py:30` [V]) for QCD, a future EW or BSM declaration otherwise. FC always works (MASTER_PLAN §2.15; judge X1).

### 6.8 Gauge and scheme

- **One gauge.** Born, blocks, loop lines and CTs all use 't Hooft–Feynman gauge (ξ = 1) with explicit Goldstones and ghosts. Ranks then stay ≤ N, whereas unitary-gauge kμkν/M² terms would break rank ≤ N + 2.
- **Complex-mass scheme.** Complex masses and couplings are runtime inputs from the snapshot (R7). The fast NinjaRS engine accepts complex masses with real shifts and a real S-matrix (`ninjars:fast/mod.rs:60-79` [V]). The known complex-mass master defects are risk 13 (§15).
- **No scheme choices in v2.** The γ5 prescription and the finite terms are those of the model's UV/R2 rules: loop_sm sets lhv = 1, and nothing is added to the four-dimensional numerator (`MASTER_PLAN.md:1547-1556` [V]).

### 6.9 Direct-R2 bonus

The inherited per-diagram direct path stays, quarantined, for processes whose loops have at most four propagators. Its roles are validation and the model certifier (§6.3). v2 is model-R2 only.

The plan keeps the (k_μ², ε) monomial labels and the `MuTensorNumerator` ingress, so a future evanescent segment builder (X = Y + q·Z + μ·W, MASTER_PLAN §2.6) can feed the same tries. `r2_strategy = "direct"` is refused for other processes.

---

## 7. Colour

### 7.1 Representation

- **Colour flow.** U(N_c) colour flow with explicitly graded U(1) pieces, so the dressed kinematics never depends on N_c.
- **Born basis.** From `build_color_plan` (`color/plan_build.py:17-24` [V]): single traces, open strings or products of these; optionally the DDM basis where the tree certificate applies.
- **Seed tables.** K[c, b] = ⟨b|κ_c⟩, an exact Laurent polynomial in N_c with rational coefficients, computed by integer cycle counting (`color/adjoint_kernel.py:36-62`) or exact overlaps (`color/connections.py:492`) [V-s]. Fermion signs (§3.3) multiply the leaf weights, not K.
- **No Gram inversion.** Loop colour is never decomposed into the Born basis; only overlaps are needed. This avoids the missing partial-contraction routine (tree report §8d) and the Idenso degree limits (assessment W3).
- **Born Gram.** The Born–Born Gram matrix is needed only for the Born square and CT contractions.

### 7.2 Seed classes

- **Generic (default).** As in §3.4: a per-skeleton colour-flow vector, term-factored. Per point, W_t(h) = Σ_b ⟨b|t⟩ conj(A0_b(h)) is computed once per distinct structure t, and then U0[s] = Σ_t c_{s,t}·W_t.
  - This is correct for any model whose coloured vertices lower to colour flow (Recola2 App. A method [L-s 1705.06053]).
  - Its cost grows with #skeletons × terms. For gg→ggggg FC it would be ≈0.7–2.4×10⁹ complex MAC per point [I], i.e. +0.4–2.4 s on Table 1a. **The FC targets of the gluon-rich rows assume the certified compressions below**, which is why WP2.4 builds the DDM and trace certificates first. The effect is negligible at 6 legs.
- **Certified compressions.** Each needs a loop-level certificate: the model's colour structures in the relevant orders, plus an exact colour-space identity on the smallest instance of the species pattern, plus the cited theorem.

  | Class | Seed per |
  |---|---|
  | Gluon and ghost loops | Ordering σ: Tr(F^{σ₁}…F^{σₙ}) (DDM) [L] |
  | Closed fundamental loops | σ: Tr(T^σ) |
  | One quark line | (σ, L/R): DDM q̄q ring [L] |
  | Several quark lines | Generic, or published decompositions when certified [L-s 1111.4193, 1310.0413, 1908.02695] |

  With compression the per-point seed cost for gg→ggggg FC is ≈1.2×10⁸ complex MAC for 4 loop classes × 360 orderings [V-s Design 1 §5.1], about 2 % of dressing.
- **Proportional classes (WP2.4).** With compression, the seed table also merges classes whose colour vectors are proportional (`colour._ClassKeys`). These are quark rings of different arc words, and generic seeds equal to them. On uu~→ggg this cuts 315 classes to 99.
- **Closed-fermion orientation folding** (Tr(σ) + (−1)ⁿ Tr(σᵀ) in one class) is used only when the attached vertices carry a C-parity certificate.
  - It is not used with Z or W attachments (judge finding 11). Those classes keep both orientations, and their dressing doubles.
  - The counts of Table 1a assume folding for the closed quark loops with QCD attachments (certified for loop_sm QCD vertices). Without folding, FC dressing of the gluon processes would be ≈1.75× (gluon loop plus three closed-quark classes doubled, with a massive b).

### 7.3 Seeds per point and FFT

U0 = A0† K. Leading colour is diagonal.

- FC/NLC use direct exact-cast contraction, or the existing rectangular symmetric-group FFT where certified orbits exist (`color/symmetric_group.py:362`; Rust `recurrence/symmetric_group_fft.rs` [V-s]).
- The `auto | direct | fft` policy and the reported algorithm are kept (MASTER_PLAN §2.16).
- DoubleFloat and arbitrary precision use exact direct contraction.
- Correlated FFT stays unsupported, as today.
- **Implemented (WP2.4).** `loops/v2/colour.seed_fourier` finds the orbits of S_m on the external adjoint labels (all of them, or all but one when every coloured leg is adjoint) directly on the literal Born tensors and the seed classes: regular Born orbits, and complete loop orbits whose stabilisers carry signs (an adjoint ring is fixed up to (−1)ⁿ by its reflection). Each Born/loop orbit pair is certified on every exact cell, K[class(g), b(h)]·sign_g = k(h⁻¹∘g). Other pairs stay direct. The native `V2SeedFourier` (`rust/loop-reduction/src/runtime/v2_seeds.rs`, binding `_loop.V2SeedFourier`) contracts the certified pairs for a batch of helicity rows with `SymmetricGroupFftPlan`. Kernels are projected exactly per sector degree P and Born grade, then rounded once.
  - The slice's `SliceRequest.colour_contraction` defaults to `auto`: binary64 at NLC/FC, with `PointResult.colour_contraction` reporting the algorithm. `fft` refuses a table without a certified pair, and LC is refused. Exact dressing, DoubleFloat and arbitrary precision stay direct.
  - Orbits found:
    - gg→gg and gg→ggg: one Born orbit and two loop orbits (adjoint rings and closed traces).
    - uu~→ggg: 19 loop orbits cover every cell.
    - gg→ggggg: S_6, with 1 080 classes in two orbits.

### 7.4 Projection

For each virtual squared-order sector:

1. P is the maximum N_c degree over all non-zero exact cells: seed cells for every class contributing to the sector, CT cells (with CT colour grades, `physical_color.split_color_grades` [V-s]) and field cells. P is structural and independent of helicity, ε and numerics.
2. LC keeps degree P; NLC keeps P, P−1 and P−2, odd powers included; FC keeps all.
3. N_c = 3 is substituted after projection, and the external average after that. N_f has degree 0.

Born sectors keep independent degrees. Truncation acts on the contracted cells K[c, b], never on a Gram matrix alone (contracts §2.6 caveat). Without a continuation, only FC is available (§6.7).

### 7.5 Leading colour, structurally

LC prunes skeletons before any numerics, by the face rule of the leading-colour report §4.1 [V-s, verified there against exact SU(N_c) overlaps].

| Skeleton class | Maximal degree | Kept at |
|---|---|---|
| Planar gluon or ghost ring of a leading Born word | P | LC |
| q̄q adjacent, gluons on the gluon side (A^L, p = 2); quark arc in a colourless gap | P | LC |
| Four-quark box with crossed colour, gluon outside or on a gluon arc | P | LC |
| Closed quark loop (any mass class) | P−1 | NLC |
| Flow overlap differing by one quark transposition (identical flavours) | P−1 | NLC (in seeds) |
| Two-sided ring / double trace; A^R; abelian vertex correction across an emission; gluon on a quark arc inside a four-quark box; uncrossed four-quark box | P−2 | NLC |

**Implementation.**

- **Pruning.** The trie builder applies the local rules: word admissibility through the tree engine's LC segment tables (`generation/dag_color.py:186-243, 1041-1076` [V-s]) extended by cut-terminated segments; the arc rule; no closed quark loops; no U(1) or Fierz-singlet currents. Each surviving class's degree is then certified exactly from its seed table.
- **Seeds.** LC seeds are diagonal: U0 = N_c^P·κ·conj(A0(σ, h)). There is no interference between orderings and no Gram matrix.
- **Strict-LC Born.** The strict LC Born uses the graded Fierz and U(1) pieces (§3.2 item 4).
- **Counts** in the providers' configuration [M `revision/mb0_all.log`; massive-b values in `otf/run_all.log`, `lcs/lc_structure_counts.log`]:

| Process | LC orderings | LC / FC skeletons | LC/FC dressing (structured) | LC/FC families | LC/FC time (Table 1a recipe) |
|---|---|---|---|---|---|
| gg→gggg | 60 | 3,060 / 9,180 | 0.35 | 0.50 | 0.37–0.42 |
| gg→ggggg | 360 | 40,680 / 122,040 | 0.35 | 0.50 | 0.37–0.42 |
| uu~→ggggg | 120 | 13,560 / 91,440 | 0.18 | 0.52 | 0.22–0.28 |
| uu~→e⁻e⁺ggg | 6 | 306 / 3,990 | 0.12 | 0.35 | 0.13–0.14 |
| uu~→e⁻e⁺dd~g | 4 | 144 / 2,484 | 0.19 | 0.47 | 0.21–0.23 |
| gg→tt̄gg | 24 | 1,224 / 6,888 | 0.26 | 0.35 | 0.26–0.27 |
| gg→tt̄e⁻e⁺g (QCD) | 6 | 306 / 3,990 | 0.15 | 0.28 | 0.14–0.15 |

With a massive b the gluon-process ratios are 0.26 (dressing), 0.33 (families) and 0.27–0.30 (time), because FC then has one more closed-quark class. The inherited engine measures LC/FC ≈ 1, because it projects after computing everything (`colour-measurements.json` [V-s]).

The same face rule makes seeds composition-independent within an ordering, so schedule M is valid at LC without any decomposition theorem [I] (§2.2).

### 7.6 Next-to-leading colour

NLC keeps every class whose **exact** maximal degree over the sector's Born basis is ≥ P − 2, and truncates the cells to degrees P, P−1 and P−2. No class is dropped on a literature-based expectation; the degree always comes from the exact seed table. Structurally:

- **gluon processes:** NLC adds the quark-loop classes (P−1). The double-trace and off-diagonal weights of the same gluon-loop skeletons arrive through truncated seeds.
- **quark lines:** NLC adds A^R, mixed and closed primitives, and every class whose exact degree reaches P−2.

**Literature check (O14).** In BDK's q̄qggg colour sum [L hep-ph/9409393 eq. NLOCrossSection, l.1748-1772]:

- against the leading (N_c²−1)² A_{5;1}, the other two orderings of A_{5;1} and two of the three A_{5;3} terms enter at relative N_c⁻², i.e. degree P−2, inside NLC;
- the remaining A_{5;1}, A_{5;3} and A_{5;4} terms enter at N_c⁻⁴.

WP2.4 tests v2's exact-N_c interference for uu~→ggg against this formula, with the partial amplitudes obtained from resolved-colour seed columns on the BDK colour structures. Revision 1's claim that A_{5;3} enters only at N_c⁻⁴ was a misreading and is withdrawn.

NLC/FC cost ≈ 0.9–1.0: the value of NLC is accuracy, not speed. The tree NLC truncation (`color/contraction_factors.py:100-108`) is a different definition and is not reused for loops [V-s].

**Opt-in, labelled, non-default modes** [L 1209.0100 `njet.tex:529-540`, tab:timedsymgluons l.1231-1251; L-s 0907.1984]:

- separate LC and (FC − LC) outputs for colour-expansion sampling;
- **integrated-equivalent full colour** (NJet's de-symmetrised colour sums). Valid only after integration over a phase space symmetric in identical final-state particles:
  - v2 generates loop seed classes only for the de-symmetrised orderings, and the seeds use the de-symmetrised colour matrix;
  - NJet reports 7 gluons going from 171 to 3.07 s;
  - labelled, never the default, never reported as pointwise FC; evaluated in WP4.7.

### 7.7 Directed colour correlations

V_{αβ,k} = B†K^{αβ}_{BL}L_k + L_k†K^{αβ}_{LB}B, with K_{LB}^{αβ} = (K_{BL}^{βα})† (MASTER_PLAN §2.17).

- Each (history, direction) is a seed column. The second direction is seeded with K^{βα}, reduced, and conjugated after reduction. Results stay complex per history.
- The colour operators come from `color/correlator_matrices.py:295` [V-s], applied on the Born side.
- The complete-family LC/NLC degree conventions (`loop_runtime.rs:13-14` [V-s]) are kept.
- **Cost.** Dressing and reduction scale with 2·N_hist columns: 42 at 7 legs with 21 pairs.
  - Above the break-even, a generation-time cost model switches to schedule 3: F with amplitude-level (h, b) closures, plus a transposed NinjaRS reduction that returns each family's integrals of all monomials once (request 2 in §11.5). Every observable then becomes a dot product.
  - The break-even is ≈15–20 columns at 6 legs and ≈45–60 at 7 legs [M/I Designs 1 and 2].
  - Schedule 3 is adopted only after a measured prototype, because the owner's transposition study is a symbolic instruction count, not a timing (§11.5).

### 7.8 Caps

The loop path never passes `max_sectors` (`color/plan_build.py:21` [V]) and does not inherit `MAX_ADJOINT_GLUONS = 12` (`color/adjoint_kernel.py:20` [V]). Direct contraction is always available when FFT eligibility (degree ≤ 10 today) fails.

Verified for WP2.4 [V]:
- `loops/v2` calls neither `color/plan_build.py` nor `color/adjoint_kernel.py`. Its Born plan is the tree engine's trace basis, and no caller in the repository passes `max_sectors`.
- The seed FFT has no degree bound. It enumerates S_m only when |S_m| ≤ the Born basis size, the structural condition for a regular orbit, and contracts directly otherwise.
- The native plan's own bound is its factorial and `u32` index storage.
- `MAX_ADJOINT_GLUONS` remains the tree engine's DDM-basis resource bound (dense int64 kernels on S_{n−2}) and is outside the loop path.

---

## 8. Helicities and spin

### 8.1 Catalogue, selection and identifiers

- **Catalogue.** The full Cartesian helicity catalogue in leg order is kept (contracts §2.1).
- **Selection mask.** Massless chirality conservation plus Born-zero rows. A row h with A0_b(h) = 0 for all b gives a zero seed and is dropped at the leaves, but only for observables whose bra is the Born (assessment R3).
- **Identifiers.** Resolved helicity IDs keep indexing the full catalogue. `structural_zero_helicity_count`, hard-coded to 0 today (`runtime/virtual.py:508` [V]), reports the dropped rows.

### 8.2 Summation

- **B and M** sum the helicities of each block at the step that attaches it: objects carry only rows of undressed legs.
- **F** keeps attached-leg rows and sums at closure. That makes it the natural producer of helicity-resolved outputs.

### 8.3 Parity halving (optional, wave 4)

**Identity.** Under three conditions, the conjugate helicity configuration gives U0(c,−h)·c_α(−h) = conj(U0(c,h)·c_α(h)) per monomial in the real monomial basis:

- real external momenta;
- real couplings and masses (no widths);
- parity-certified kernels: tree, loop-segment and CT.

Hence C_α = 2 Re C_α^{half} [I, derived; the absorptive parts sit in the integrals, which are applied afterwards].

**Implementation.**

- Restrict seed rows to h_j = + for one leg j chosen inside the last-dressed position of each family's frame, so all earlier steps carry half the rows.
- Apply 2 Re to the Born-weighted coefficients before reduction.

**Gating and scope.**

- The certificate is a loop-level re-proof: the tree engine's `global_helicity_flip_equivalence_is_proven` (`models/external_catalog.py:465-472` [V]) extended to loop kernels, plus a numerical check on random points. It is never inherited (tree report §4).
- It is invalid with chiral couplings, widths or the complex-mass scheme.
- In the required set it applies to **gg→gggg, gg→ggggg, uu~→ggggg and gg→tt̄gg only**: vector-like QCD couplings, real masses, zero widths.
- It does not apply to uu~→e⁻e⁺ggg, uu~→e⁻e⁺dd~g or gg→tt̄e⁻e⁺g, which are generated with QED ≤ 2 and include Z exchange.
- Expected gain: ×1.6–2 on dressing, ×1.25–1.6 per point, for those four processes.

### 8.4 Spin sources and correlations

- **Literal sources.** Complex vectors replace the polarisation of the source legs in both the Born and the blocks (`spin_sources.py:17-40` semantics [V-s]), and the source rows replace that leg's helicity rows. Joint classes and spin-correlation tensors form seed columns. Non-transverse sources use full kernels (§3.2 item 2).
- **Asymmetric Ward mode** (validation only, judge X2). The Born bra uses physical polarisations; blocks and CT = 1 currents use ε_j → p_j with full kernels. Seeds and blocks are computed separately, so this needs no new machinery. The renormalised loop amplitude must give zero interference, which tests UV + R2 completeness and gauge invariance per point.

### 8.5 Resolved outputs

- **Per helicity:** schedule F, or one-hot seed columns (×H cost at 7 legs).
- **Per Born colour component:** seed columns restricted to the component's Born basis elements. Components still sum exactly to the inclusive result, and `ResolvedVirtualEvaluation.total()` equals inclusive. The `kind` is renamed if anchors change (§1.6).

---

## 9. Flavour mappings

- **Representative-only programs.** Identity-port members are records executed by the representative on the same momenta (`AdmittedFlavorMember`, `loop_runtime.rs:467-470` [V-s]).
- **Admission** keeps `prove_flavor_relabeling` (`loops/flavor_mappings.py:186` [V]) and runs it on the prepared model before compilation (assessment R11).
- **Permutation aliases** (e.g. u~ u for u u~) become relabelling tables: momentum and label permutations on the representative plan, with Bose transport as a special case and fermion signs from the same permutation parity as §3.3. This is the same mechanism as primitive templates. Public generation rejects aliases today (`loop_service.py:70-74` [V-s]); that restriction is lifted only with this proof-free relabelling (R11).
- **Retired.** The paired-program comparison of `flavor_transport.py` (contracts §2.7).
- **Composition.** Correlations and spin sources compose with members, because member execution reuses the representative's seed columns.

---

## 10. Precision and rescue

### 10.1 Carriers

- **One instruction stream for f64, DoubleFloat and Float.** Dressing and closure are generic Rust. Segment builders are SymJIT (f64) and exact evaluators (DD/AP). Seeds and colour tables are exact rationals cast per carrier. Ingress is exact decimal → rational, and the invariant table is exact until cast (contracts K4, K6).
- **Trees.** The tree recurrence lane gains DoubleFloat and arbitrary-precision prepared kernels built from the retained `exact_expressions` (tree report §7.3).
- **NinjaRS.** The fast engine for f64; the generic engine for DD/AP.
  - Today the generic engine's numerator ingress is `PolynomialNumerator`, built from a Symbolica expression (`ninjars-symbolica/src/polynomial.rs:258` [V]). At rank 7 it has 330 parameters, and its compile cost is unmeasured (WP2.3 measures it).
  - A generic-precision tensor numerator exists only in the CLI harness (`ninjars-cli/src/case_check/harness.rs:1290`, `TensorGenericNumerator<T>` [V]). Request 8 (§11.5) asks the owner to upstream it.
  - Float(128) needs Symbolica 9cdc2949 (NinjaRS guidance [V]), pinned since the coordinated re-pin (WP0.2).

### 10.2 Triggers, cheapest first

1. **NinjaRS status.** Lane status ≠ Success, a master failure, or a non-finite value. This is free. The fast engine has no cancellation test (NinjaRS guidance [V]).
2. **Reduction-side re-test of every family (default).** For each (family, point, sector), the stored family polynomial is reduced a second time in a relabelled frame.
   - **Relabelling.** The propagators are rotated cyclically by a fixed per-family k_f ≠ 0. This maps q → q − V_{k_f} and permutes the shifts and the S-matrix.
   - **Re-expansion.** The polynomial is re-expanded by the exact affine map (`loops/v2/poly_affine.py`; in Rust, the same generic routine in the working precision, O(4·R·C(R+4,4)) operations).
   - **Masters.** Box and triangle masters hit the cache. Bubbles with swapped mass order miss, because they are keyed by ordered arguments (§11.2).
   - **Estimate.** |Ω_f − Ω′_f| is the per-family error estimate. This is MadLoop's production check in the form recommended by NinjaRS guidance (b). Ninja's authors recommend a global scaling or rotation test, not Ninja's internal tests, as the accuracy estimate [L 1403.1229 `main.tex:1642-1651`]; the rescaling variant below is the scaling test applied per family.
   - **Rescaling variant** (measured in WP2.3). Replace c_α by x^{d_f − |α|}c_α, scale all invariants, masses and μ by a non-dyadic x so that rounding differs, and compare after undoing x^{4−2N+d_f}. Here d_f is the family numerator's mass dimension. This is NJet's scaling test applied per family [L 1209.0100 §Accuracy]. It recomputes the masters and so also tests them.
   - **Cost.** +0.75 R (relabelling, cached masters) or +1.0 R (rescaling), included in Table 1a: +12–25 % per point at 7 legs.
3. **IR-pole check** per point and sector, kept as an additional signal.
   - The 1/ε² and 1/ε coefficients of the renormalised result are compared with Catani's operator, built from the Born and the colour-correlated Borns (tree correlators) plus UV CT poles. Massive emitters use the massive dipole form; electroweak sectors use the QED analogue.
   - It is free and catches missing counterterms.
   - It is not sufficient: pole residues involve only divergent integrals, so finite boxes, the R1 rational part and Gram-determinant failures in N ≤ 5 families pass it [I].
4. **Optionally,** the generic engine's κ cancellation test (guidance (a)).

**Error budget.** The point's estimated absolute error is E = Σ_f |ΔΩ_f|, compared with the target accuracy × |Σ_f Ω_f + CT|. When the target is missed, the families with the largest |ΔΩ_f| are recomputed (§10.3) until the remaining estimate meets the target. Per-family locality makes this targeted.

**Relaxation.** The default (every family, every point) may be relaxed only after WP2.3 measures the false-negative rate of each trigger against Float(256) on ≥ 10⁴ points per required process. Possible relaxations are N ≥ 5 families, or sampled points with a reported rate.

### 10.3 Escalation

Escalation always starts from the original inputs and never promotes binary64 values (MASTER_PLAN §2.18):

1. **Per (family, point):** recompute trees, seeds and that family's trie (or its M closures) in DoubleFloat from the original decimals with the same source snapshot, reduce with the generic engine, and replace the lane. Per-family locality makes this cheap.
2. Whole point in DoubleFloat, then Float(p) per policy.

Statuses:

- They stay success / rescued / failed / accuracy-not-established, with one implementation (`rust/loop-reduction/src/verification.rs`); `loops/stability.py` is deleted.
- Lane compaction replaces the whole-batch restart (`loop_execution.rs:749-826` [V-s]).
- The single-level defect (`verification.rs:346-351` [V]) is fixed per decision 6 in §1.6.
- The R2 strategy is fixed during rescue.

### 10.4 Expected rescue load and calibration

- NJet needed quadruple precision for 1.37 % (6 gluons) and 8.63 % (7 gluons) of points with unitarity in double [L 1209.0100 `njet.tex:1161-1227`].
- Ninja's exact Laurent expansions avoid the OPP sampling part of that instability [L-s 1403.1229].
- NinjaRS 0d0d7e7 lost 5×10⁻⁹–10⁻⁵ on ill-conditioned n = 7 lanes in binary64; the fix is in the pinned main aa6a6307 (guidance [V]; not re-measured here). The guidance expects many n = 7 μ² lanes to need DoubleFloat at a 10⁻⁸ target.

WP2.3 measures on 10⁴ points per process:

- the rescue rate;
- the rescue cost;
- the false-negative rate of each trigger (relabelling, rescaling, pole check, κ).

### 10.5 Optional labelled hybrid tier (needs sign-off)

Re-reduce flagged families in DoubleFloat from their binary64 coefficients (the OpenLoops 2 hybrid-precision analogue [L-s 1907.13071 §5]). It is off by default, never reported as DoubleFloat accuracy, and has its own status.

---

## 11. NinjaRS and oneloopmaster interface

### 11.1 Hand-off per family and channel

| Item | Content |
|---|---|
| Denominators | N_f lines: real shifts from leg masks; masses from mass ids (complex under the complex-mass scheme) |
| S-matrix | N_f × N_f from one canonical invariant table per event, on-shell bindings patched |
| Rank | True structural rank; never over-declared, since over-declaring costs up to 981× in accuracy (guidance [V]) |
| Numerator | C(R+4,4) complex coefficients scattered into `TensorNumerator` lanes; lane-major storage, `coefficients_mut` (`ninjars:fast/tensor.rs:30-90` [V]); no text |
| DD/AP numerator | `PolynomialNumerator` (Symbolica expression) until the owner upstreams the generic tensor numerator (request 8) |
| Re-test | The same coefficients after the exact affine relabelling (§10.2), same bucket |
| Output | Raw Laurent [finite, 1/ε, 1/ε²] of μ_R^{2ε}/(iπ^{2−ε} r_Γ) ∫ N/ΠD, post-processed as 2 Re(i/16π²·Σ_f Ω_f) (`rust/loop-reduction/src/lib.rs:5-9, 59-69` [V-s]); directed cross terms stay complex |

### 11.2 Batching, rank and masters

- **Buckets.** One `FastAmplitude::new(N, R, FastOptions::default())` per bucket; rank ≤ N + 2 is enforced (`ninjars:fast/engine.rs:620-632` [V]).
  - Lanes are (family, point, sector, column), and every lane carries its own shifts, masses and S-matrix (`FastInput`, `ninjars:fast/mod.rs:60-79` [V]).
  - So unrelated families share calls: 5–11 physical calls per batch instead of one per family (§11.4). The re-test reuses the same buckets.
- **Per-lane rank.** Pinned NinjaRS aa6a6307 reduces a lane declared at rank n+2 at its own rank (`FastNumerator::lane_rank`), bitwise as declaring that rank, so buckets may merge by N. `runtime/v2_reduce.rs` still buckets by (N, true rank), with the same results; merging by N is open.
- **Master cache.** A pyAmpliCol-side `CachingMasters` wrapper implements `FastMasters` (`ninjars:fast/masters.rs:56-64` [V]) and is cleared per batch.
  - Keys are the exact bit patterns of the request arguments. This works because all S-matrices come from one invariant table per event.
  - **Canonicalisation only where the outputs are invariant:** C0 under the triangle's symmetry group acting consistently on its external invariants and masses, and D0 under the dihedral relabelling of the box acting consistently on (p_i², s12, s23, m_i²). A0 needs none.
  - **Bubbles are keyed by the ordered tuple** [p², m0², m1², μ²]. `BubbleRank2/3/4` return [B0, B1, B11(, B111, B1111)] for ordered arguments (`ninjars:fast/masters.rs:16-26`, `fast/engine.rs:1817` [V]), and the tensor coefficients are not invariant under m0 ↔ m1: B1(p²; m0, m1) = −B0 − B1(p²; m1, m0). Revision 1's mass-swap canonicalisation was wrong for unequal-mass bubbles and is withdrawn.
  - The canonicalisation is applied identically in cached and uncached runs, so they stay bitwise equal. WP1.6 tests unequal-mass bubbles on gg→tt̄ (top–gluon).
  - Integer master ids from generation are an upstream optimisation (request 4).
  - The measured redundancy at 7 gluons is 308,448 requests for 2,822 distinct masters (109×) [M recola-ninjars §5.3].
- **DD/AP.** Same structure with the generic engine and its master provider.

### 11.3 Topology bookkeeping, and the user's questions

**Bookkeeping**

| Aspect | Rule |
|---|---|
| Identity | Canonical cyclic (leg mask, mass id) sequence (§3.3): Recola's (ho, hm) |
| Canonicalisation | Key = minimum over rotations and reflections, at generation. Execution frames: per-family cost-optimal for B; leg-1 rule for F; cut-ordering frames for M. Oracles map frames with exact affine transforms |
| Selection | Structural rule plus certified-scaleless drop (§3.3); the drop is logged |
| Merging in the recursion | All partial results with equal key are summed: across block orderings, colour classes, species, helicities, sectors and columns. B: within (family, cut species, undressed contents). M: across families with equal prefix sets and undressed word. Never across denominators |
| Merging at closure | Species with equal masses, all cut species and all colour classes accumulate into one polynomial per (family, sector, column) |
| Merging at reduction | (N, R) buckets as lanes; master cache; optional stage 2 below |
| Born weighting | B: seeds at the leaves. M: seeds per ordering at the cut. Every state is Born-weighted and summed over the helicities of its dressed legs, so each family closes into the Born-weighted, helicity-summed numerator NinjaRS needs: one reduction per (family, sector, column), plus the re-test |
| Pinched topologies | Separate families; no lifting (MASTER_PLAN §2.13) |

**Lifting pinched families into parents** (user guidance). Rejected in both forms:

- *Globally* (multiply by the missing D_j): +2 rank and μ² per lifted line, and at most ≈2 lifts fit rank ≤ n + 2.
  - Cost model at gg→ggggg: 0.80 s (≤ n+1) and 2.7 s (≤ n+2), against 0.79 s for one reduction per family.
  - Probe errors: 3×10⁻⁴ (n = 6) and 8×10⁻³ (n = 7), against 10⁻⁷–5×10⁻⁶ for separate reductions [M recola-ninjars §4.1; recursion-literature §4.3].
- *Inside the recursion* (merging members after their first step):
  - Even the idealised bound with no rank inflation only takes the 7-gluon dressing from 9.0×10⁸ to ≥ 4.0×10⁸.
  - The alignment it needs (OpenLoops' pinch-invariant cut) costs 2.7×.
  - Families with N ≤ 4 cannot be lifted, and lifted hosts reduce at rank n+1 or n+2 [M `otf/lift_bound.log`].

**Family-tree / integrand-sum reduction sharing cuts without rank increase** (user guidance). The mathematics is exact: Ninja's extraction is linear, triangles need no subtraction, and bubble and tadpole subtractions are linear in the parent residues [L 1203.0291 `introduction.tex:98-117`; V `ninjars:fast/subtractions.rs` per physics judge]. GoSam already reduces pinched diagrams together with their parent "with virtually no additional computational cost" [L 1111.2034 `diagen.tex:100-130`]. Three realisations:

| Realisation | Gain on the NinjaRS stage | §2.13 | API | Decision |
|---|---|---|---|---|
| (b2) per-member division / process-level cut lattice fed by family tensors: each family expanded on its cuts through NinjaRS's own jets; geometry, subtractions and masters once per cut | ≈1.15–1.3× (the expansions, 72–75 % of NinjaRS time with cached masters, are unchanged) [I, from the measured split in recola-ninjars §3.5] | Compliant | Owner extension: a **public** jets API (`fast/tensor_jets.rs` is private, `ninjars:fast/mod.rs:46` [V]; request 7) | Default stage 2, if gated in |
| (b1) cut-restricted lift into parent groups: callback returns T_i = Σ_m N_m Π L_j^(i) with affine L_j on cuts containing line i; rank class kept | 1.5× (n = 6), 2.0× (n = 7) at rotation-noise accuracy, on synthetic numerators [M recola-ninjars §4.2.3] | Needs an amendment (cut-reduced extra denominators) | None to start (`cut` is passed to callbacks, `ninjars:fast/numerator.rs:58-66` [V]) | Only with sign-off and validation on physical numerators |
| Residue lattice with pyAmpliCol jets (Design 3) | Masters once per cut; best for many channels | Compliant | New NinjaRS lattice API plus private jets | Not adopted (private jet arithmetic duplicates NinjaRS; production would wait on an API) |

**Gate.** Pursue stage 2 only if NinjaRS exceeds 30 % of the total after the master cache. At 7 legs FC the stage is 22–46 % of the target including the re-test (Table 1a), so stage 2 can bring ≈3–10 % overall with (b2) and ≈7–20 % with (b1). The master cache comes first: it alone removes 0.23–1.5 s at gg→ggggg [M].

**Recola's tensor-integral bookkeeping** (user guidance). Adopted:

- the identity (ho, hm);
- canonical cut and orientation rules during the recursion (F);
- species sharing by mass;
- the counterterm flag in the current key.

Recola's pinch sharing happens inside Collier's call-position cache [L-s 1604.06792 §4]. That cache shares tensor sub-reductions as well as scalar integrals. The NinjaRS master cache shares only scalar masters, so it reproduces the scalar part of Collier's sharing, not its tensor-reduction sharing. In integrand reduction the corresponding work (per-cut expansions) is per family by construction, and stage 2 (b2) is its analogue.

Recola's "reduce numerator-independent integrals once, contract per (colour, helicity)" is unavailable to integrand reduction, which needs the numerator first. It returns through the transposed reduction (schedule 3, §11.5 request 2) for many-observable requests [V-s recola2 `amplitude_rcl.f90:1470-1496`].

**OpenLoops on-the-fly reduction** (user guidance). Its output, low-rank polynomials over fixed denominator sets, would be acceptable to NinjaRS; that is not why it is rejected. It is not adopted because:

1. **Speed.** OpenLoops' own benchmark finds reduction after dressing (OpenLoops2+Collier) faster, depending on the process; OFR is kept there for stability [L 1710.11452 `OL_OFRED_paper_epjc.tex:3836-3846`].
2. **Private mathematics.** It would be private reduction mathematics next to NinjaRS, against the user's reuse rule.
3. **Integrand-level validity.** The del Aguila–Pittau step holds at integrand level only for N ≥ 4 [L-s 1710.11452 §4.1]. It needs Gram-determinant expansions and hybrid precision.
4. **More families.** Pinch closure adds families: uu~→e⁻e⁺dd~g goes from 244 to 422 [M].

A few integrand-level steps could bring rank > N+2 effective-theory numerators into NinjaRS's domain. That belongs in NinjaRS if ever needed.

### 11.4 NinjaRS reductions per event (per squared-order sector)

Providers' configuration (massless b). The re-test doubles the reductions in the same buckets with cached box and triangle masters.

| Process | Colour | Families = reductions | Physical calls (buckets) | Distinct masters after cache | Stage-2 groups | NinjaRS stage: mass-aware model / histogram model |
|---|---|---|---|---|---|---|
| gg→gggg | FC | 1,100 | 5 | 627 (of 20,800 requests) | 120 | 40–45 ms / 44–51 ms |
| gg→gggg | LC | 550 | 5 | 310 | 60 | 17–20 ms / 22–26 ms |
| gg→ggggg | FC | 9,414 | 6 | 2,822 (of 308,448) | 720 | 0.79–0.82 s / 0.85–1.0 s |
| gg→ggggg | LC | 4,707 | 6 | 1,407 | 360 | 0.36–0.38 s / 0.43–0.50 s |
| uu~→ggggg | FC | 5,258 | 11 | 1,724 (of 146,576) | 420 loose / 971 strict | 0.19–0.21 s / 0.22–0.29 s |
| uu~→ggggg | LC | 2,712 | 11 | 1,122 | 120 / 671 | 80–90 ms / 92–124 ms |
| uu~→e⁻e⁺dd~g | FC | 242 (Recola TI table: 244) | 10 | — | 39 | 2–4 ms / 1.8–3.3 ms |
| uu~→e⁻e⁺ggg | FC | 463 | 9 | — | — | — / 9–13 ms |
| gg→tt̄gg | FC | 1,086 | 9 | — | — | — / 19–27 ms |
| gg→tt̄e⁻e⁺g (QCD) | FC | 579 | 9 | — | — | — / 10–14 ms |
| gg→tt̄e⁻e⁺g (EW) | FC | ≈1,450 [I] | ≈10–12 [I] | — | — | — / ≈25–50 ms [I] (gate E0) |

Sources:

- The mass-aware model of the recola-ninjars report skips scaleless cuts [M recola-ninjars §5.2–5.3, §6].
- The histogram model applies the same constants to the (N, R) histogram without scaleless skips [M `revision/ninjars_stage.log`]. Table 1a uses it for every row.
- Reductions multiply by the number of active sectors and observable columns; physical calls stay at the number of buckets.

**On MadLoop's loop groups.** MadLoop 5's published table has 1,168 loop groups for gg→tt̄gg, which matches our 1,164 families with a massive b [L 1604.01363 timing table]. The MG5 3.7.0 exports on this host have 1.6–3.9× more groups than our canonical families: 2,459 (gg→tt̄gg), 4,241 (gg→gggg), 998 (uu~→e⁻e⁺ggg), 406 (uu~→e⁻e⁺dd~g), 1,191 / 2,958 (gg→tt̄e⁻e⁺g QCD / EW) [V `NLOOPGROUPS` in each `loop_matrix.f`]. Only their EW/QCD ratio is used here.

### 11.5 Requests to the NinjaRS owner

These may be pushed to NinjaRS main under the 20 October authorisation (`MASTER_PLAN.md:96-103` [V]), in priority order:

1. **Coordinated re-pin** to the new mains with per-lane rank: done (WP0.2). Still requested: an upstream test that lanes of unrelated families give bitwise the same result batched and unbatched.
2. **Transposed / multi-numerator reduction.** H numerators share one geometry; equivalently, each family's integrals of all monomials are returned once. This enables schedule 3 for correlations and resolved outputs.
   - The owner's study (`docs/audits/transpose-probe.md` [V]) is a symbolic instruction count on a three-propagator, rank-3 toy, starting from Laurent callback heads.
   - It is a negative result at one numerator per changing point (`:21`). Its thresholds, H > 1.85 and H > 3.97 for fixed geometry, are explicitly not timing predictions (`:45`).
   - A measured prototype is required before schedule 3 is adopted.
3. **Per-(cut, lane) conditioning diagnostics:** minimal uncut-denominator leading coefficient, Gram determinant. These feed the error budget (§10.2).
4. **Optional integer master ids** in master requests, so the cache needs no hashing.
5. **Gated stage 2:** process-level lattice in tensor-provider mode or per-member division; or, after sign-off, a layout-aware affine-product builder plus a cut-reference adapter.
6. **Optional** coefficient-major `TensorNumerator` ingress, to avoid the closure transpose.
7. **Public jets API** for (b2), if stage 2 is gated in: `fast/tensor_jets.rs` is private today.
8. **Upstream the generic-precision tensor numerator** (`TensorGenericNumerator<T>`, now only in `ninjars-cli/src/case_check/harness.rs:1290`), so that DD/AP ingress needs no Symbolica expression.

---

## 12. Artifacts and public API

### 12.1 Artifact

- **Plan v4.** A binary PACBIN member under the existing kind `LOOP_RUNTIME_PLAN = 11` (`generation/evaluator_container.py` [V-s]).
  - The tree recurrence plan stays an existing member.
  - Kernels are referenced in the prepared model bundle (SymJIT applications and exact states, transverse and full variants).
  - Exact seed tables and CT columns are exact-rational members.
- **Capability** `rusticol.loop-runtime.v4`. No backward compatibility (AGENTS.md:10, contracts K9).
- **Kept envelope:** schema-v3 directory, transactional writer and loader admission (`rust/crates/rusticol-core/src/loop_artifact.rs`). `LoopExecutionMetadata` (`generation/loop_artifact.py:31-91` [V-s]) is simplified to: families, buckets, schedule, colour accuracy and degrees, R2 strategy, re-test mode, rescue policy.
- **Sizes:** < 20 MB at 6 legs, 20–80 MB at 7 legs [I], against 394 MB today at 6 legs.

### 12.2 Public API and configuration

- **Unchanged surface.**
  - `Generator(RunConfig)` with `process.loop_coupling_orders`, then `generate_loop_artifact` (`generation/loop_service.py:577-794` [V-s]).
  - `Runtime.load → VirtualRuntimeBackend` with `evaluate`, `evaluate_resolved` and `evaluate_correlated(_many)`.
  - The result types (`api/virtual_results.py`).
  - `LoopInputs` and `CompleteLoopResult` (`loop_execution.rs:114, 188` [V]), `LoopLaneFailure` (`:235` [V]).
  - The C, C++, Fortran, Rust and UMAMI SDK ABI.
- **Changes.**
  - `VirtualColorContraction.algorithm` (`Literal["direct", "symmetric-group-fft"]`, `api/virtual_results.py:141` [V]) gains `"diagonal"` for LC seeds.
  - The resolved-colour `kind` is renamed if decision 3 is taken.
  - `structural_zero_helicity_count` is populated.
  - Per-point diagnostics gain the schedule, the per-family error estimate, the rescue tier and per-stage times.
- **New expert options** (defaults resolve automatically):
  - `[loop] schedule = "auto" | "forward" | "backward" | "merged"`;
  - `[loop] reduction_grouping = "family" | "lattice"`;
  - `[loop] accuracy_retest = "relabel" | "rescale" | "off"` (default `relabel`; `off` reports accuracy-not-established);
  - `[loop] colour = … | "integrated-full"` (opt-in, labelled, §7.6).
  
  The existing stability policy and `r2_strategy` are unchanged; direct R2 is limited to ≤ 4-point loops.
- **Delivered path = measured path.** `virtual_runtime.rs:210-216` hard-wires `"interpreted"` [V]; the v2 runtime uses SymJIT kernels.

### 12.3 Commands and the live report

- `inspect` shows family, skeleton and trie counts, certified-zero families, buckets and degrees.
- `profile` and `benchmark` report numerator dressing, NinjaRS, the re-test and masters separately, as the live benchmark report requires (`MASTER_PLAN.md:1405-1451` [V]).
- The report quotes providers in both production and protocol modes, labelled.
- The report is updated at every gate.

---

## 13. Reuse and deletion

### 13.1 Reused (exact modules)

| Area | Modules |
|---|---|
| Model and rules | `loops/compiler.py` (`prepare_loop_request_model`), `loops/model.py`, `loops/counterterms.py` (`prepare_counterterm_model`, `CountertermRule`; not the matcher), `loops/wavefunction.py` (rules), `loops/orders.py`, `loops/physical_orders.py`, `loops/kinematics.py` (conventions), `loops/color_projection.py` (degree semantics), `loops/physical_color.py::split_color_grades`, `loops/qcd_r2_color.py`, `loops/source_color.py`, `loops/scalar_bindings.py`, `loops/scalar_frontend.py`, `loops/covariant_model.py`, `loops/model_restrictions.py`, `loops/request_scope.py`, `loops/flavor_mappings.py` (`prove_flavor_relabeling`) |
| Oracles and the direct-R2 bonus | `loops/graphs.py` (FeynKit catalogue, families, overall factors: oracle only), `loops/evanescent.py`, `loops/evanescent_gauge.py`, `loops/direct_kernels.py` (bonus and certifier) |
| Tree engine | `generation/dag_compiler.py`, `dag_types.py`, `dag_color.py`, `fermion_ordering.py` (also for loop fermion signs), `helicity_replay.py`, `physics_metadata.py`; `color/plan_build.py`, `plan_types.py`, `plan_replay.py` (pairing parity), `connections.py`, `contraction_trace.py`, `adjoint_kernel.py`, `fft_basis_strategy.py`, `symmetric_group.py`, `correlator_matrices.py`; `models/compiler_kernels.py`, `compiler_entry.py`, `prepared_catalog.py`, `prepared_compile.py`, `recurrence_catalog_builder.py`, `compiler_color_flow.py`, `external_evaluation.py`, `tensors.py`; Rust `recurrence/{construct,layout,color,direct_plan,direct_runtime,symmetric_group_fft}.rs` |
| Rust loop shell | `rust/crates/rusticol-core/src/loop_execution.rs` (trait shell without `schedule`, `LoopInputs`, `CompleteLoopResult`, `LoopLaneFailure`, `evaluate`/`evaluate_resolved`/`evaluate_directed`), `loop_runtime.rs` (surviving records: `StateLeg`, `OrderLayout`, `ModelBinding::Source`, `ColorProjection`, `DirectedHistory`, `SpinProjection`, `AdmittedFlavorMember`), `loop_artifact.rs`; `rust/loop-reduction/src/{lib.rs, verification.rs, runtime/{batch,stability,exact,fast,reduction}.rs}` (ingress replaced); SDKs |
| Dependencies | Symbolica (coefficient extraction, evaluators, Laurent series); SymJIT; Idenso/Spenso (colour-flow lowering incl. CT, Dirac algebra in kernels); HepKit/FeynKit (covariant model; `IntegralFamily` and overall factors as oracle); NinjaRS (`FastAmplitude`, `TensorNumerator`, `tensor_layout`, `FastMasters`, generic `Amplitude`); oneloopmaster |

### 13.2 New modules

- **Python:** `src/pyamplicol/loops/v2/` with `catalogue.py`, `blocks.py`, `families.py`, `skeletons.py`, `schedules.py`, `colour.py`, `ct_sites.py`, `ct_recursion.py`, `fields.py`, `poly_affine.py`, `reference_executor.py`, `conventions.py`, `assemble.py`, `observables.py`, `artifact.py`, `certify.py`; plus `src/pyamplicol/models/ct_color_flow.py`.
- **Rust:** `rust/crates/rusticol-core/src/loop_v2/` with `plan.rs`, `poly.rs`, `dress.rs`, `trie.rs`, `forward.rs`, `merged.rs`, `seeds.rs`, `closure.rs`, `ct.rs`, `observables.rs`, `rescue.rs`, `retest.rs`, `generate.rs`; `rust/crates/rusticol-core/benches/loop_v2_kernels.rs`; `rust/loop-reduction/src/runtime/{v2_reduce.rs, master_cache.rs}`; `rust/loop-reduction/python/src/v2.rs`.
- **Tools:** `tools/loop_v2_costmodel.py`, `tools/loop_reference/legacy_capture.py`.

### 13.3 Deleted after gate R3 (legacy kept as test-only oracle until then)

Line ranges are [V-s contracts §5].

| Python (`loops/`) | Rust |
|---|---|
| `physical_process.py` `NativeBareProcessProgram` (186-1570) and control builders; `physical_counterterms.py` `NativeModelVirtualProgram` (140-1225) and controls; `graph_basis.py`, `kernels.py`, `color_terms.py`, `recurrence.py` (and `tensor_components.py` if unused); `dual_weights.py` symbolic builders (293-371) and FFT selection (128-290); `family_projectors.py`; `color_orbits.py`; `prepared_process.py`; `portable.py` `PortableLoopPlan` (476-834) and legacy writer branches; `physical.py`; `control_models.py`; the pairwise path of `color_bilinears.py` (50-119); `physical_color.project_physical_color` (289-416); `directed_color_projection.py`, `prepared_correlations.py`, `physical_correlations.py`, `physical_model_correlations.py`, `scalar_incidence.py`; `counterterms.match_counterterms` (756-914) and `vertex_symmetry.py`; per-Born-graph field matching in `wavefunction.py`; `flavor_transport.py` paired programs; `stability.py`, `numerical_lanes.py`, `result_bridge.py`; test-only `model_qcd.py`, parts of `model_gluons.py`; `qcd_source_color.py`, `model_compatibility.py` after a test check | `rust/loop-reduction/src/runtime/schedule.rs`; `loop_execution.rs::evaluate_inner` (830-1705), the whole-batch retry (749-826) and `LoopBackend::schedule`; `loop_runtime.rs` `ModelBinding::Legacy`, `ScalarBinding::Evaluator`, `Schedule`, `ScheduleGroup`, text numerators, `projector_state`, `HelicityProjection`, `RectangularDualWeights`, `DirectedProjection`; `runtime/reduction.rs:32-67`, `runtime/fast.rs:59-114` and `lib.rs` `FastPolynomialReducer` (text numerators); the FFT path of `runtime/dual_weights.rs` |

Also deleted: the 18 `RETAINED_*` injection tests and the per-process timing factories. The model whitelist (`model_catalogue.py`) goes with R6, not with v2. `control_models.py` is deleted only after the O7 tests are ported to fixtures, because the FeynKit oracle uses it.

---

## 14. Validation strategy and oracles

| # | Oracle | What it isolates | When |
|---|---|---|---|
| O1 | **Frozen legacy outputs** (double, AP ≥ 200 bits) for DY, uudd, mixed uudd, tt, qq→gg, gg→gg, uu~→ggg, uu~→dd~g, uu~→e⁻e⁺gg, uu~→dd~ss~, ee→μμ, sgluon, **uu~→uu~, uu~→uu~g**. Covers Born, 6 components × sectors × poles, 19 totals, LC/NLC degrees, directed, spin, resolved, flavour, CT insertion multisets | Everything, end to end | Wave 0 freeze; every gate |
| O2 | **Strong per-family oracle.** Legacy Born-weighted coefficient vectors (capture hooks `physical_process.py:1030, 1362` [V-s]) against v2 polynomials, after the exact affine map from FeynKit `IntegralFamily.find_mappings`, modulo certified-zero families. Error norm: the coefficient max-norm with the coefficients of rank r weighted by √s^r (the raw max-norm mixes mass dimensions) | Numerator, colour, helicity, fermion signs; isolates reduction | S1 and wave 2 |
| O3 | **Weak per-family oracle.** Reduced Ω_f per family | Frame-independent check | When frames differ |
| O4 | **Schedule equality.** F = B = M per family, exact in rational arithmetic (M on LC and certified classes) | Merging and sharing logic | Every change to schedules |
| O5 | **Random-q family oracle** (Design 3 A′). Evaluate the family numerator at random exact q by direct per-skeleton evaluation (kernels with numeric q) and compare with the polynomial | Polynomial algebra, raise tables, layout | Unit level |
| O6 | **CT site closure against CT recursion.** Equal CT amplitudes per (site class, kind, ε); multisets equal legacy `match_counterterms` | Counterterm recursion, symmetry | Wave 2 |
| O7 | **FeynKit structure oracle.** Family sets (modulo logged certified-zero families: gg→gg 48 + 4 = 52 in the control model), signs and symmetry factors from overall factors on gg→gg, gg→ggg, uu~→gg, uu~→dd~, uu~→dd~g, uu~→e⁻e⁺gg, uu~→uu~; Recola tensor-integral sets on 5 processes [M] | Conventions (judge X4) | S1 |
| O8 | **Provider references.** `workspace/reference/runs/*` controls, `tools/loop_reference/*.json`, and `tests/fixtures/loop/required/<process>.json` from the required-references campaign. gg_gggg passed Recola–OpenLoops at max rel 7.3×10⁻¹¹ [V `required/watch.log`] | Physics truth | All gates; mandatory, never env-gated |
| O9 | **IR poles** against Catani from the Born and colour-correlated Borns; renormalised poles | Every process without a reference | Per point, always |
| O10 | **Asymmetric Ward identity** (ε → p on loop and CT sides only, full kernels) | Gauge invariance, UV + R2 completeness | Validation runs |
| O11 | **Precision convergence:** double vs DoubleFloat vs Float on frozen points | Stability, genuine carriers | Wave 2 |
| O12 | **Counts and costs:** families, skeletons and trie nodes equal Appendix B; measured operation counts within ×2 of the recalibrated model | Generator and cost model | G0, G1 |
| O13 | **Fermion signs:** uu~→uu~, uu~→uu~g (identical-flavour pairings), ee→μμ and e⁺e⁻→e⁺e⁻ at NLO EW (lepton lines in the loop) against O1/O8 and FeynKit overall factors | The §3.3 sign rule | S1 (QCD), wave 3 (EW) |
| O14 | **BDK NLC check:** exact-N_c interference of uu~→ggg against hep-ph/9409393 eq. NLOCrossSection, with partial amplitudes from resolved-colour seed columns | NLC degree classification | Wave 2 (WP2.4) |
| O15 | **Re-test calibration:** false-negative rates of relabelling, rescaling, pole and κ triggers against Float(256) on ≥ 10⁴ points per required process | Accuracy claims | Wave 2–3 (WP2.3) |

**Gates**

| Gate | Criterion |
|---|---|
| G0 | Rust segment kernels on real trie shapes (gg→gggg, uu~→e⁻e⁺dd~g); throughput and dispatch measured; recipe recalibrated; required QCD targets within their gates with ×1.3 headroom (§5.4). |
| S1 | First vertical slice (§16.3). |
| G1 | Measured end-to-end throughput on gg→gg, gg→ggg, uu~→ggg, uu~→dd~g, gg→gggg, uu~→e⁻e⁺dd~g, uu~→e⁻e⁺ggg; model recalibrated; providers re-measured in production and protocol modes; gg→gg faster than MadLoop at B = 128; B-vs-M decision recorded. |
| W2 | All 2→2…2→4 controls through Rust v2 equal O1 and O8 in three carriers. CT recursion = site closure. Observables and public API pass. |
| E0 | Before any EW timing work (WP3.4), the v2 generator counts families per mass assignment, skeletons, trie nodes, buckets and NinjaRS reductions for gg→tt̄e⁻e⁺g (EW), with uu~→e⁻e⁺ggg (EW) as a control. The G1-calibrated prediction must be within the gate (≤ 1.24 s) with ×2 headroom; otherwise EW-specific work (species sharing by mass, M for EW classes) is scheduled first. |
| R3 | Required set: ≥ 2 providers per process within their mutual agreement; O9 per point; re-test and rescue rates reported; MASTER_PLAN §5.8 speed gates against production-mode providers (protocol columns also reported). |
| D | Legacy deletion. |

---

## 15. Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | Kernels slower on real tries than the probe (dispatch, gathers, memory); thinnest margin uu~→e⁻e⁺dd~g at 1.7× against Recola production at the target's upper end | Medium / high | G0 in wave 0/1 on real shapes; ≥ 16 lanes; per-node split/fused/register-blocked choice; kernel specialisation (WP4.6) promoted into wave 2 if G0 fails; V+jets in G1. Production margins (Table 1b) are 1.7–17× |
| 2 | Generic colour seeds wrong for multi-quark, U(1) or electroweak classes; orientation errors | Medium / high | O2 exact on every control before 7 legs; certificates for compressions; F = B = M; FeynKit O7 |
| 3 | Born–loop convention seam: phases, i^N, symmetry ½, closed-loop sign, fermion signs, both traversal orientations | Medium / high | §3.3 explicit rules; WP1.8 conventions module reusing the tree engine's sign functions; O2, O7, O13; strong-oracle failures localised per family |
| 4 | NinjaRS heptagons in binary64; rescue cost | Medium / medium | Coordinated re-pin; default per-family re-test with error budget; per-family DoubleFloat recompute; rescue rates measured (WP2.3) |
| 5 | CT completeness and admission for general and electroweak models; CT colour (δδ, d); missing formal continuation | Medium / high for EW | Witness table from divergent families + power counting + certifier; site-closure oracle; LC/NLC refused without continuation; R6/R7 dependencies |
| 6 | Memory at 7 legs (tree arena, seeds, buckets; M accumulators) limits the batch | Low / medium | Chunked buckets; per-family DFS (B); chunked states (M, ≈85 MB/lane); batch sized from the workload |
| 7 | Many-observable cost (directed ×2·N_hist; resolved ×H) | High / medium | Seed-column sharing of trees; schedule 3 with transposed reduction (request 2) after a measured prototype; cost-model policy |
| 8 | Electroweak prerequisites (Feynman gauge in the tree engine, complex-mass scheme, R6) | Certain / high for EW members | Wave 3 work package with its own gates; QCD members unaffected |
| 9 | Python generation time at 7 legs (tree DAG) | Medium / low | Rust builder (WP3.1) |
| 10 | Dependency coordination (re-pin, Symbolica pin for Float, upstreaming the generic tensor numerator) | Medium / medium | WP0.2 with the NinjaRS owner; re-pinned together to the announced mains (NinjaRS aa6a6307, oneloopmaster 8ec256fc, Symbolica 9cdc2949); later bumps only on owner announcements, again in one coordinated rebuild |
| 11 | Three executors (F, B, M) diverge | Low / low | F kept small; O4 in CI |
| 12 | Asymptotics: B's Θ((n−1)!·4ⁿ) seed term grows at n ≥ 8 | Low (required set n ≤ 7) | M (WP2.8) is the planned path and the default for n ≥ 8 |
| 13 | Complex-mass master defects (NinjaRS guidance [V]): one-/two-mass boxes with wide widths wrong, also in OneLOop (fixed in oneloopmaster 8ec256fc, pinned); still open: C0(0,0,0; m_complex,0,m_real) wrong on all routes, generic-engine small-\|p²\| unequal-mass bubbles fail, opposite-mass two-mass boxes near collinear limits ≈10⁻⁸, real negative m² not the −i0 limit | Medium / high for CMS processes | Re-pin done; the CMS controls pass on it in double, DoubleFloat and AP80, zero width included; per-defect regression points in WP2.3; affected master signatures refused or forced to the re-test plus rescue; no silent use |
| 14 | Electroweak family proliferation (gg→tt̄e⁻e⁺g EW: 2.5× QCD families by MadLoop's ratio, heptagons through the lepton line) | Medium / medium | Gate E0 with exact counts before EW timing work; species sharing by mass; M for certified EW classes |
| 15 | Provider baselines drift (MadLoop exports rebuilt with production flags during this revision; Recola cache defect at N = 7 with COLLIER 1.2.9/1.2.10) | Medium / low | One protocol for all providers at G1 and R3; production and protocol columns both reported; provider versions frozen in the report |

---

## 16. Implementation plan

Sizes are in agent-days: S ≤ 3, M 4–8, L 9–18. Every work package owns its files exclusively; shared files are split by function as stated. Dependencies on the assessment's no-regret packages (test harness WP01, release builds WP02, massive-quark CT admission WP08, R6/R7) are external.

### 16.1 Waves

| Wave | Content | Exit gate | Wall time with 4–6 agents [I] |
|---|---|---|---|
| 0 | Oracles, references, NinjaRS coordination, kernel gate | Fixtures frozen; **G0** | ≈1–2 weeks (parallel with wave 1) |
| 1 | Vertical slice in Python (F, B, M) plus the reduction adapter | **S1** | 3–4 weeks |
| 2 | Native Rust core, CT recursion, precision and re-test, colour, observables, API, measurement, M in Rust | **G1**, **W2** | 5–7 weeks |
| 3 | Scale and required set (QCD, then EW after E0), deletion | **E0**, **R3**, **D** | 4–6 weeks |
| 4 | Gated optimisations | Each ≥ 15 % measured or dropped | As needed |

Total ≈ 270 agent-days [I].

**Critical path:** WP1.2 → WP1.3 → WP1.4 → WP1.5 → WP1.8 (S1) → WP2.1 → WP2.7 (G1) → WP3.1 → WP3.3 (R3). WP0.3 (G0) runs in parallel with wave 1 and must finish before WP2.1 fixes the kernel form and lane count.

### 16.2 Work packages

**Wave 0**

- **WP0.1 Oracle freeze and slice references.**
  - *Goal:* freeze O1 and the per-family O2 captures (double, AP 200 bits) for the control set, including uu~→uu~ and uu~→uu~g. Produce three-provider references for uu~→dd~g, uu~→uu~ and uu~→uu~g with the existing required-references toolkit (`tools/loop_reference/{exact_points.py, madloop_export.py, references.py, run_required.sh}` on branch `work/required-references`). Add the shim for the stale `_loop_reduction` gate and make provider comparisons mandatory.
  - *Owns:* `tests/fixtures/loop/legacy/` (new), `tests/fixtures/loop/required/{uux_ddxg,uux_uux,uux_uuxg}.json` (new, coordinated with the references branch), `tools/loop_reference/legacy_capture.py` (new), `tests/unit/test_loop_v2_oracles.py` (new).
  - *Depends:* none.
  - *Accept:* fixtures regenerate bitwise; the providers agree with each other and with the legacy engine within ≤ 10⁻⁹ relative.
  - *Size:* M (6).
- **WP0.2 NinjaRS / oneloopmaster coordination.**
  - *Goal:* coordinated re-pin (per-lane rank, Symbolica 9cdc2949); specify requests 2–4, 7 and 8 of §11.5 with the owner.
  - *Owns:* pins in `Cargo.toml` and lockfiles (one coordinated rebuild); upstream work is owned by the NinjaRS session.
  - *Depends:* owner.
  - *Accept:* batched equals unbatched bitwise for unrelated-family lanes (upstream test); pyAmpliCol loop tests green on the new pins.
  - *Size:* S (2) on the pyAmpliCol side.
  - *Status:* re-pin done on `work/v2-repin`.
    - Pins: NinjaRS aa6a6307 and oneloopmaster 8ec256fc; Symbolica 9cdc2949 for both the native and the Python Community kernel; FeynKit d1904bf2 in the native only (the Community overlay keeps gammaloop 0cfb7dbb9).
    - Build pair: native `loop-native-v2-repin-783aaf47` (digest b71edd24) with Community `community-release-python-9cdc2949` (`dependencies/README.md`).
    - Loop tests are green on that pair. The upstream batched-vs-unbatched test is still requested (§11.5, request 1).
- **WP0.3 Kernel gate G0 (new).**
  - *Goal:* Rust segment kernels generic over T with the generic raise over `ninjars::fast::tensor_layout`: split (dense and structured vertex), fused, and register-blocked fused.
    - Benchmark on the probe's synthetic shapes immediately (seed: `revision/kbench/`), then on real trie shapes exported by WP1.3 for gg→gggg and uu~→e⁻e⁺dd~g.
    - Sweep 8, 16 and 32 lanes, and measure per-node dispatch.
    - Recalibrate the Table 1a recipe; fix the default lane count and kernel-form selection rule.
  - *Owns:* `rust/crates/rusticol-core/benches/loop_v2_kernels.rs` (new), `tools/loop_v2_costmodel.py` (new). It consumes `loop_v2/poly.rs` from WP1.1 when available.
  - *Depends:* none to start; WP1.3 shape export for the real-trie part.
  - *Accept:* gate G0 (§5.4, §14).
  - *Size:* M (4).

**Wave 1: vertical slice**

- **WP1.1 Loop kernel catalogue.**
  - *Goal:* the `loop-segment` (split and fused), `loop-propagator-numerator`, `loop-closure` and `ct-site` kinds, with:
    - both orientations; full Yang–Mills; ghosts; massive Dirac lines;
    - Symbolica coefficient extraction in q; SymJIT and exact evaluators;
    - the Rust generic raise over `ninjars::fast::tensor_layout`;
    - the full-kernel catalogue flag (transverse and full tree-kernel variants, §3.2 item 2).
  - *Owns:* `loops/v2/catalogue.py`; new contract kinds and the variant flag in `models/prepared_catalog.py` (only those additions); the variant hook in `models/compiler_entry.py` (`use_transverse_massless_yang_mills` selection only); `rust/crates/rusticol-core/src/loop_v2/poly.rs`.
  - *Depends:* none.
  - *Accept:*
    - every kernel equals the expanded UFO vertex × propagator numerator at random exact points;
    - the raise order equals `homogeneous_index`;
    - Δ ≤ 1 for loop_sm;
    - transverse and full variants agree on conserved inputs and differ exactly by the p_i·J_i terms otherwise.
  - *Size:* M (8).
- **WP1.2 Tree blocks in loop-request mode (Python builder).**
  - *Goal:* §3.2 items 1–3:
    - replace the three pruning layers (`dag_compiler.py:131-133, 341-378`) in loop mode by the loop-aware reachability test;
    - blocks keyed by the full `CurrentIndex`/`ColorState`;
    - per-class kernel variants;
    - block and Born exports through `loops/v2/blocks.py`; exact Born-order bucketing.
  - *Owns:* loop-request code paths in `generation/dag_compiler.py`; `loops/v2/blocks.py`.
  - *Depends:* WP1.1 variant flag (interface).
  - *Accept:*
    - Born unchanged to ≤ 10⁻¹⁴;
    - block currents equal FeynKit tree sums on gg→gg and uu~→dd~g at random points;
    - every position content requested by the WP1.3 family recursion on gg→ggg, uu~→dd~g and uu~→uu~g is present;
    - tree test suite green.
  - *Size:* M (7).
- **WP1.3 Families and schedules.**
  - *Goal:* the loop-prefix recursion with:
    - species transfer and the §3.3 selection rule;
    - certified-scaleless drop with a log;
    - automorphism weights; structural rank; per-family frames;
    - B tries, F prefix graph and M state graph;
    - buckets, master keys, re-test relabellings;
    - trie-shape export for G0.
  - *Owns:* `loops/v2/families.py`, `skeletons.py`, `schedules.py`.
  - *Depends:* interfaces of WP1.1 and WP1.2 (can start on stubs).
  - *Accept:*
    - O7 family sets modulo certified zeros on the oracle processes (gg→gg: 48 + 4 = 52 in the control model);
    - Recola tensor-integral sets on 5 processes;
    - counts equal Appendix B in both configurations;
    - no FeynKit call outside tests.
  - *Size:* L (11).
- **WP1.4 Colour seeds (generic; FC and LC).**
  - *Goal:* generic κ via colour-flow closure; exact K; P per sector; FC and LC projection; LC face-rule degree certification.
  - *Owns:* `loops/v2/colour.py`.
  - *Depends:* WP1.3 (contents), WP1.2 (Born basis).
  - *Accept:* seed-weighted sums reproduce the legacy Born⊗loop contractions exactly via O2 on gg→gg and uu~→dd~g; LC degrees equal `loops/color_projection.py` on gg→gg.
  - *Size:* M (7).
- **WP1.5 Python reference executor (F, B and M).**
  - *Goal:* exact, AP and f64 execution of F, B and M (M on LC, and on certified FC classes once WP2.4 provides certificates; generic fallback to B); exact operation counters per kernel form for G1; exact affine frame maps.
  - *Owns:* `loops/v2/reference_executor.py`, `loops/v2/poly_affine.py`.
  - *Depends:* WP1.1, WP1.3, WP1.4.
  - *Accept:* O4 F = B = M per family, exact; O5 random-q oracle; operation counts exported for gg→gggg (LC and FC), uu~→e⁻e⁺dd~g and uu~→e⁻e⁺ggg.
  - *Size:* M (8).
- **WP1.6 NinjaRS adapter v2.**
  - *Goal:* direct `TensorNumerator` ingress; (N, R) buckets with families as lanes; the caching `FastMasters` wrapper with the §11.2 key rules (C0/D0 canonicalised, bubbles ordered); generic engine from exponent arrays; re-test entry point; Python binding for the slice.
  - *Owns:* `rust/loop-reduction/src/runtime/v2_reduce.rs`, `runtime/master_cache.rs`, `rust/loop-reduction/python/src/v2.rs`.
  - *Depends:* none.
  - *Accept:*
    - equals the existing `PolynomialPlan` path on legacy family numerators;
    - cached equals uncached bitwise;
    - **gg→tt̄ (unequal-mass top–gluon bubbles) equals legacy and uncached**;
    - hit rate on gg→ggg reported.
  - *Size:* M (6).
- **WP1.7 Counterterms in slice form, and fields.**
  - *Goal:* §6.6 site closure with arity from the model's CT rules, witness admission from divergent families, and §6.3 completeness; scalar δZ fields; component split.
  - *Owns:* `loops/v2/ct_sites.py`, `loops/v2/fields.py`.
  - *Depends:* WP1.1–1.4.
  - *Accept:* insertion multisets equal legacy on gg→gg, uu~→dd~g, uu~→ggg, DY and uudd; UV_local, R2 and field components equal O1; a deliberately removed rule makes generation fail.
  - *Size:* M (7).
- **WP1.8 Conventions, assembly and slice acceptance.**
  - *Goal:* phases, i-factors, normalisation; **the §3.3 fermion-sign rule**, reusing `generation/fermion_ordering.py` and `color/plan_replay.py`; NinjaRS post-processing; 6 components and 19 totals; slice runner.
  - *Owns:* `loops/v2/conventions.py`, `assemble.py`, `tests/unit/test_loop_v2_slice.py`.
  - *Depends:* WP1.5–1.7.
  - *Accept:* gate S1; O13 (QCD part) on uu~→uu~ and uu~→uu~g.
  - *Size:* M (6).

**Wave 2: native production core**

- **WP2.1 Rust executor and plan v4.**
  - *Goal:* plan v4; B and F executors generic over T; SIMD lanes (default from G0, ≥ 16); family threads; bucket streaming; lane compaction; re-test path; new `LoopBackend` evaluation path replacing `evaluate_inner`.
  - *Owns:* `rust/crates/rusticol-core/src/loop_v2/{plan,dress,trie,forward,seeds,closure,retest}.rs`, `loop_runtime.rs` (v4 records), `loop_execution.rs` (new path).
  - *Depends:* S1, G0.
  - *Accept:* Rust equals the Python reference (≤ 10⁻¹³ f64; equal in AP) on the slice processes plus gg→ggg, uu~→ggg, uu~→e⁻e⁺gg and uu~→uu~g; memory per lane reported.
  - *Size:* L (15).
- **WP2.2 Counterterms in the tree recursion.**
  - *Goal:* §6.1:
    - CT vertices with pseudo-order and slots; unary node;
    - `models/ct_color_flow.py` (multi-trace state kind); general-N_c four-gluon R2 asset;
    - chiral and mixing δZ sources;
    - Python and Rust builders.
  - *Owns:* `models/loading.py` (CT lowering path), `models/ct_color_flow.py`, `generation/dag_types.py`, the CT state kind in `generation/dag_color.py`, Rust `recurrence/{construct,layout,color}.rs` (CT parts), `loops/v2/ct_recursion.py`.
  - *Depends:* S1.
  - *Accept:* O6 on every control; components equal O1; 7-leg CT generation < 10 min and < 4 GB.
  - *Size:* L (14).
- **WP2.3 Precision, re-test and rescue.**
  - *Goal:* §10:
    - DD/AP prepared kernels in the tree recurrence; exact seeds; generic NinjaRS (measure the `PolynomialNumerator` compile cost at ranks 6–7);
    - the default per-family re-test (relabelling) and the rescaling variant; error-budget escalation; per-family DoubleFloat recompute;
    - single rescue; single-level fix; per-defect complex-mass master regression points (risk 13);
    - **calibration O15**: false-negative rates against Float(256) on ≥ 10⁴ points per required process.
  - *Owns:* `rust/loop-reduction/src/verification.rs`, `runtime/stability.rs`, `loop_v2/rescue.rs`, the precision generics of `recurrence/direct_runtime.rs` (coordinated with WP2.1).
  - *Depends:* WP2.1.
  - *Accept:* O11; injected-instability tests rescued; `test_loop_physical_states.py:227-250` and `test_loop_stability.py` semantics; rescue and re-test rates and the O15 calibration report.
  - *Size:* L (14).
- **WP2.4 Colour production.**
  - *Goal:* §7, in this order:
    - (1) DDM adjoint-ring and fundamental-trace certificates with loop-level proofs (the FC targets depend on them);
    - (2) the q̄q ring;
    - (3) NLC, the continuation requirement, graded U(1) and Fierz pieces, local LC pruning;
    - (4) the O14 BDK test;
    - (5) seed FFT and cap removal.
  - *Owns:* `loops/v2/colour.py` (continued), `color/adjoint_kernel.py`, U(1) grading in `models/compiler_color_flow.py`, Fierz-grading functions in `generation/dag_color.py` (WP2.2 owns only the new state kind).
  - *Depends:* S1.
  - *Accept:* LC/NLC values and degrees equal O1 on gg→gg, uu~→ggg, DY, uu~→e⁻e⁺gg and tt in three carriers; O14; FFT = direct; pruning counts equal Appendix B; 7-gluon seed tables in < 5 min.
  - *Size:* L (13).
- **WP2.5 Observables.**
  - *Goal:* §7.7, §8.4, §8.5, §9: mixed sectors; directed seed columns; spin sources including the asymmetric Ward mode (full kernels); resolved outputs; flavour members and aliases.
  - *Owns:* `loops/v2/observables.py`, `loop_v2/observables.rs`, `runtime/virtual.py` wiring.
  - *Depends:* WP2.1, WP2.4.
  - *Accept:* O1 directed, spin, resolved and flavour outputs reproduced; `.total()` equals inclusive; O10 on gg→gg and uu~→ggg.
  - *Size:* L (10).
- **WP2.6 Artifact and public API.**
  - *Goal:* §12.
  - *Owns:* `generation/loop_service.py`, `generation/loop_artifact.py`, `rust/crates/rusticol-core/src/loop_artifact.rs`, `api/virtual_results.py` (literal), `loops/v2/artifact.py`.
  - *Depends:* WP2.1.
  - *Accept:* public generate, evaluate and correlated calls on the controls through v2; SDK smoke tests; 6-leg artifact < 20 MB.
  - *Size:* M (7).
- **WP2.7 Measurement gate G1 and live report.**
  - *Goal:*
    - operation-count instrumentation;
    - complex MAC/s and per-stage timings on the G1 set, including uu~→e⁻e⁺dd~g and uu~→e⁻e⁺ggg;
    - providers re-measured in production and protocol modes with one protocol (`revision/ol_modes.py`, `rcl_modes.py` as seeds);
    - model recalibration;
    - B-vs-M decision per process class from WP1.5 counts × measured throughput;
    - decision on kernel specialisation;
    - report update.
  - *Owns:* `docs/performance_reports/loop/*`; `tools/loop_v2_costmodel.py` (from WP0.3, after G0).
  - *Depends:* WP2.1, WP0.3.
  - *Accept:* gate G1.
  - *Size:* M (5).
- **WP2.8 Schedule M in Rust (new; planned).**
  - *Goal:* M executor generic over T: LC generic; FC/NLC for certified classes; chunked states by cut-word suffix; family accumulators streamed into buckets where chunk order allows; generic fallback to B per class.
  - *Owns:* `rust/crates/rusticol-core/src/loop_v2/merged.rs`.
  - *Depends:* WP2.1, WP2.4 (certificates); scheduling priority from the G1 decision.
  - *Accept:* O4 M = B in Rust; measured gain and memory report; adopted per class where ≥ 15 % faster at equal accuracy, and as the n ≥ 8 default.
  - *Size:* L (12).

**Wave 3: required set**

- **WP3.1 Generation at scale.**
  - *Goal:* loop-request mode in the Rust recurrence builder (template catalogue: all-subset blocks with loop-aware reachability, CT = 1); Rust family, trie and M-state generator.
  - *Owns:* `generation/recurrence_projection.py`, `models/recurrence_template.py`, the loop mode of `recurrence/construct.rs`, `loop_v2/generate.rs`.
  - *Depends:* WP2.1, WP2.2.
  - *Accept:* gg→ggggg FC generation < 10 min and < 8 GB; plans equal the Python generator's on the 5–6-leg controls.
  - *Size:* L (14).
- **WP3.2 Electroweak enablement and gate E0.**
  - *Goal:*
    - Feynman-gauge massive vectors, explicit Goldstones, ghost loop lines;
    - complex-mass-scheme integration (R7); generic CT admission (R6); species sharing for vector/Goldstone masses;
    - **gate E0**: exact counts (families per mass assignment, skeletons, trie nodes, buckets, NinjaRS reductions) for gg→tt̄e⁻e⁺g (EW) and uu~→e⁻e⁺ggg (EW), with the G1-calibrated prediction;
    - EW lepton-in-loop sign oracles (O13 EW part).
  - *Owns:* `models/external_evaluation.py`, `models/external.py`, `models/compiler_gauge.py`, `models/recurrence_catalog_builder.py`, `loops/model_cms.py`.
  - *Depends:* R6 and R7 packages; WP1.1; G1 calibration for E0.
  - *Accept:* Feynman = unitary gauge on gauge-invariant trees; ee→μμ, massive μμ, eeuu and the CMS fixtures through v2; gg→tt̄ EW against MadLoop; O13 EW; gate E0.
  - *Size:* L (17).
- **WP3.3 Required set, QCD.**
  - *Goal:*
    - uu~→e⁻e⁺dd~g, uu~→e⁻e⁺ggg, gg→tt̄gg, gg→tt̄e⁻e⁺g (QCD), gg→gggg, uu~→ggggg and gg→ggggg in FC, LC and NLC;
    - O8, O9, O10, O11, O15; re-test and rescue rates;
    - the MASTER_PLAN §5.8 timing campaign with production-mode and protocol-mode provider columns.
  - *Owns:* `tests/integration/test_loop_v2_required.py`, report sections (sequenced after WP2.7).
  - *Depends:* WP2.x, WP3.1, assessment WP08.
  - *Accept:* gate R3 for QCD.
  - *Size:* L (12).
- **WP3.4 Required set, EW.**
  - *Goal:* gg→tt̄e⁻e⁺g (EW) and the EW controls.
  - *Depends:* WP3.2 including gate E0.
  - *Accept:* R3 for EW.
  - *Size:* M (8).
- **WP3.5 Model certifier.**
  - *Goal:* §6.3, using the direct-R2 path for 1PI functions with k ≤ k_max.
  - *Owns:* `loops/v2/certify.py`; quarantined tests of the direct modules.
  - *Depends:* WP1.7.
  - *Accept:* reproduces loop_sm R2 and UV for every admitted function; detects a removed rule.
  - *Size:* M (6).
- **WP3.6 Legacy deletion.**
  - *Goal:* §13.3.
  - *Depends:* gate R3.
  - *Accept:* no legacy imports; tests green; FeynKit oracle tests retained (ported to fixtures).
  - *Size:* M (6).

**Wave 4: gated optimisations**

Each package is kept only if it measures ≥ 15 % at equal accuracy:

- **WP4.1** Parity halving (§8.3), for gg→gggg, gg→ggggg, uu~→ggggg and gg→tt̄gg only.
- **WP4.2** KK canonical pure-gluon blocks (cubic, CT-free, source-free; numerical KK identity gate).
- **WP4.3** M for the 2→6 stretch (eight gluons) and memory streaming refinements.
- **WP4.4** NinjaRS stage 2 (§11.3; only if NinjaRS exceeds 30 %).
- **WP4.5** Schedule 3: F plus transposed reduction for many observables, after a measured prototype of request 2.
- **WP4.6** Kernel specialisation and SIMD tuning (promoted into wave 2 if G0 fails).
- **WP4.7** Integrated-equivalent (de-symmetrised) full colour, opt-in and labelled (§7.6).

Sizes: S–M each; WP4.3 and WP4.5 are L.

### 16.3 First vertical slice (gate S1)

**Scope.**

- gg→gg in FC and LC with **mb = 0** as in the provider references: 24 families = 12 massless + 12 top [M `revision/mb0_all.log`]. The control model with massive c, b and t is used only for O7.
- uu~→dd~g in FC; uu~→uu~ and uu~→uu~g for fermion signs.
- QCD corrections in loop_sm, model R2 and UV, all three Laurent coefficients, double and arbitrary precision.

**Path through the new engine.** No FeynKit diagram generation is used in this path; FeynKit appears only in oracle tests.

1. **Model and kernels** (WP1.1): `prepare_loop_request_model`, then the loop kernel catalogue (gluon, ghost, light and massive quark loop lines; QCD CT site kernels; transverse and full variants).
2. **Trees** (WP1.2): all needed blocks (loop-aware reachability) and the Born per (basis element, helicity), from `GenericDAGCompiler` in loop-request mode.
3. **Families and schedules** (WP1.3): loop-prefix recursion, certified-scaleless drop, families, B tries, F graph, M states (LC).
4. **Colour** (WP1.4): generic seed classes, exact K; FC; LC for gg→gg with `QCDR2Continuation`.
5. **Dressing** (WP1.5): Python reference executor, F, B and M, exact and f64.
6. **Reduction** (WP1.6): NinjaRS v2 adapter, fast engine in f64 and generic engine in AP, cached masters, re-test.
7. **Counterterms and fields** (WP1.7): site closure, then UV_local, R2 and fields.
8. **Assembly** (WP1.8): fermion signs, six components, 19 totals, IR-pole check, per-family error estimate.

**Acceptance.**

| Check | Reference | Tolerance |
|---|---|---|
| gg→gg full result × 3 poles | `tools/loop_reference/gluon_scattering_qcd.json` (MadLoop, Recola, OpenLoops; mb = 0) | Within the providers' mutual agreement (≤ 10⁻⁹ relative) |
| gg→gg components (bare, UV, R2, UV_local = `uv_vertex`, fields = `wf`) × 3 poles | `workspace/reference/runs/gluon-scattering-component-control.json`: a single instrumented MadLoop run ("independent diagnostic component controls"), plus frozen legacy O1 | ≤ 10⁻⁹ relative to the instrumented run, and ≤ 10⁻¹² against legacy in f64 |
| uu~→dd~g, uu~→uu~, uu~→uu~g, full and components | WP0.1 three-provider references (full results); frozen legacy engine (components) | ≤ 10⁻⁹ against providers; ≤ 10⁻¹² against legacy in f64, ≤ 10⁻⁴⁰ in AP |
| Strong per-family oracle O2, all slice processes | Legacy Born-weighted family coefficients after `find_mappings`, modulo certified zeros | ≤ 10⁻¹³ f64; ≤ 10⁻⁵⁰ at 200 bits |
| Schedules O4 | F = B = M (LC) per family | Exact (rational) |
| Structure O7 | FeynKit families (modulo the logged certified zeros), signs, symmetry factors | Exact |
| Fermion signs O13 (QCD) | uu~→uu~, uu~→uu~g against providers and FeynKit overall factors | Exact signs; values as above |
| Counterterms | Insertion multisets equal legacy; completeness assertion passes; removed-rule fixture fails | Exact |
| gg→gg LC | Legacy LC projection and degrees | ≤ 10⁻¹² |
| Poles O9 | Catani operator from Born and colour-correlated Born | ≤ 10⁻¹⁰ relative |
| Precision | double vs AP; re-test estimate brackets the true double-precision error on all slice points | Agreement to the reducer's double accuracy |

**Deliberately excluded from S1:** Rust execution, the CT recursion, NLC, uu~→dd~g at LC (needs graded U(1)), observables beyond inclusive, electroweak, and performance (measured at G0 and G1).

---

## Appendix A. Literature basis and alternatives considered

The user asked for a literature pass to make sure the structure is promising. The sibling reports read the primary sources ([L-s] below). For this document I re-checked, in source:

- van Hameren's complexity statement;
- the OpenLoops 2 pole and performance statements;
- NJet's timing and de-symmetrisation tables;
- Ninja's accuracy caveat;
- BDK's colour sum.

I also searched for 2023–2026 developments [L*] and measured the providers on this host in both modes [V]. No newer program contradicts the structure:

- GoSam-3.0 (arXiv:2507.23549) is about effective field theories and speed;
- arXiv:2502.07060 is a GPU tree-level Berends–Giele code;
- there is still no production SIMD/GPU one-loop code.

| Structure | Evidence | Verdict |
|---|---|---|
| Born-seeded open loops with on-the-fly helicity sums and merging (OpenLoops 2) | Merging ≈2×, plus a further 2–3× from the on-the-fly treatment. Parent–child recycling is lost because the seed carries whole-diagram colour [L 1710.11452 l.1405-1411]. A-posteriori reduction is faster than OFR depending on the process [L l.3836-3846]. Defaults return I-operator poles "for performance reasons"; actual poles need multiple evaluations [L 1907.13071 `OL2.tex:4881-4885, 6455-6457`]. gg→tt̄gg 0.60 s on an i7-4790K [L tab:perfnew]; 0.49 s production on this host [V] | Adopted as schedule B, over BG blocks instead of diagrams; OFR rejected |
| Recursive loop currents keyed by propagator history (Recola) | (ho, hm) keys, canonical cut rules, colour flows [L-s 1211.6316 §2.2; recola2 source]. Generation on this host: 8–63 s for the 7-leg V+jets and tt̄ processes, 2,035 s (gg→gggg), 10,220 s (uu~→ggggg) [V]. Tensor integrals are 2.4–29 % of its time [M]. Production mode is 3.1–5.9× faster than the IR-pole protocol in steady state [V] | Adopted for generation keys and schedule F; its amplitude-level economics only through the transposed reduction |
| Born-weighted per-topology numerators into Ninja (MadLoop5) | Coefficients 36 s/helicity vs Ninja 0.78 s at gg→tt̄ggg [L 1604.01363 App. B]. Ninja tensor layout identical to NinjaRS [V `ninjars:fast/tensor.rs:1-6`]. Loop groups: 1,168 published for gg→tt̄gg; MG5 3.7.0 exports here have 1.6–3.9× our family counts [V] | Interface and checks adopted; per-diagram structure rejected |
| Grouped or cut-dependent numerators (GoSam, Samurai, Ninja) | Cut-dependent integrands are legal [L-s 1006.0710, 1403.1229 §3.1]. Ninja's master cache [L-s 1403.1229 §3.3]. GoSam reduces pinched diagrams with their parent at almost no extra cost [L 1111.2034 `diagen.tex:100-130`]. Ninja's internal tests are not an accuracy estimate; use scaling or rotation tests [L 1403.1229 `main.tex:1642-1651`] | Master cache and per-family re-test adopted; grouping only as gated stage 2 |
| Tensor recursion per primitive and helicity (van Hameren) | O(n⁴2ⁿ) per ordered amplitude and helicity, competitive with unitarity up to n = 10 [L 0905.1005 `introduction.tex:56-99`, `results.tex:142`] | On-the-fly helicity summation removes the extra 2ⁿ from explicit helicity sums |
| Unitarity over BG trees (NGluon, NJet) | n⁶ observed, n⁸ asymptotic [L-s 1011.2900]. FC 7 gluons 171 s, 2u5g 195 s for two evaluations including the scaling test; 1.37 % / 8.63 % quadruple precision at 6 / 7 gluons [L 1209.0100 `njet.tex:1161-1227`]. De-symmetrised (integrated-equivalent) FC: 7 gluons 3.07 s [L tab:timedsymgluons] | Independent cross-check; per-point scaling-test practice adopted as the per-family re-test; de-symmetrised FC as a labelled opt-in |
| HELAC-1Loop | Loop as (n+2)-point tree; R2 as one-insertion trees [L-s 0903.4665 l.790]; colour and helicity sampling | Counterterms as trees and the (n+2)-point sign convention adopted; sampling rejected (exact per point) |
| Colour decompositions | DDM ring, BDK primitives and NLO colour sum, Ita–Ozeren, Reuschle–Weinzierl, Kälin [L/L-s hep-ph/9910563, hep-ph/9409393 l.1748-1772, 1111.4193, 1310.0413, 1712.03539, 1908.02695] | Seed compressions, the LC face rule, the O14 test |
| Leading colour in production (BlackHat, NJet) | LC ≈ 10 % off for Wbb̄ [L-s 1712.05721]; subleading sampled separately [L-s 1312.7140] | Exact LC/NLC by degree; LC and FC−LC as opt-in outputs |
| Forward-limit / Q-cut sewing of (n+2)-point trees | Integrands with linear propagators or loop-momentum relabellings [L* 1509.02169, 1512.02860, 2301.08043] | Rejected: not a fixed-denominator polynomial per family without a canonical cut; with one, it reduces to the Recola construction used here |
| Colour-dressed unitarity; numerical loop integration; loop-tree duality | Colour sampling, own reduction or no master integrals [L-s 0911.1962, 1111.1733] | Rejected |
| GPU / SIMD virtuals | No production one-loop SIMD or GPU evaluation; virtuals become the bottleneck after GPU trees [L-s 2303.18244, 2609.23500]; LoopFest 2024 Kokkos tree and one-loop abstract without details [L*]; GPU tree-level BG [L* 2502.07060] | Data-independent lockstep lanes are a distinguishing lever (≥ 16 lanes, §5.4) |
| Collier cache | Pinched sub-integrals and tensor sub-reductions computed once per event by call position [L-s 1604.06792 §4] | Scalar-master cache only; tensor-reduction sharing has no integrand-reduction analogue except stage 2 |

## Appendix B. Sources of numbers and reproduction

| Numbers | Source |
|---|---|
| Families, skeletons, trie nodes, dressing (structured), rank profile, lifting bound, n = 8 scaling (massive b) | `workspace/tmp/engine-v2/otf/{run_all.log, scaling8.log, schedule_cost.log, lift_bound.log, rank_profile.py}` [M] |
| Same in the providers' configuration (massless b), structured and dense constants; gg→tt̄e⁻e⁺g (QCD) | `revision/{mb0_all.py, mb0_all.log, mb0_all.json, b_dense.py, b_dense.log, ttll_cost.py, ttll_cost_mb*.json}` [M] |
| NinjaRS stage (histogram model) and target times, both configurations | `revision/{ninjars_stage.py, ninjars_stage.log, estimate_table.py, estimate_table.log, estimate_massive_b.py, estimate_massive_b.log}` [M] |
| Design 1 model: F/M ratios, memory of M, merge toy check (1.15×10⁻¹⁵ … 1.37×10⁻¹⁴, re-run by the judges) | `workspace/tmp/engine-v2/cft/{schedules_gluon_n8.log, schedules_general.log, memory_gluon.py, merge_toy_check.py}` [M] |
| LC classes, LC counts (massive b), KK colour check | `workspace/tmp/engine-v2/lcs/{lc_structure_counts.log, lc_kk_block_probe.log, rerun_lc_colour_probe*.log}` [M] |
| NinjaRS stage (mass-aware), master redundancy, groups, lifting probes | `workspace/tmp/engine-v2/rn2/{required_analysis.json, groups.json, master_probe/*.log}`, `lift_probe/` [M] |
| Kernel probe | `revision/kbench/{src/main.rs, kbench_all.log}` [M] |
| FeynKit family check (gg→gg: 52 = 4 × 12 + 4 scaleless) | `revision/fam_gg_classes.py` [V] |
| Provider timings, campaign | `workspace/reference/runs/required/<process>/{madloop-production,openloops,recola}-result.json`, field `rows[*].seconds` [V] |
| Provider timings, production vs protocol on one core | `revision/{ol_modes.py, ol_modes_*.json, ol_*.log, rcl_modes.py, rcl_modes_*.json, rcl_*.log, recola_driver_modes.f90}` [V] |
| MadLoop loop groups | `required/<process>/madloop-export/SubProcesses/P0_*/loop_matrix.f`, `NLOOPGROUPS` [V] |
| Inherited engine stage split | `pyamplicol/docs/performance_reports/loop/data/measurements.json` [V] |
| NinjaRS interfaces | `ninjars:fast/{tensor.rs, tensor_layout.rs, numerator.rs, mod.rs, engine.rs, masters.rs, tensor_jets.rs}`, `ninjars-symbolica/src/polynomial.rs`, `ninjars-cli/src/case_check/harness.rs`, `docs/audits/transpose-probe.md` [V]; `workspace/reports/ninjars-rank-stability-guidance-20261020.md` [V] |

To reproduce:

```sh
source /shared/external/pyLoopAmpliCol/workspace/env.sh
PY=/shared/external/pyLoopAmpliCol/workspace/venv/bin/python
cd /shared/external/pyLoopAmpliCol/workspace/tmp/engine-v2/otf && $PY -I run_all.py
cd ../cft && $PY -I schedules_gluon.py 8 && $PY -I memory_gluon.py 7
cd ../lcs && $PY -I lc_structure_counts.py
cd ../revision && $PY -I mb0_all.py && $PY -I b_dense.py && $PY -I ttll_cost.py mb0 \
  && $PY -I ninjars_stage.py && $PY -I estimate_table.py && $PY -I estimate_massive_b.py
# provider modes (each < 2 min; pinned idle core):
$PY -I ol_modes.py gg_ttxgg 185 && $PY -I rcl_modes.py uux_emepddxg 191
# kernel probe (set CARGO_TARGET_DIR inside the scratch tree; env.sh points it at the shared build dir):
cd kbench && CARGO_TARGET_DIR=$PWD/target RUSTFLAGS="-C target-cpu=native" cargo build --release --offline \
  && taskset -c 186 ./target/release/kbench
```

## Appendix C. Provider re-measurement protocol (revision 2)

- **Binaries, points, settings.** These are the campaign's (`workspace/reference/runs/required/<process>/`):
  - the OpenLoops driver in `openloops-run/` (OpenLoops 2.1.5, loop libraries from the public repository, `loop_optimisation -O0` as in the OpenLoops default configuration);
  - a scratch copy of the Recola driver that adds one `irpoles` command, built against the same Recola 2.3.0 tree (self-reports 2.2.5) with COLLIER 1.2.8 (`revision/recola_driver_modes.f90`).
- **Modes.**
  - OpenLoops protocol: `truepoles 1`, which sets `coli_cache_use = 0` and `polecheck_is = 1` (`init_ui.F90:455-463`) and forces full re-evaluation in the stability rescaling (`stability.F90:479-483`). OpenLoops production: `truepoles 0`.
  - Recola protocol: `compute_ir_poles = 1`, which makes three `compute_amplitude` calls per point (`process_computation_rcl.f90:152-190`), with the campaign's cache setting. Recola production: `compute_ir_poles = 0`, cache on. A third mode, no poles and no cache, separates the two effects:

    | Process | No poles, cache on | No poles, no cache | Poles (protocol) |
    |---|---|---|---|
    | uu~→e⁻e⁺dd~g | 0.035 s | 0.061 s | 0.186 s |
    | uu~→e⁻e⁺ggg | 0.25 s | 0.32–0.36 s | 1.02 s |
- **Timing.** Single pinned idle core of this host; the driver's own timer; point 1 evaluated untimed for initialisation, then points 2, 3, 1. Recola's cached production rows are quoted from the second timed point on.
- **Equality checks.** Finite parts: OpenLoops production = protocol bitwise on all points. Recola cached = uncached to ≤ 1×10⁻¹².
- **Ratios, protocol / production.**
  - OpenLoops: 2.85–3.03 on all six processes measured.
  - Recola: 3.1–5.9 in steady state; 2.2–5.9 including the first timed point, which pays the cache set-up.

---

## Review resolution

Two adversarial reviews of revision 1 were received: **literature** (L1–L9) and **NinjaRS implementation** (N1–N11). Both judged the design "sound with fixes" with no fatal issue. Every issue is resolved below; disagreements are marked **Rebuttal**, with evidence.

### Literature review

| # | Severity | Issue | Resolution | Where |
|---|---|---|---|---|
| L1 | Major | Provider timings were protocol-mode (OpenLoops `truepoles 1`, Recola `compute_ir_poles 1`, partly uncached), so margins were inflated ≈2.5–5× | **Re-measured production mode on this host** for six processes (OpenLoops) and five (Recola), with the campaign's binaries and points [V]. OpenLoops production is 2.85–3.03× faster than protocol, e.g. gg→gggg 2.92–2.96 s (reviewer's estimate: ≈3–4 s). Recola production is 3.1–5.9× faster in steady state, e.g. uu~→e⁻e⁺dd~g 35 ms (reviewer: 41 ms from the harness). Table 1b gives production and protocol columns. Gates use 1.3 × best production. "20–50× / 100–250×" removed. The V+jets processes are treated as speed-critical (G1 includes uu~→e⁻e⁺dd~g and uu~→e⁻e⁺ggg). The baseline choice is decision 7. MadLoop production builds were also re-timed during this revision (lower values quoted). | §1.3, §1.5, §1.6 (7), §5.4, §14 (G1, R3), §15 (1, 15), App. C |
| L2 | Major | Absolute cost model unvalidated; inconsistent range arithmetic; 1–2 G complex MAC/s assumed | One recipe for every row, applied in a script. Both ends now come from the same schedule-B model (structured vs dense kernel constants, 1.7–1.8× apart) instead of B-in-one-model / M-in-another. **Rust kernel probe**: 1.3–1.8 G complex MAC/s per core at 16 lanes for the split, fused and register-blocked forms; 0.4–0.9 G at 8 lanes. **Gate G0** (new WP0.3) measures the kernels on real trie shapes in wave 0/1, before WP2.1. V+jets in G1. All v2 times labelled targets. Recomputed ranges confirm the reviewer's arithmetic: with re-test, mb = 0, gg→gggg 0.17–0.47 s, gg→ggggg 3.4–9.0 s, uu~→ggggg 1.0–3.0 s; with a massive b, 0.25–0.64 / 4.8–12.4 / 1.1–3.2 s. Even at the upper ends every measured row passes its gate; the thinnest is uu~→e⁻e⁺dd~g at 1.7× against Recola production. | §1.3, §5.4, §14 (G0, G1), §16 (WP0.3, WP2.7) |
| L3 | Major | Default accuracy checks weaker than published practice | **Per-family re-test of every family of every point is the default**: relabelling plus loop-momentum shift with cached masters (MadLoop practice, NinjaRS guidance (b)); exact rescaling variant (NJet's scaling test per family). Costed in Table 1a (+0.75–1.0 NinjaRS pass, +12–25 % per point). Error-budget escalation per family. Relaxation only after O15: false-negative rates against Float(256) on ≥ 10⁴ points per required process (WP2.3). The IR-pole check is kept as an additional signal, explicitly not sufficient. | §10.2, §10.4, §14 (O15), WP2.3 |
| L4 | Major | gg→tt̄e⁻e⁺g (QCD and EW) not quantified | **QCD counted** with the generic enumerator: 579 families (629 with a massive b), 3,990 skeletons, 2,840 trie nodes, 9 buckets, hexagons at most, NinjaRS 10–14 ms, target 0.035–0.092 s against OpenLoops production 0.43 s and MadLoop 3.0 s. **EW estimated** [I]: ≈1,450 families (MadLoop's EW/QCD loop-group ratio 2.5), heptagons through the lepton line, target 0.07–0.32 s against best production 0.95–1.0 s. Replaced by exact counts at the new **gate E0** before any EW timing work, with an EW proliferation risk (15 #14). Provider EW/QCD ratios on this host: OpenLoops 2.2–2.6, Recola 1.4–1.7, MadLoop 2.3. | §1.3, §11.4, §14 (E0), §15 (14), WP3.2 |
| L5 | Minor | A_{5;3} misread (claimed relative N_c⁻⁴) | Example deleted. NLC prunes only by exact computed degree. Two of the three A_{5;3} terms enter at P−2 [L hep-ph/9409393 l.1748-1772]. New O14 test against BDK's colour sum. | §7.6, §14 (O14), WP2.4 |
| L6 | Minor | Parity halving over-claimed | Restricted to gg→gggg, gg→ggggg, uu~→ggggg and gg→tt̄gg with real masses and zero widths. Excluded for the three processes with Z couplings and for complex masses. | §1.3, §8.3, WP4.1 |
| L7 | Minor | B-over-M choice rests on one uncalibrated model; "F ≈ 30× B" mislabelled | M is implemented in the Python reference executor (WP1.5) for LC and certified classes, with exact operation counters. G1 decides B vs M per class from counts × G0 throughput. M in Rust is a planned package (WP2.8). The 30× was the Design 1 model's F/M (29.1); corrected. M's memory (≈85 MB/lane, ≈1.4 GB at 16 lanes) is stated as affordable, not a cap. | §1.4, §1.6 (5), §2.2, WP1.5, WP2.7, WP2.8 |
| L8 | Minor | FC headline silently assumes certified colour compression | Stated in §1.3 and §7.2 (generic seeds add 0.4–2.4 s at gg→ggggg). The DDM and trace certificates are the first items of WP2.4. | §1.3, §7.2, WP2.4 |
| L9 | Minor | Inaccurate statements | (a) OFR: its output is acceptable to NinjaRS; rejection now rests on speed, private mathematics, integrand-level validity and family growth (§11.3). (b) MadLoop gg→gggg: 285 s graph generation + 8,354 s export + 20.6 s compilation (§1.5, §4.4). (c) Recola generation: 2,035 s (gg→gggg) and 10,220 s (uu~→ggggg), not "seconds to minutes" (App. A). (d) "B matches the best recursive class" restated precisely against open-loop, tensor and unitarity classes and M (§1.4). (e) Integrated-equivalent de-symmetrised **full** colour added as a labelled opt-in (§7.6, WP4.7). | as listed |

### NinjaRS-implementation review

| # | Severity | Issue | Resolution | Where |
|---|---|---|---|---|
| N1 | Major | Fermion sign of a skeleton never specified | Explicit rule using the tree engine's conventions: the skeleton is treated as an (n+2)-point tree with the cut pair ranked n+1, n+2; the tree engine's Koszul/orientation `merge_sign` at each ring step; per-class pairing parity for coloured lines (`color/plan_replay.py:671-701`); −1 per closed fermion or ghost loop at sewing. Implemented in `conventions.py` by calling `generation/fermion_ordering.py`. Oracles O13: uu~→uu~, uu~→uu~g, ee→μμ and e⁺e⁻→e⁺e⁻ at NLO EW (lepton lines in the loop), added to O1, WP0.1, S1 and WP3.2. The Majorana limit is stated. | §3.3, §14 (O13), WP0.1, WP1.8, WP3.2 |
| N2 | Major | B's asymptotic class overstated | §1.4 states B = Θ((n−1)!·4ⁿ): it beats open-loop and tensor recursions but is asymptotically worse than unitarity (crossover n ≈ 30–50) and M. M (Θ(n!·2ⁿ·n⁴)) is the **planned** asymptotic production path (WP2.8, not optional), adopted per class at G1 and by default for n ≥ 8. B stays the default for n ≤ 7 until G1. The trie's 3.0× sharing gain over unshared dressing is acknowledged; the remaining structural gains are listed in §1.5. | §1.4, §1.5, §2.2, WP2.8 |
| N3 | Major | Master cache returns wrong tensor bubbles under mass-swap canonicalisation | Bubbles (`BubbleRank2/3/4`) are keyed by the ordered tuple; only C0 and D0 are canonicalised, under symmetries that leave their outputs invariant. WP1.6 acceptance adds gg→tt̄ (unequal-mass top–gluon bubbles). | §11.2, WP1.6 |
| N4 | Major (generic models) | Selection rule, counts and oracles disagree | Generation follows the §3.3 rule. Two-position rings with a quartic partner are generated; families are dropped only with a structural scaleless certificate (all masses zero, all invariants structurally zero) and logged. Verified: FeynKit gives 52 gg→gg families in the control model = 4 mass classes × 12 + 4 scaleless bubbles [V]. QCD counts are unchanged; EW/BSM massive quartic rings are kept. O2/O7 compare modulo certified zeros. | §3.3, §4.2, §14 (O2, O7), WP1.3 |
| N5 | Minor | Loop-request mode under-specified | (a) The three pruning layers (`dag_compiler.py:131-133, 341-378`) are replaced in loop mode by one loop-aware reachability test from the family recursion's position table. (b) Blocks are keyed by the full `CurrentIndex` incl. `ColorState` (`sector_id`, `line_groups`, `basis_key`). (c) Agreed: the transverse YM replacement is exact for blocks with transverse sources and is kept there; full kernels for loop segments, CT = 1 currents and Ward/non-transverse sources. WP1.1 owns the full-kernel flag and the selection hook in `compiler_entry.py`. | §3.2, §3.7, WP1.1, WP1.2 |
| N6 | Minor | Slice configuration does not match references | gg→gg slice at mb = 0: 24 families. Components are compared with the single instrumented MadLoop run (and legacy), not with "providers' mutual agreement", which now applies only to full results. | §16.3 |
| N7 | Minor | Rescue trigger 3 not in the timings | It is now the default and is budgeted in every target (+0.75–1.0 R). | §1.3, §10.2 |
| N8 | Minor | Parity-halving claim contradicts its own condition | As L6. | §8.3 |
| N9 | Minor | Cost assumptions not stated | Stated: C-parity folding of closed quark loops (×1.75 FC dressing without it); ghost loops inside the 1.2–1.35 factor; ≥ 16 lanes; certified colour compression. "20–50×" removed. Object counts compared honestly in §1.5 (9,180 skeletons / 7,619 nodes vs 26,975 diagrams at gg→gggg). **Rebuttal (partial):** a separate like-for-like OpenLoops 2 operation model is not built. The speed gate is a measurement, and production-mode OpenLoops timings now exist for every OpenLoops-covered required process (Table 1b), which is a stronger baseline than another unvalidated model. The expected advantage is decomposed instead (cut choice 2.1–2.7× in the same model; blocks; seeds; reduction), and G1 measures it. | §1.3, §1.5, §7.2 |
| N10 | Minor | NinjaRS claims overstated | `tensor_jets` is private (`fast/mod.rs:46`), so (b2) needs a public jets API (request 7). The transpose study is described as a symbolic n3/r3 instruction count, a negative result at one numerator per point, thresholds "not timing predictions" (`transpose-probe.md:21, 45`); schedule 3 needs a measured prototype. DD/AP ingress via `PolynomialNumerator` from a Symbolica expression (`polynomial.rs:258`, compile cost measured in WP2.3); upstreaming `TensorGenericNumerator<T>` (`harness.rs:1290`) is request 8. | §2.4 (18), §7.7, §10.1, §11.3, §11.5 |
| N11 | Minor | Wording and genericity | Collier's cache shares tensor sub-reductions; the master cache only scalar masters (§2.3, §11.3, App. A). OFR reason fixed (§11.3). M memory: ≈85 MB = 65 MB accumulators + ≈20 MB chunked states (§2.2, §5.3). CT arity m ∈ {2, …, k_max}, witness attachments ≤ k_max, and divergent witnesses from R_f ≥ 2N − 4 derived from the model; effective theories refused without declared completeness (§6.2, §6.3, §6.6). Complex-mass master defects added as risk 13. Majorana limit stated (§3.3). | as listed |

**Further corrections made while resolving the reviews** (not raised by either reviewer):

- **NinjaRS stage for gg→tt̄gg.** The histogram model gives 19–27 ms (massless b), not revision 1's "≈50 ms [I]".
- **LC/FC ratio in the providers' configuration.** It is 0.37–0.42 for the gluon processes, not 0.27–0.30, which holds only with a massive b. Both are now stated.
- **MadLoop loop groups.** The literature reviewer's check "families equal MadLoop's loop groups" holds against the published MadLoop 5 table (1,168 vs 1,164). The MG5 3.7.0 exports on this host carry 1.6–3.9× more groups than our canonical families (§11.4).
- **Shared build directory.** `env.sh` points `CARGO_TARGET_DIR` at the shared `workspace/builds/cargo`. A first kbench build landed there and was removed (only the `kbench*` files this revision created); the reproduction recipe now sets the target directory inside the scratch tree.
