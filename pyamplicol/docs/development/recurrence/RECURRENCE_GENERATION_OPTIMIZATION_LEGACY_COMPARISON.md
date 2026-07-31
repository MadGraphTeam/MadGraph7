# Original AmpliCol recurrence-generation comparison

This is a read-only comparison against the pinned original AmpliCol checkout at
revision `79c96cecf2a722e50c3d2030b6894d755f96518a`.  The legacy implementation
is useful as an independent source of construction techniques, but it is not a
semantic oracle for pyAmpliCol and does not define its artifact or runtime ABI.

## Storage and lifetimes

Legacy AmpliCol stores currents and interactions as allocation-heavy Fortran
derived types.  A current owns support, process, color, spin, type, contribution
and value arrays; an interaction owns metadata, couplings and a value buffer
(`dependencies/checkouts/legacy-amplicol/amplitude_QCD.f03:9-53`).  Construction
starts with capacity buffers, grows them by allocate/copy/deallocate, and later
copies the live rows into exact-sized arrays (`amplitude_QCD.f03:889-987`).
Construction-only dictionaries and most ordering/spin metadata are discarded
after pruning (`amplitude_QCD.f03:1867-1886`).

Evaluation does not implement last-use overwrite: it lazily allocates and then
retains every current and interaction value (`amplitude_QCD.f03:2069-2095`).
Generated routines likewise allocate complete current and interaction value
arrays (`amplitude_QCD.f03:3828-3847`).

Directly transferable:

- separate transient construction indexes from the compact persisted program;
- keep support, transition, color and hash lookup data in compact side arrays;
- discard those indexes before lowering and serialization.

Not transferable:

- the array-of-structures growth strategy;
- full resident runtime value arrays;
- fixed-capacity support and color storage.

## Recurrence ordering

Legacy construction is monotonic in external-support size.  For each size it
enumerates all size splits and then the corresponding current pairs
(`amplitude_QCD.f03:129-161`).  Evaluation follows the same stage order:
vertices are evaluated first, then reduced into currents, and propagators are
applied in place (`amplitude_QCD.f03:2099-2150` and `:2454-2516`).  A final
backward walk from amplitude closures removes dead trees
(`amplitude_QCD.f03:4795-4842`).

Generated source groups rows by stage, vertex type and chirality, and groups
current reductions by output type and contribution count
(`amplitude_QCD.f03:3992-4030` and `:4350-4614`).

Directly transferable:

- stage slices and support buckets;
- disjoint-pair enumeration before expensive transition work;
- closure-rooted backward demand;
- deterministic late grouping when it does not alter persisted order.

Persisted row reordering or grouping metadata is not part of this optimization
round because it crosses the recurrence plane/runtime boundary.

## Overwrite and reuse

Legacy modes merge structurally identical currents by appending contribution
IDs.  The all-color path uses a precomputed integer key dictionary and binary
search, whereas other paths linearly scan currents
(`amplitude_QCD.f03:1529-1721`).  Backward compaction rewrites all surviving
IDs exactly (`amplitude_QCD.f03:4843-4988`).

Legacy also contains numerical shortcuts that are not acceptable for
pyAmpliCol:

- a single evaluated point and tolerance `1e-10` merge currents/interactions
  (`amplitude_QCD.f03:3672-3776`);
- ten points and tolerance `1e-8` infer same-flavour decompositions
  (`handling_processes.f03:89-158`);
- ten events and tolerance `1e-12` remove or merge helicities
  (`amplicol_generate.f03:844-901`).

Only exact hash-consing, collision-checked side indexes, late cloning and exact
closure liveness are transferable.  pyAmpliCol's authenticated independent
high-precision evidence, probe counts, digests and failure behavior must remain
unchanged.

## Parameters and couplings

Legacy particles, vertices and coupling capacity are model-specific and
hard-coded (`particles.f03:3-26` and `:195-273`).  Mass and width values are
copied into currents, and two real coupling components are copied into
interactions (`amplitude_QCD.f03:1188-1205` and `:1543-1606`).  Generated code
literalizes masses, widths and couplings (`amplitude_QCD.f03:4143-4165` and
`:4522-4576`), while fixed QCD/EW powers are applied outside the recurrence
(`amplicol_generate.f03:704-722`).

Directly transferable:

- decode immutable transition metadata once;
- retain compact coupling-slot and coupling-order IDs;
- perform exact-zero rejection only when guaranteed by the prepared model
  restriction.

Not transferable:

- literal runtime parameter values;
- fixed two-component couplings;
- hard-coded QCD/EW power extraction and particle/vertex identifiers.

## Batching and vectorization

Legacy generated code emits structure-of-arrays indices for parents, outputs,
momenta and couplings and executes tight loops for homogeneous stage/type/
chirality groups (`amplitude_QCD.f03:3992-4169`).  Current reductions are
similarly grouped by contribution count and current type
(`amplitude_QCD.f03:4350-4614`).

This is compiler-friendly row batching for one phase-space point.  It is not
multi-point batching and contains no explicit SIMD scheme.  It supports
profiling homogeneous late row groups as a future lowering opportunity, but it
does not justify changing pyAmpliCol plane descriptors, scratch ownership,
runtime epilogues or row scheduling.

## Generation shortcuts

The strongest transferable legacy ideas are cheap rejection before
materialization:

- disjoint external-support masks and process-membership intersection
  (`amplitude_QCD.f03:1042-1086`);
- final-leg exclusion and immediate closure feasibility
  (`amplitude_QCD.f03:1072-1083` and `:1192-1197`);
- color, singlet and order rejection before current allocation
  (`amplitude_QCD.f03:1087-1185`);
- exact keyed current lookup and final backward dead-tree filtering.

These directly motivate pyAmpliCol's transient compact support keys,
support-indexed disjoint-pair enumeration, closure/source indexes, hoisted
transition metadata, indexed color-target acceptance, and delayed
clone/hash work.

Model-specific shortcuts that must not be copied include all-gluon reversal
identities without pyAmpliCol proof certificates, the three-quark-line and
5000-color-order limits (`amplitude_QCD.f03:546-637`), fixed-width support
masks, auxiliary tensor/scalar decompositions, and all numerical
current/helicity/same-flavour pruning.

## Read-only build and generation benchmark

The pinned checkout remained unmodified and clean.  Benchmarking used only the
workspace copy at `.artifacts/recurrence-generation-opt/legacy/work`; its probe,
probe source and process-list SHA-256 digests were respectively
`c5e6ba8a276f841fe1d56c941624b364a44dd5e8a9a9b4a84764fd8ef81c911a`,
`5dca3e302cbc5dccd3d03ad9b3e897292dbdd776c8e8c1f54c4f57db912a3653`
and
`ea2f01973842974315aa4deb585dbcacb0b58ba675fbd14797fbf749bae752f0`.
The two source digests match revision
`79c96cecf2a722e50c3d2030b6894d755f96518a` exactly.

The probe built successfully on macOS arm64 with GNU Fortran 14.2.0,
`-ffast-math -O3`, and a watchdog peak of 0.423 GiB:

```bash
repo_root="$(git rev-parse --show-toplevel)"
legacy_root="$repo_root/.artifacts/recurrence-generation-opt/legacy"
(
  cd "$legacy_root/work"
  "$repo_root/.venv/bin/python" \
    "$repo_root/tools/ci/memory_watchdog.py" --limit-gib 30 -- \
    make amplicol_color_probe
)
```

Every process-list and probe child was independently guarded.  For example,
the exact n=9 commands recorded in the logs were:

```bash
repo_root="$(git rev-parse --show-toplevel)"
legacy_root="$repo_root/.artifacts/recurrence-generation-opt/legacy"
case_root="$legacy_root/cases/n9"
(
  cd "$case_root"
  "$repo_root/.venv/bin/python" \
    "$repo_root/tools/ci/memory_watchdog.py" --limit-gib 30 -- \
    "$repo_root/.venv/bin/python" "$legacy_root/work/process_list.py" \
    --serial "d d~ > z g g g g g g g g"
)
(
  cd "$case_root"
  "$repo_root/.venv/bin/python" \
    "$repo_root/tools/ci/memory_watchdog.py" --limit-gib 30 -- \
    "$legacy_root/work/amplicol_color_probe" \
    1 1 1 lc processes.txt momenta.txt \
    -1 1 -1 1 -1 1 -1 1 -1 1 0
)
```

The process string used explicit gluons, rather than inclusive jets, so every
case is the exact partonic ladder `d d~ > Z + (n-1)*g`.  The process-list
generator ran serially.  The probe selected group 1/integral 1 and evaluated
one fixed-helicity point.  Deterministic on-shell final momenta have zero total
three-momentum, conserve energy against two massless incoming beams, contain no
soft legs, and place the 91.188 GeV Z on shell.  Every reported one-point value
was finite.

| n | setup CPU s | process-list wall s | process-list peak RSS GiB | probe wall s | probe peak RSS GiB | watchdog peak guard GiB | currents | vertices | color orders |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.000411 | 0.336 | 0.000 | 0.064 | 0.000 | 0.000 | 7 | 4 | 1 |
| 3 | 0.000323 | 0.333 | 0.000 | 0.063 | 0.000 | 0.000 | 15 | 16 | 2 |
| 4 | 0.001527 | 0.335 | 0.000 | 0.068 | 0.000 | 0.000 | 46 | 76 | 6 |
| 5 | 0.012522 | 0.334 | 0.000 | 0.331 | 0.000 | 0.006 | 184 | 406 | 24 |
| 6 | 0.111564 | 0.334 | 0.000 | 0.337 | 0.000 | 0.005 | 919 | 2,516 | 120 |
| 7 | 1.203470 | 0.336 | 0.000 | 1.420 | 0.382 | 0.382 | 5,512 | 17,959 | 720 |
| 8 | 18.437698 | 1.141 | 0.488 | 19.091 | 3.256 | 3.256 | 38,581 | 145,657 | 5,040 |
| 9 | 594.051455 | 10.630 | 5.125 | 611.212 | 8.770 | 24.146 | 308,644 | 1,324,649 | 40,320 |

`setup CPU s` is the legacy probe's `CPU_TIME` interval around
`amp%init(...)` plus `amp%init_col(...)`; process-list construction, process
file parsing, momenta I/O and evaluation are outside it.  The n=9 setup takes
594.05 seconds (9 minutes 54 seconds), substantially below pyAmpliCol's
historic hour-scale topology case but not negligible.  From n=8 to n=9 the
number of color orders grows by 8x while setup grows by 32.22x.  This supports
the static finding that legacy's compact support rejection and transient
lifetimes are useful, while its color setup itself has a high-order scaling
path that should not be copied.

On macOS the watchdog enforces the maximum of sampled process-tree RSS and
Darwin physical footprint.  The n=9 maxima were 8.770 GiB RSS and 24.146 GiB
physical footprint, safely below but close to the 30 GiB limit.  Sub-250 ms
cases can complete between RSS samples, explaining their displayed `0.000`
RSS; their physical-footprint samples are retained in the machine-readable
record.  Process-list generation is also material at n=9: 10.630 seconds and
5.125 GiB peak RSS.

These are single measurements of one legacy LC flow and cannot be compared as
equivalent work to either pyAmpliCol LC layout.  They do not cover NLC/full
color, multiple topology schedules, numerical certification, artifact
serialization or the current recurrence plane ABI.  The complete guarded
commands, wall times, RSS/footprint values, counters, finite probe values and
input/output paths are in
`.artifacts/recurrence-generation-opt/legacy/benchmark.json`; the concise
human-readable record is
`.artifacts/recurrence-generation-opt/legacy/BENCHMARK.md`.  Per-case process
lists, momenta and logs are under
`.artifacts/recurrence-generation-opt/legacy/cases/`.

## Boundary decision

This audit authorizes only construction-transient indexes, exact-output
enumeration, immutable metadata hoisting and profiling.  It does not authorize
changes to recurrence artifacts, parameter layout, numerical evidence,
`pyamplicol-symjit-plane-application-v1`,
`pyamplicol-recurrence-plane-binding-v2`, scratch/liveness allocation, runtime
epilogues, or runtime scheduling.  Any such proposal belongs in the separate
redesign scouting report and requires review.
