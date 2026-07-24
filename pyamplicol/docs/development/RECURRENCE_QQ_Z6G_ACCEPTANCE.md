# Recurrence `qq -> Z+6g` LC Acceptance

## Scope

This record covers the Direct-Arena recurrence implementation for
`u u~ > Z g g g g g g` on macOS arm64. It compares the two independently
generated LC layouts:

- `topology-replay`: one flow selected at runtime, summed over retained
  helicities;
- `all-flow-union`: one helicity selected at runtime, summed over all retained
  physical flows.

The prepared-model build is amortized and excluded from process-generation
timings. Both prepared bundles use portable SymJIT optimization level 2.

The measurements below were refreshed from exact committed checkpoint
`585456ed1726c43eef3ce35c7a126c17730e8a0d`. The loaded release extension
and source tree had the same native-input digest,
`a0d05eafc7236a44bc22de1a68c0f1f3839166e99668be4c60551f75a2399ec0`.
All substantial generation and profiling commands ran under the 30 GiB
watchdog.

## Runtime Results

The wall headline comes from unprofiled native Rusticol repetitions after
caller-side packing. The evaluator value is paired attribution over the same
batch and repetition contract.

| Model or reference | Layout/workload | Batch | Wall (us/pt) | Recurrence schedule (us/pt) | Relative to AmpliCol |
|---|---|---:|---:|---:|---:|
| Original AmpliCol | mode 1, selected flow/helicity sum | 100,000 points | 39.401 total | 38.945 amplitude | 1.000 |
| Built-in SM recurrence | topology replay | 128 | 37.690 | 36.839 | 0.957 |
| Built-in SM recurrence | topology replay | 1024 | 37.398 | 36.538 | 0.949 |
| UFO-SM recurrence | topology replay | 128 | 37.986 | 37.568 | 0.964 |
| UFO-SM recurrence | topology replay | 1024 | 37.879 | 37.041 | 0.961 |
| Original AmpliCol | dynamic mode 2, all flows/single helicity | direct probe | 312.385 total | 306.148 amplitude | 1.000 |
| Built-in SM recurrence | all-flow union | 128 | 335.793 | 331.977 | 1.075 |
| Built-in SM recurrence | all-flow union | 1024 | 325.772 | 319.992 | 1.043 |
| UFO-SM recurrence | all-flow union | 128 | 365.471 | 365.922 | 1.170 |
| UFO-SM recurrence | all-flow union | 1024 | 365.043 | 355.412 | 1.169 |

Both model implementations pass the `1.20x` runtime gate for both layouts.
Prepared contribution bindings now carry an authenticated parent permutation,
resolved once while lowering the schedule. This permits built-in and UFO-SM
catalogs with different canonical parent orders to select the same exact
Direct-Arena intrinsic without adding work to the runtime loop.

For context, the performance-report library probe evaluates its all-flow
workload through multiple generated mode-1 modules rather than AmpliCol's
dynamic mode-2 recurrence. That distinct report workload must not be confused
with the direct mode-2 reference above.

The all-flow-union measurements use the nonzero alternating runtime helicity
`h:-1,+1,-1,+1,-1,+1,-1,+1,-1`. The default first helicity ordinal is a
structural zero for this process and was explicitly rejected as a performance
acceptance workload.

## Generation Results

`Phase total` is model-bundle loading, process expansion, and recurrence
construction. `Outer wall` additionally includes artifact publication and the
ten-sample post-build validation used by this developer harness.

| Model | Layout | Construction (s) | Phase total (s) | Outer wall (s) | Peak RSS (GiB) | Artifact (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Built-in SM | topology replay | 1.492 | 2.810 | 6.051 | 0.360 | 9.76 |
| UFO-SM | topology replay | 1.664 | 3.396 | 7.634 | 0.412 | 17.35 |
| Built-in SM | all-flow union | 3.412 | 4.692 | 25.246 | 0.369 | 10.22 |
| UFO-SM | all-flow union | 3.643 | 5.373 | 24.765 | 0.408 | 17.82 |

The preserved same-host AmpliCol mode-1 measurement required approximately
`3.047 s` for process-specific recurrence emission and library compilation
once the fixed generator executable was available. Its complete cold command,
including the fixed Fortran generator build, took `22.744 s`. The report's
shared generated-library setup for the corresponding Z ladder is about
`10.58 s`.

The recurrence timings above exclude prepared-model creation. In particular,
there is no process-specific SymJIT application construction or JIT
compilation in recurrence generation. For reference, source-freeze prepared
model creation took `4.592 s` for the built-in SM (61 kernels) and `6.809 s`
for the UFO-SM (426 kernels); this amortized work is excluded from every row
above.

## Structural Results

### Topology replay

Built-in SM and UFO-SM have the same schedule:

| Quantity | Count |
|---|---:|
| Source rows | 19 |
| Currents | 1,425 |
| Contribution attachments | 8,338 |
| Finalizations | 858 |
| Closure terms/amplitude destinations | 384 |
| Physical replay targets | 720 |
| Retained helicities | 768 |
| Structural-zero representatives | 384 |
| Packed input/output/scatter bytes | 0 |

### All-flow union

Built-in SM and UFO-SM have the same schedule:

| Quantity | Count |
|---|---:|
| Source rows | 9 |
| Currents | 5,512 |
| Contribution attachments | 33,439 |
| Finalizations | 4,168 |
| Closure terms/amplitude destinations | 720 |
| Retained helicities | 768 |
| Physical flows | 720 |
| Arena components | 17,138 |
| Semantic components | 23,278 |
| Arena component reuse | 6,140 |
| Packed input/output/scatter bytes | 0 |

The union schedule now matches the corresponding AmpliCol mode-2 state and
attachment counts exactly: 5,512 currents and 33,439
current-to-interaction attachments. AmpliCol also reports 17,959 unique
interaction records before those attachments.

## Numerical Validation

- Every generated artifact passed ten post-build validation samples.
- Optimized totals reproduce their same-artifact resolved sums exactly for
  both workloads.
- The selected `(flow, helicity)` component agrees across topology replay and
  all-flow union.
- Built-in and UFO union artifacts expose identical 720-flow and 768-helicity
  axes.
- At the shared nonzero-helicity point, all 720 built-in/UFO union components
  agree with maximum absolute difference `8.50e-33`.
- For one selected flow, all 768 built-in/UFO topology components agree with
  maximum absolute difference `3.87e-34`.
- The selected `(flow, helicity)` component agrees across the two layouts with
  maximum absolute difference `4.70e-38`.
- The topology and union components agree with a freshly generated compiled
  reference with maximum absolute differences `1.18e-36` and `3.35e-35`,
  respectively.
- All comparisons pass `rtol=1e-12`, `atol=1e-15`.

Pure-adjoint reflection aliases are finalized only after the complete fan-in
of every current has been observed. A late missing or conflicting proof now
retains both orientations as exact residual states; only reciprocal,
stage-complete proofs remove the noncanonical orientation.

## Acceptance Status

| Gate | Status |
|---|---|
| Built-in topology correctness and runtime | PASS |
| Built-in all-flow-union correctness and runtime | PASS |
| UFO-SM topology correctness and runtime | PASS |
| UFO all-flow-union correctness and runtime | PASS |
| Built-in/UFO structural and component parity | PASS |
| No `GenericDAG` or process-specific evaluator compilation | PASS |
| Direct-Arena zero packing/scatter contract | PASS |
| Independent architecture-audit closure | PASS |
| Focused Python/Rust/native-allocation suite | PASS |
| C++/ASM Direct-Arena backend parity | PASS |

The exact-checkpoint acceptance includes 251 focused Python unit tests, 22
public recurrence integration tests, 202 Rust recurrence tests, and the guarded
`p p > j j j j` process-set test. The latter produces 11 bindings backed by
8 schedules and 3 exact aliases. Genuine topology-replay and all-flow-union
`Z+6g` artifacts perform zero heap allocations over repeated warmed native
`evaluate_f64_into` calls.

The independent post-implementation audit accepted checkpoint
`585456ed1726c43eef3ce35c7a126c17730e8a0d`. It found no remaining LC
correctness, architecture, performance, or allocation blocker. The missing
native ordered-forest inspector is recorded as observability debt; production
state identity preserves those ordered forests exactly.

Fresh built-in/UFO-SM `Z+2g` checks also exercise the same Direct-Arena
callable contract for JIT, C++, and ASM in both LC layouts. Across all 96
resolved components, the worst C++/ASM-to-JIT difference is `2.03e-20`
absolute and `1.77e-13` relative. Repeating the comparison after setting the
Z mass to `100` gives the same bound. The C++ and ASM checks establish backend
parity without claiming a separate `Z+6g` performance result for those scalar
backends.
