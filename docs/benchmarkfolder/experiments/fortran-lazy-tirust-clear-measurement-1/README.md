# Measured private Fortran improvement: lazy TIRust event clearing

The guarded Fortran adapter removes empty TIRust lifecycle work when backend 9
has never been used. This closed experiment completed with actual controller
exit 0 in 64.154 seconds. Production sources, Rust binaries, dependency pins and
the frozen main campaign remain unchanged. `candidate.patch` is the complete
proposed one-procedure implementation; promotion requires separate review.

For `loop_sm`, `u u~ > z [virt=QCD]`, the uninstrumented full-amplitude CutTools
lane improved by about 0.32% on this host. TIRust's full-amplitude confidence
interval includes equality. These are shared-host observations, not satisfaction
of the agreed precise local performance acceptance.

| Measured scope | Candidate / original | Paired 95% interval |
|---|---:|---:|
| Direct wrapper, TIRust never used | 0.663651 | [0.662873, 0.664430] |
| Direct wrapper, pre-registered context | 1.007618 | [1.002979, 1.012277] |
| Full amplitude, explicit CutTools | 0.996826 | [0.994922, 0.998734] |
| Full amplitude, explicit TIRust | 0.996882 | [0.990792, 1.003011] |

The isolated active wrapper pays about 0.76% for the added Fortran context check;
that is explicitly retained. The never-used wrapper median falls from
405.57 ns to 269.34 ns per clear. These direct timings measure the wrapper and
empty lifecycle boundary, not an estimated tensor-reduction time. Seven pairs
of one million clears per arm/state were taken in alternating order.

The original-only profile was completed before constructing or compiling the
candidate. In 10,101 physical calls (initialization and 10,000 timed amplitudes),
explicit CutTools originally created one unused TIRust context and cleared it
20,202 times. The candidate created and cleared it zero times. Selected TIRust
originally cleared 20,202 times, including one before first plan registration;
the candidate skipped that one initial empty clear and preserved all 20,201
registered-context clears and its one registration. Instrumented brackets
include clock overhead and are explanatory counts/costs, not speedup evidence.

Full-amplitude timing used plain persistent workers, the existing 100 deterministic
on-shell boost points, identical cards/stability policy, all helicities/orders,
single-core nonexclusive CPU 12, seven alternating pairs, and approximately
one-second batches. CutTools used 88 traversals per timed batch and TIRust 115.
All retained timed results used native DP. Observed-vs-plain, original-vs-candidate
and every timed batch matched byte for byte for Born, finite, simple/double
poles, host estimates and statuses. Initialization times are retained separately
and excluded from the warmed interval.

Both actual installed-module lifecycle fixtures passed. They check absent-context
diagnostic preservation, active diagnostic reset, a seeded real cache cleared by
the other shared prefix, public creating-clear compatibility, finalization and
recreation epochs, and IEEE flags/rounding across the initial clear. Active
clearing still calls the existing public API unchanged. No native mutex-poison
failure injection was attempted.

The original adapter's actual historical build log and effective compiler recipe
are included. The private candidate uses the same pinned GNU Fortran wrapper
and exactly the same nonpath flags, `-w -fPIC -ffixed-line-length-132`, with no
optimization, bounds or floating-point-policy change. The new explicit original
process include directory replaces the original implicit current-directory
lookup. `compile-policy-receipt.json` states the limits: original object bytes
were bound before this experiment without claiming individual historical hashes,
and the private controller did not separately snapshot all compiler-injected
environment variables at the compile instant. A post-run empty environment probe
is retained without presenting it as retroactive proof.

The first preparation's missing-identification-card startup remains separately
retained in `../fortran-lazy-tirust-clear-failed-run1-1`. The corrected stage used
the unchanged successful hashed build artifacts, reproduced the original
`Cards/param_card.dat` and `Cards/ident_card.dat` layout, and consumed only the
remaining 895.3899148390628-second budget. It ran once; no numerical failure was
retried. All eight actual warmed worker maps match the fifteen original runtime
providers. No Rust rebuild or repinning occurred.

All 553 regular work artifacts are inventoried. The 539 nonbinary files are
copied exactly; fourteen binary build artifacts are bound by hash/size instead
of being committed. Cards, all raw warmup/measurement outputs, worker command
receipts, provider maps, compiler commands, lifecycle/micro stdout and terminal
records remain available. The small controller source and frozen input proposal
are preserved in `../fortran-lazy-tirust-clear-proposal-3`.

Replay all raw numerical rows, byte comparisons, provider maps, compiler policy,
lifecycle success and paired timing calculations without compilation, foreign
calls or the original workspace:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-lazy-tirust-clear-measurement-1/verify.py
```

The local timing TODO is unchanged: repeat the agreed full multi-process campaign
on Hua-Sheng's machine with the pinned environment. This experiment establishes
a bounded measured Fortran improvement and its tradeoff; it does not qualify
other processes/backends, higher-multiplicity stress, complex masses or precise
per-process timing acceptance.
