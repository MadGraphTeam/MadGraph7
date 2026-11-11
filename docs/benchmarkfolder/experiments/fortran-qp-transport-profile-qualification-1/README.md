# Closed ordinary-QP transport profile

All three fresh workers passed the unchanged100-point learning replay and the
two selected ordinary QCD n3 anchors. Point1 remained219DP; point16 remained323QP
after automatic rescue. Both orders' coefficient/accuracy words, statuses and
observed IEEE states matched. The original four private card paths were retained
and identical. No optimization was applied. These are single-occurrence,
instrumented shared-host observations, not precise performance acceptance.

For point16, the recorded **sequential QP stages1–6** total1.402470612s:

| Scope | Seconds | Fraction of that recorded QP interval sum |
|---|---:|---:|
| Initialization/budget |0.000141391|0.010%|
| Kinematic preprocessing |0.004562806|0.325%|
| Allocation/input decimal encoding |0.023001381|1.640%|
| Native FFI, inclusive |1.294910484|92.331%|
| Output decimal decoding and subsequent tensor mapping |0.074575912|5.317%|
| Contraction |0.005278638|0.376%|

This denominator excludes host/numerator work, cache clearing and probe overhead.
The served-QP path executes these intervals sequentially: stage5 has two separate
pieces (module decode, then wrapper mapping), hence twice as many interval counts
as tensor requests. Other clock scopes must not be added blindly. No empty-clock
correction, repetition confidence interval or cross-precision speed ratio exists.

Before the QP pass, DP complete-module calls total1.212934220s across primary and
independent checks. Five of116 tensor requests in each DP direction returned
native failure status1; all116 QP requests in each direction were valid. The
recorded DP bucket includes the full Fortran module call, while QP bucket4 covers
the native FFI. These are different scopes. Lifecycle cache-clear intervals total0.000059560s
in the lifecycle scope; they are excluded from the sequential QP denominator.

The untimed exact-key trace contains116 eligible primary full-QP requests:
7 prior exact valid keys already produced7 native cache hits, with109 misses,
109 inserted entries and2,053,904 bytes added to the QP store. The independent
check produced116 bypasses and no cache insertions. There is no simulated
Fortran cache or shared-budget change. All7 exact repeats occur below rank4:

| Primary N/rank | Calls | Exact repeats / native hits | Encode ms | Native ms | Decode/map ms |
|---|---:|---:|---:|---:|---:|
|2/1|18|2|0.912|2.383|0.727|
|2/2|6|0|0.364|2.397|0.770|
|3/1|2|0|0.165|0.662|0.105|
|3/2|20|2|1.562|11.451|2.742|
|3/3|27|3|2.234|52.674|8.450|
|4/2|1|0|0.146|2.042|0.160|
|4/3|21|0|2.420|78.761|8.203|
|4/4|9|0|1.118|167.137|5.893|
|5/4|12|0|2.070|327.229|10.205|

Per-topology clocks include both hits and misses. Repeat-specific elapsed costs
were not measured, so the table cannot estimate savings from an earlier lookup.
This fixture does not show Fortran decimal I/O dominating the recorded QP work.
A future bulk-I/O or cache proposal still needs exact wire/error/IEEE/cache
semantics tests and measured benefit; none is implemented or claimed here.
Point1 has232 valid DP tensor requests, zero QP eligibility and zero QP repeats.

Actual anchor pre/post observations use nearest rounding; the five observed
flags and halting settings are false in all modes. MadLoop may clear flags inside
its call: these observations do not prove that no internal operation raised a
flag. True→true sticky states would likewise not identify new exceptions.

## Evidence and execution lineage

- `science/` preserves the original17-payload proposal049c26fb… unchanged.
- `controller1/` and `failure-build1/` retain the rejected unselected compound
  dry-run prerequisite; no Fortran compilation occurred in that attempt.
- `controller2/` and `build2/` retain the successful five-unit build,61 recorded
  commands, one link, one actual PT_INTERP provider query,56 admitted LOADs and
 23 unique MTIRUST definitions. No Fortran main ran during BUILD.
- `failure-mode0/` retains the private three-versus-four card-inventory failure
  before any worker/model/physics call; it is not relabeled as passed.
- `controller3/` pins the RUN-only alias correction and unchanged BUILD2 reuse.
- `mode-0/`, `mode-1/`, `mode-2/` retain complete raw learning/anchor/IEEE/map and
  timer/counter/shadow evidence, actual terminal0 and observed-descendant cleanup
  with no survivors. Every mode used a new private clone and full learning.

The retained current scientific source is0fcd2cd873, exporterafb6e0be…,
numerical3c13ad88…, native DSOeb3d67fd…, and BUILD2 executableac465e7c… .
The frozen9228 baseline is reference ancestry; this experiment did not execute a
new baseline comparison. All native stages used nonexclusiveCPU12, the common
mutex,300s caps with60s cleanup reserve,16GiB address-space/512MiB file/core0
limits. No source/card/library/policy changes, Rust rebuild or retry in an old
namespace occurred.

This portable packet contains UTF-8 source/raw/JSON records, plus the text link
map compressed with its decoded-byte hash. It contains no ELF, object, module or
archive binary. Absolute historical paths are provenance strings. The pure
verifier reads only indexed packet bytes, replays copied frozen parsers and
checks joins; it does not reload historical executables or reread host libraries.
Recorded provider/source/build hashes describe the closed execution, not a new
live-host verification. The ldd-only foreign interpreter alias is distinguished
from the actual PT_INTERP/providers and observed maps.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1/verify.py
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1/analyze.py
```

`analysis.json` is reproducible from the parity-qualified retained records.
Root review of this packet remains separate from its construction. Precise
performance timings remain Hua-Sheng's local TODO; no <=1.05 gate is claimed.
