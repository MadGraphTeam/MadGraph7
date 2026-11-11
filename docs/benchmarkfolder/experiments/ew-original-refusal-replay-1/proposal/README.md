# Original-input EW refusal replay: private source proposal

This packet prepares two isolated calls through the original installed
MTIRUST Fortran adapter and TIRust shared library. It has not been compiled,
linked or executed. Root must separately review the source, closed build and
actual runtime providers before admitting a run. There are no input
interventions, tensor coefficient reads or amplitude evaluations.

The closed current-source EW point-2 trace contains one selected loop group 5
per precision: four propagators, rank three, mode 1 (DP) or 4 (genuine QP).
Both native calls returned status 1. Their detailed error strings were not
captured by that trace. This replay exists solely to retain the immediate full
original error. A divergent status remains an incomplete reproduction; a
successful status would not establish independent accuracy.

`extract_inputs.py` authenticates the published runtime packet
`730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`,
the closed stress receipt and derived source/decoder indexes. It checks that
private trace bytes equal the published retained captures. All 41 DP or 82
low-word-first QP words come unchanged from the selected native boundary:
16 shift components, eight complex-mass words, sixteen invariant entries and
the scale. Matrix-to-native shift transposition is checked bit for bit.
The finite output capacity is 35 components times four Laurent slots, matching
the original module's rank-three count. No failed output tensor is inspected.

Initialization preregisters all thirteen plans in the exact generated EW
source order: `(1,0)`, `(2,0/1/2)`, `(3,0/1/2/3)`, `(4,1/2/3)`,
`(5,2/3)`. It then sets the original 268435456-byte cache budget. Evaluation
uses all poles, finite enabled, stability-check pass false and cache false.
Each worker has a fresh context, exactly one tensor call, and finalization.
The original adapter archive is `0c86349a…`, module `d7838e75…` and shared
library `eb3d67fd…`; no private or adaptive library is used.

The audited causal controller is adapted from source seal `ba130c54…`.
Existing compiler/loader prebinding, owned cleanup, shared numerical mutex,
single-thread settings, READY/DONE live provider maps and pre/post word echoes
are retained. The input header explicitly carries rank 3. The error is copied
immediately after the tensor call, before another structural adapter call can
clear it. Native OS exit and returned numerical status are distinct outcomes.

Proposed BUILD: one compile, one link and one actual PT_INTERP `--list` query,
180 seconds total (120 work plus 60 cleanup), no Fortran main or tensor call.
Only the reviewed original prelink inventory plus the newly compiled object
may satisfy actual LOAD dependencies. The standalone runtime provider subset
must be frozen before a separately reviewed RUN.

Proposed RUN: two fresh workers and two calls, maximum 20 seconds per worker,
180 seconds total (120 work plus 60 cleanup), nonexclusive CPU 12, 4 GiB address
space, 16 MiB file cap, no core files and 1 MiB stdout per worker. Both unchanged
requests must reproduce status 1. No ordinary controls or substitutions are
added; the original captured refusals are the two controls. Any admission,
protocol, timeout, cleanup or reproduction failure is retained without retry.

Pure source checks, from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/ew-refusal-replay-proposal-1/extract_inputs.py --verify-only
python3 -B -m unittest discover -s .tirust-build/ew-refusal-replay-proposal-1 -p 'test_*.py' -v
python3 -B .tirust-build/ew-refusal-replay-proposal-1/run.py --reviewed-source-index SOURCE_SEAL
```

The last command defaults to source-only verification. Do not add `--execute`
without root's separate stage review. This packet does not modify production
Fortran, reference repositories, the installed library, thresholds, models,
precision policy, or the CMS correction being developed separately.
