# Closed private DP diagnostics: twelve original refusals preserved

The single admitted replay completed with tool session 90460 exit 0 and
`diagnostic_protocol_passed` after 3.3226160858757794 controller seconds.
This means the diagnostic protocol succeeded: all twelve tensor calls still
returned their original status-1 refusals. It is not an accuracy resolution.

The exact proposal was
`6cc9b0f96e2493fcd3f967d3c39f1ae273ab461ae7917bf220ccb504f1387fb5`.
The six frozen Zgg inputs at point 25, groups 59/61 and methods 1/2/3 were each
evaluated twice in their own fresh context. The boundary status and error bytes
match the earlier EB3 replay exactly. All thirty lifecycle records are retained;
failed tensor values were neither read nor serialized. There was no amplitude
evaluation, precision rescue, QP qualification or retry.

The public workspace metadata now identifies the failed method-choice attempt.
All twelve snapshots report `fallback_used=[true]`, `last_lane_routes=Some([Refused])`,
one expansion row and zero expansion successes, with `precision_rescue=[]`.
The original boundary error reports the primary ill-conditioned Gram refusal;
`last_diagnostics` additionally reports small-Gram expansion nonconvergence
with last relative change `2.5968712461921914e-3` and small Cayley determinant
`2.0110026150849244e-3`. The alternative PV path reports
`modified Cayley determinant cancels (relative rounding inf)`.

These are retained internal diagnostic strings, not amplitude error estimates,
independently measured recurrence operands, or a proof that a guard is too
strict. The reported relative-rounding infinity is not an observed hardware
floating-point exception or an infinite physical integral. No numerical policy,
threshold, mass, source or installed library was changed. The available
metadata does not qualify a repair or justify using a failed tensor result.

The original glibc 2.40 loader and thirteen unchanged providers were used with
the private diagnostic bridge replacing EB3. Its exact provider aliases passed
the loader check; four complete sampled application maps authenticate the
driver plus all fourteen expected providers by path, inode/device and byte
identity. No production EB3 bridge was mapped alongside the private DSO.
The original installed EB3 file remains unchanged. Loader and driver both
returned zero, and the ownership helper recorded no cleanup survivors. A
post-terminal observation found neither recorded process identity still present.
A nonblocking mutex probe succeeded and immediately released the lock; this
describes that probe, not subsequent workers.

The retained started and terminal file modification timestamps are
2026-11-11 00:14:20.824624 UTC and 00:14:24.147635 UTC, respectively.
`retained-tree.json` preserves nanosecond file timestamps. These are file-write
observations; exact process-start UTC was not recorded. Native durations come
from the controller's monotonic clock. The scope remained twelve calls maximum,
4 GiB, nonexclusive CPU 16 and the sixty-second acceptance deadline. Sampled
maps are observational rather than a barrier against transient mappings.

This packet retains every regular run file, all six input copies, raw diagnostic
bytes, exact alias targets, loader output, map samples, command and cleanup
receipts, timestamps and post-terminal identity observations. The private DSO
and executable remain bound to the already published build evidence; binaries
are not duplicated here. The original six missing ZZgg captures remain missing.

The portable verifier reconstructs the frozen protocol and runtime evidence
from retained data alone; it needs no ignored build directory or installed
runtime files and never launches a loader or native calculation:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-replay-closed-1/verify.py
```

Remaining work is to investigate why these expansion and alternative-PV paths
refuse the frozen inputs, using separately reviewed diagnostics if needed.
The current result does not close numerical accuracy, stress, QP or runtime
acceptance items.
