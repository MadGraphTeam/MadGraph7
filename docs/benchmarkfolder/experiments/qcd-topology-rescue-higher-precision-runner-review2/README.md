# Bounded private512 replay controller — source only

This controller implements the frozen proposal `51f45d9e6f2c13227dc504fa284cba967f947db0f073dfb18bb9bf924c28c298`. It has not been executed. It requires independent source review and explicit root execution clearance. It performs no build or relink and must not modify any production/campaign artifact or the frozen256/NinjaRS controls.

The one proposed process uses the unchanged topology-c executable and parameter/point files, with private copies of the exact original backend9 resources. Its only runtime-library substitution is the existing512-bit DSO. Working precision and its shared guard change together; all other settings, including cache policy, remain unchanged. Every source, executable, DSO, card, resource, historical comparison and controller file is bound before launch and checked again afterward.

Admission uses `LOCK_EX|LOCK_NB` on the shared numerical lock **before any native process or one-shot attempt is created**. If busy, the controller records a non-native admission failure and exits75. It neither waits indefinitely nor retries. A later explicitly invoked admission may try again only while no attempt exists. Once `attempt1` exists, every success, timeout, drift or other failure consumes the sole native attempt. It never automatically reruns the process.

The supervisor preserves the reviewed spawn ownership protection: Python interruption handlers defer exceptions until `Popen` has returned an owned handle, then restore the handlers and raise inside the process-group cleanup scope. It distinguishes attempted from confirmed spawning. The parent's brief handler-installation/removal transitions block the handled signals atomically, then restore its original mask before any child is spawned. Native signal/trap policy is unchanged.

Linux `waitid(WNOWAIT)` observes completion without reaping the leader. Its reserved PID keeps the process-group identity owned until final cleanup. Both normal completion and failure use a signal-deferred group kill/reap/ownership update, so repeated handled signals cannot unlock a live child or leave a recycled PID eligible for later signaling. Default parent `SIGCHLD` disposition is required; an incompatible disposition fails before spawning. Post-exit hashing sits outside active-child cleanup. The read-only `ldd` helper uses the same ownership discipline. The numerical process has CPU11, one thread and a180-second monotonic deadline, including mapping capture and exit observation.

The controller first verifies `ldd` against the declared runtime map. During the actual process it reads `/proc/PID/maps` and leader CPU affinity, requires the unchanged executable and exactly the private bridge, and rejects unexpected or deleted ELF mappings. It persists every successfully read raw maps snapshot before interpreting it, including failed snapshots. The successful mapping/affinity evidence is one snapshot, not continuous monitoring. Hashes of these actual paths are checked before launch and again after process exit, keeping large file reads outside the active deadline loop. The record distinguishes native-process wall time from supervision/post-exit hash overhead. No preload, loader-audit hook or numerical observer change is introduced.

Successful protocol completion requires exit0, complete actual mapping capture, zero observer faults, all ordinary rows/raw words/status/IEEE parity, and unchanged protected files. Stress host codes and coefficients may change and are retained without physical acceptance. The controller records all actual CT4/5 coverage; exact stress input/handoff/mass/matrix comparisons and Laurent adjudication remain offline gates. If mapping capture fails before normal exit, the raw failure is retained and the process is stopped; that is not a reason to retry.

Outputs belong in the separate `qcd-topology-rescue-higher-precision-native` namespace, with work/resources under ignored `.tirust-build/qcd-topology-rescue-higher-precision/attempt1`. The exact source inventory therefore stays immutable while execution evidence accumulates elsewhere.

Read-only source verification:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner/verify_source.py
```

Proposed command **only after execution clearance**:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner/controller.py
```

No finite/single-pole truth, production promotion or performance qualification follows from a completed run.

The unexecuted first review candidate is retained unchanged in sibling `qcd-topology-rescue-higher-precision-runner-review1`, with source index `862dba591e70367d0a4b50d836b9f232c9ec1ea6027edb7c88211af67baebce4`. Source review identified early raw-map loss, potential signaling after reaping, repeated-signal cleanup interruption, and imprecise spawn-record semantics. This revision addresses those findings before any numerical execution.
