# One bounded original/disabled/enabled replay

This is a source proposal for a separately authorized numerical invocation. The
frozen source preparation and successful CPU-12 build already exist; this
controller has not launched an amplitude. It never builds, repairs, retries, or
changes the frozen 19-file source package.

`run.py plan` reads and verifies the exact preparation (`c7dba79d…`), completed
build manifest (`40dedfe9…`), source/object/card inputs, executables and runtime
libraries. `run.py execute --self-sha256 REVIEWED_SHA` is the distinct native
stage. A root message naming that hash and these resources is the authorization;
the hash argument is an identity check, not an approval file or self-permission.

Execution acquires the shared numerical flock without waiting. A busy lock exits
75 before creating an attempt. It then reserves one fresh directory at
`.tirust-build/qcd-n4-cache-occupancy/attempt1/runtime-1/attempt1`, uses CPU 12 and
one computational thread, and runs original, disabled, enabled serially under
the same lock. Each arm has 180 seconds and the entire operation has 600 seconds.
There are exactly three owned workers and no retries. All arms use the unchanged
installed eb3d DSO; disabled/enabled share the same observed executable.

Each arm gets a fresh exact point file, Cards and MadLoop5_resources. No root
parameter card is created: the original LHA fallback must find Cards/param_card.dat
and Cards/ident_card.dat. HelFilter/LoopFilter outputs start absent. The frozen
driver makes bootstrap 0 and public call 1 at the same registered point, performs
normal MadLoop exit, writes its records and exits on its own. Internal numerical
initialization remains whatever the unchanged host performs. There is no
READY/STOP protocol, extra warmup, further point or timing loop.

The controller imports the exact pinned `6a0512b6…` OwnedChild and Deadline
implementation directly. Its local monitoring code is an adaptation, not a
claim of identical controller source. The owned session leader remains unreaped
until group cleanup, preventing process-ID reuse during cleanup. Every exception
terminates and reaps only that owned group; retained metadata errors do not
replace the original exception. Mandatory cleanup and failure preservation run
even after the work deadline. No unrelated process is signalled.

Live /proc maps must expose exactly the declared ELF closure and executable,
including one eb3d bridge. The first complete mapping is hashed and retained;
subsequent samples reject unexpected libraries or changed inode identities.
Incomplete loader startup can be sampled again, but an exit without a complete
live capture fails qualification without rerunning. Task affinity and absence of
child processes are checked while alive; the number of OS threads is recorded,
not restricted to one. A polling observation is not proof against an arbitrarily
short mapping or thread change between samples.

Stdout is capped at 16 MiB, public records at 64 KiB, occupancy at 80 MiB, and the
entire per-arm file inventory at 128 MiB/128 files. Raw streams, first/last live
maps, generated resources, process records and failures remain in the attempt.
Original inputs, executable and library hashes are checked before and after;
initial private copied resources must also remain unchanged. The pinned analyzer
requires exact input-before/input-after, answers, accuracies, public statuses and
all entry/exit IEEE states across the three arms, and validates the integer
occupancy journal. The analysis is retained before requiring its second-call
accepted-status indicator: matching failed or bootstrap-only calls cannot qualify
point 1, and nonacceptance never causes another call. No performance or
internal-precision claim follows from this diagnostic.

After source review, use the repository cwd and the recorded environment:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/qcd-n4-cache-occupancy/runtime-1/run.py plan
# Only after the separate root instruction:
python3 -I -B docs/benchmarkfolder/experiments/qcd-n4-cache-occupancy/runtime-1/run.py execute --self-sha256 REVIEWED_SHA
```
