# Closed private diagnostic build failure

The admitted first build terminated with tool exit 1, Cargo exit 101, after
518.865636 seconds. Symbolica's rustc coordinator failed to spawn a work
thread (`EAGAIN`, resource temporarily unavailable); later oneloop compilation
reported `rustc-LLVM ERROR: out of memory` and aborted. The complete build log
and original terminal receipt are retained. The private bridge was not built,
loaded or replayed: there were zero tensor evaluations and no private DSO,
C object or driver executable. The source proposal remains sealed as
`755a2d95c1478508ee41c69f11174339440210458e97e4e1eb259de912f3c6e0`.

Post-failure inventories confirm all staged source and Cargo source-cache
bytes remained identical. Original source/tool/runtime pins still matched
when observed after terminal. The 1130 partial compiler-output file hashes
are retained; these are not qualified compiled artifacts or numerical evidence.

The process/memory facts were read after terminal. The session and user
cgroups reported no local OOM or pids-max events and had no memory/pid maximum;
the user.slice ancestor has historical events that cannot be attributed to
this build. RLIMIT_NPROC was 4642218; every compiler child had the admitted
4 GiB RLIMIT_AS. The node reported roughly 238 GiB MemAvailable; its memory
facts describe the host, not a reserved budget. The explicit allocation
failure is consistent with the compiler address-space cap. The original
Symbolica thread-spawn failure has no failure-time VmSize sample, so its
precise cause is unproven. Host-wide memory pressure or other allocation
constraints cannot be excluded from these later observations.

The sourced CXX selection, wrapper/default compiler hashes and version are
separate post-build observations. They do not reconstruct a complete build
environment or prove that this C++ compiler was actually invoked. Compiler
flags visible in the retained rustc command already include release opt-level
3 and codegen-units 1. No dependency, precision, guard or numerical policy
was changed, and this packet does not authorize another build.

Run the portable check from the required checkout:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-private-build-failed-1/verify.py
```

This replays the retained failure and integrity comparisons using Python only.
It does not compile, load a library, or run a tensor calculation.
