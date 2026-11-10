# Proposed stopped-operand diagnostic after the Cayley guard

This is a source-only proposal. No target has been loaded, attached, executed
or continued under the debugger. Root must review and admit `run` separately.
The closed physical capsule remains immutable at index `715f436c…`.

Use the existing GDB 17.2 at the exact Nix path in `plan.json`; its target-free
capability receipts are retained in `capability/`. It embeds Python 3.14.7 and
supports the required signal, breakpoint and register APIs. Actual ptrace
permission, register union field access and the target handshake are untested.
No new executable, DSO, patch, installation or build is proposed.

Run exactly two diagnostic workers, with the unchanged checked executable,
the existing private guarded runtime and the exact checked/forced-QP fixtures.
Both use their original trapped policy. A stopped fork precedes target exec:
the wrapper becomes GDB, which attaches to its own stopped child. This avoids
a sibling ptrace relationship and preserves the owned process group. GDB catches
the target exec, reaches `main`, verifies the complete target ELF mapping and
CPU 12, and writes `mapping-ready`. The Fortran gate is inside the program after
its argument reads and control-log open (`driver.f90:23–29,106–119`); neither
frozen target environment has `LD_PRELOAD`. Acknowledgment at main therefore
precedes this gate. The frozen link map places main at `0x13d5d2`.
The retained startup loader listing already resolves exactly these 15 runtime
libraries without running the program. The proposal still fails before
acknowledgment if that complete mapping is unavailable at main. Its breakpoint
uses the explicit entry address to avoid named-function prologue skipping.

Continue once from main to the first SIGFPE. Record signal subtype, RIP, EFLAGS,
MXCSR, both 128-bit XMM0/XMM1 registers, x87 status/control, instructions,
backtrace and maps. Stop before the target signal handler. Do not resume after
SIGFPE, call a target function, change an FPE flag, change rounding, or evaluate
a new amplitude point. Explicit GDB function calls are disabled. Kill the
inferior and exit. The expected prior sites are Gram `0x11fcecb` and MPFR
dilogarithm `0x2523cc3`; an unexpected site is retained for review.

The outer controller is a Linux child subreaper. It retains the common native
lock through debugger-group termination and reaping, then kills/reaps any
remaining adopted child while its PID is still reserved by parent ownership.
The existing supervisor supplies atomic signal handling and owned group cleanup,
including SIGALRM. A failed ptrace or unsupported register expression is a
retained failed diagnostic with no retry. This is not a new physical
qualification, and stopping/killing after SIGFPE cannot qualify trap completion.

Bounds are CPU 12 without reservation, 4 GiB address space per inherited
process, 120 seconds per worker and 300 seconds for the aggregate including
verification and publication. No automatic retries or extra control workers.
As in the existing supervisor, mandatory kill/reaping may extend cleanup beyond
the deadline for a kernel-stalled process. Such late cleanup is administrative
and cannot publish successful capture qualification.
All input hashes must remain unchanged. The current-source campaign is not
accessed. Actual mapping acknowledgment, GDB child ownership and final cleanup
must be checked in the admitted runtime receipt rather than assumed from the
target-free capability test.

Read/hash verification only:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/post-cayley-sigfpe-register-proposal-1/controller.py verify
```

After separate root admission only, the exact proposed command is:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/post-cayley-sigfpe-register-proposal-1/controller.py run
```

The stopped operands can distinguish a zero denominator from ratio overflow and
identify the actual MXCSR rounding/control state. They cannot establish
independent numerical accuracy, stability acceptance, general precision usage
or performance. This proposal does not authorize any source fix or rebuild.
