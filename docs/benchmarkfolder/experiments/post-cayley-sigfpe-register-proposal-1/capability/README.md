# Installed GDB: target-free capability inspection

GDB 17.2 is installed at the explicit Nix store path in `execution.json`,
although `gdb` is absent from the inspection PATH. Its SHA256 is
`1797464c05f14ff1cc2b5769eeb14399b32998dfc96129786d85cd90f4ff7528`.
The before/after hashes agree. Exact argv, working directory, stdout/stderr
hashes, reaping and elapsed times are retained for three commands: version,
configuration and one target-free batch session.

The debugger embeds Python 3.14.7. The session accepted SIGFPE stop/print/nopass
configuration and creation/deletion of a pending `main` breakpoint. Python
reported the stop-event, SignalEvent, Breakpoint, read-register, read-memory,
execute and parse/evaluate APIs. Its x86-64 architecture metadata describes
RIP, XMM0, XMM1 and MXCSR. `$_siginfo` evaluates to `void` without an inferior.

The selected inferior PID remained zero and no target file was loaded.
No inferior was run, started or attached; no MadLoop executable or reduction
library was loaded. These checks do not qualify ptrace access, register values,
signal-info field availability, a real main-breakpoint hit, the mapping-ack
handshake or physical/numerical behavior. Those require a separately reviewed
source-bound proposal and explicit root admission before native work.

`inspect.py` and `target-free.gdb` record the inspection procedure. This folder
contains only UTF-8 text. It is capability evidence, not a replay runner.
