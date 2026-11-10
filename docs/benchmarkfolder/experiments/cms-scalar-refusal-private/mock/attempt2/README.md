# Attempt 2: isolate the target loader's library path

Attempt 1 stopped at its first loader check with exit 127. The outer `taskset`
received the target GNU runtime through exported `LD_LIBRARY_PATH` and failed
with a `GLIBC_PRIVATE` symbol error before either mock started. Its source,
prepared manifest, run record and raw failure log are retained unchanged and
hash-bound here; failed run SHA256 is
`c79ab580f2799f0bcebbd11b12b8ca88c5ac2d47f35b35595cac0a47a91312ab`.

This append-only correction passes the frozen runtime directories directly to
the target's exact `PT_INTERP` via `--library-path`, for both `--list` and target
execution. The outer reviewed supervisor and `taskset -c 8` receive the normal
frozen environment. No target runtime path is exported. `control.patch` shows
the complete controller difference; execution, timeout, cleanup, event capture
and output-closure helper bodies are unchanged.

Numerical sources, mock binaries, all five runtime object paths/hashes,
validators and expected outputs remain exact. No compile/link job exists.
Two loader checks (five seconds each), classifier mock (15 seconds) and caller
mock (10 seconds) remain sequential under the same inclusive 35-second cap.
Each uses its new empty directory here, with new logs and `run.json`. Failed or
interrupted attempts are retained without automatic retry. These mocks use no
real scalar or Rust routines, so the original no-reducer-lock scope is retained.

From the MadGraph root in the same frozen normal environment:

```sh
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python -B .tirust-build/cms-scalar-refusal-prototype/mock-qualification/attempt2/control.py verify
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python -B .tirust-build/cms-scalar-refusal-prototype/mock-qualification/attempt2/control.py run
```

The default action is read-only `verify`. Preparation and Python tests do not
launch a compiler, loader or mock. Root has authorized the mechanical replay;
the parent executes only after receiving this source freeze. A passing replay
would qualify only the frozen mock assertions. Actual analytic accuracy, full
MadLoop rescue, hardware sticky flags/traps/rounding transparency and performance
remain outside scope.
