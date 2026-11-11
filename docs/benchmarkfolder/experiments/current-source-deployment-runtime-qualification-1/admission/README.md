# Additive interpreter binding for separate runtime admission

The controller main entrypoint has not been executed. The original proposal, compiled exports,
installed libraries, build receipt and numerical policy remain unchanged.

The recorded `ldd` command used the diagnostic loader from glibc 2.42 and
reported the executable's glibc 2.40 interpreter on the left of `=>`.
The original build controller recorded the right-hand diagnostic-loader path.
Pure ELF64 parsing now identifies the actual `PT_INTERP` in each unchanged
executable: `/nix/store/j193mfi0f921y0kfs8vjc1znnr45ispv-glibc-2.40-66/lib/ld-linux-x86-64.so.2`. Both have the same interpreter SHA-256
`1e08370bba3ee9e4f97bb0500d1f32afb0417babca6ae455c6458b9a3edb86a8`, 253696 bytes. This is prelaunch input inspection, not evidence
of an initialized provider or successful amplitude evaluation.

`interpreter-supplement.json` SHA-256 is `c85d97814cec179eb449244b0bdd700cbca0c225ddb748749634e79e192e1854`.
It binds the original proposal seal, original controller and build receipt;
each executable's exact hash/size and PT_INTERP extent/string; and the one
additional interpreter provider. The old ldd provider set remains unchanged.
The executed binary files and source tar are not copied here.

The full `smoke.py` derivative differs from the sealed controller only by:

- locating original helper imports under `../proposal` and checking every
  original sealed proposal file before importing those helpers;
- checking the exact additive supplement and original source/build bindings;
- rehashing both executables and the interpreter, then decoding their PT_INTERP
  before the first worker starts;
- adding precisely that one shared interpreter path/hash to an in-memory copy
  of the original receipt's provider set.

All modes, numerical comparisons, expected statuses, cache/pole settings,
resource caps, native-lock policy, worker handling, output locations and original
post-run artifact checks remain unchanged. Those post-run checks also rehash the
added interpreter. `smoke.diff` is the exact unified diff from the original
controller; the original bytes are preserved at `proposal/smoke.py`.
No original receipt or source seal is edited. `pure-preflight.json` records
independent source/AST and byte checks: the seven original helper functions are
unchanged, main differs only at receipt loading, both executable PT_INTERP values
match, and all 1,871 bound artifacts, 13 old providers, the one added interpreter
and protected production inputs passed fresh read-only checks. No child, native
load or amplitude evaluation occurred in that preflight.

`source-index.json` seals this derivative, supplement, diff and preflight. Root
must separately admit runtime execution after the priority EW campaign releases
the numerical lock. The intended command, from the repository directory, is:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/current-source-deployment-1/runtime-admission-2/smoke.py
```

The 900-second total allowance includes a maximum 120-second lock wait. The
original per-worker limits, CPU 20, 16 GiB address-space limit, no-retry policy,
fixed unchecked-mode scope and NLO partial-amplitude rejection remain unchanged.
