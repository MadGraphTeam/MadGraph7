# Private adaptive-QP Fortran qualification

This source-only package proposes two fresh standalone tensor workers linked
against the released ABI1 bridge and the private adaptive debug bridge. It does
not change an installer, production module, generated amplitude, or active
campaign. Compilation and native execution require separate root admission.

The four adapter/serialization sources are exact copies of the qualified mock
package `adaptive-qp-tier-telemetry-private`. `source/origins.json` binds those
copies. The optional weak getter keeps the released ABI1 library usable. There
is no new mandatory ABI symbol and no reinterpretation of return code323.

The legacy worker performs one ordinary QP tadpole. The adaptive worker performs
three ordinary calls (cold, repeat, independent check), three captured N4R3 full
calls (cold, repeat, check), three captured poles-only calls (cold, check,
repeat), and one deliberately invalid pole-mode call. The N4R3 words and ASCII
records come unchanged from revision3 of `adaptive-qp-rescue-private-build`.
That captured Gram refusal is a transport/control-flow fixture, not physics
truth. The 20 private Rust unit tests now pass in their separate bound package;
they do not substitute for these unexecuted Fortran checks.

Fortran reads exact two-word binary128 inputs, formats the actual113-bit values
through the existing64-byte transport, and emits returned raw binary128 words.
The C observer forwards each tensor call exactly once, records input/output
bytes, and copies numerical errors before telemetry queries. A second wrapper
records all11 returned C shim words for comparison with Fortran telemetry.
Neither wrapper allocates, prints, performs floating arithmetic, queries caches,
or changes numerical statuses during a wrapped call. Fixed-buffer output is
flushed outside reduction/query calls. Successful output buffers alone are read.
The analyzer checks exact input and output rounding, every scheduled call,
context generation, word transport, errors, cache/check tier deltas and cache
namespace counts. Full/partial pole comparison reuses the Rust test's bound
`2^-100*(|a|+|b|+1)`. The finite slot in partial mode is explicitly unevaluated;
its zero ABI placeholder is not a computed physical coefficient.

The first nine telemetry words are cumulative context-lifetime counters. Cache
clear/budget changes must preserve them; finalize/recreate must produce a new
generation and reset. Last tier and condition are snapshots, never additive.
256-bit success counts and accepted-amplitude precision remain unknown. The
driver does not construct a numerator, Born term, amplitude, or physical rescue
chain; no kinematic-fix or performance conclusion follows from this diagnostic.

The proposed compile stage uses CPU12 nonexclusively, one command at a time,
one120-second shared cap and4GiB address space: three Fortran objects, two C
objects and two links. All numerical objects are identical between executables;
only the selected exact bridge differs. The adaptive DSO has no SONAME, so its
absolute private filename is linked directly. Both executables use the pinned
glibc2.42 interpreter/runtime required by the private build; this is isolated
from all existing active executables. No compiler probe invokes a loader or
native worker. The stage retains exact argv, logs, link maps, all resolved LOAD
input hashes and final object/executable hashes.

The separately proposed native stage serializes two workers under the common
nonblocking numerical lock. Each receives180seconds/4GiB on CPU12; each actual
interpreter `--list` gets30seconds first. Runtime files and actual `/proc` ELF
mappings must agree. A bounded10-second startup handshake keeps the tiny legacy
worker alive until the existing supervisor captures its mappings; it occurs
before any tensor call. The reviewed supervisor is reused, with only its CPU
affinity assertion changed11→12 and a callback writing the handshake marker
after a complete capture. All error paths retain the attempt and stop without
retry. Inputs, output files, errors and mappings remain available on failure.

From the repository root, after sourcing `tools/tirust_env.sh`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification/source/controller.py verify
# Separate admission required; no loader or native call:
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification/source/controller.py build
# Separate later native admission required:
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification/source/controller.py run
```

Formal final timing is a TODO for Hua-Sheng's local machine. Any current-host
elapsed times in this package describe resource usage only.
