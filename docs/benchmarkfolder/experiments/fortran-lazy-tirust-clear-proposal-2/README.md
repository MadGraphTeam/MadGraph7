# Private Fortran experiment: lazy TIRust event clearing

This successor preserves source proposal 1 unchanged and addresses root review before any native execution. It acquires the common lock before creating the run namespace, keeps compilers and all outer helpers free of LD_* variables, applies the fifteen original providers only at the final native execve boundary, and joins observed/plain numerical byte parity for both arms. This proposal is unexecuted until root admission. It changes no production file,
installation, frozen campaign, numerical integral implementation or COLLIER code.
It addresses finding 4 of the higher-multiplicity cache audit: an exported
TIRust interface currently creates a context and crosses its mutex-protected
native clearing boundary even in a fresh process selecting only CutTools.

The candidate changes only the generated `CLEAR_TIRUST_CACHE` procedure. The
existing `TIRUST_CONTEXT_GENERATION()` getter is a Fortran `c_associated` check;
its result is -1 exactly while no shared module context exists. Calling the
unchanged public clear only when that result is nonnegative avoids the empty
native work. Guarding by the selected backend or by a subprocess-local generation
would be wrong: an initialized second subprocess can share and populate the
same module context. Both prefixes must invalidate that shared active context.

The public `tirust_clear_cache` still creates a context when explicitly called.
Existing Fortran modules and ABI1 installations expose both procedures already,
so this patch requires no new Fortran API, metadata capability, module, Rust
build or dependency revision. Once a context exists, this code calls the same
public clear, with the same local/native diagnostic reset and native failure
behavior. While absent, internal event clearing deliberately preserves existing
diagnostics and generation instead of creating a context and clearing its error.
Finalization/recreation and generated plan-registration epochs stay untouched.

`bindings.json` pins current generated n1 SM sources, 18 existing process objects,
model/DHELAS/static numerical libraries, the production Rust DSO, installed module
and the original fifteen runtime providers. These process objects were not
individually hashed in the historical campaign record; their bytes are bound
now, without claiming historical object identity. No object other than the
private candidate TIRust adapter and the measurement/test drivers is rebuilt.
The original Fortran compile policy for the adapter is retained.

The bounded native protocol is:

1. Acquire the common numerical lock nonblocking. Use CPU 12 nonexclusively,
   4 GiB address space, 90 seconds per command/worker and a 900 second work budget
   with cleanup reserve. Only owned child process groups may be terminated.
2. Compile the unchanged campaign worker and an integer-only C observation
   object. Link it against the unchanged adapter and numerical objects. Before
   constructing any candidate source/build, profile explicit backend 1 and 9
   over the existing 100 deterministic on-shell boost points. Record actual
   creation/clear call counts and time inside those two FFI boundaries, including
   startup/finalization. This instrumented profile explains lifecycle work and
   is not a full-amplitude speedup measurement.
3. Only after that profile demonstrates the expected original lifecycle work,
   compile a private adapter with the candidate guard. Link plain original and
   candidate workers and an observed candidate. Require explicit CutTools to
   perform zero native TIRust create/clear calls; selected TIRust must retain the
   original count of clears after plan registration. Initial empty clears
   preceding the first registration are counted separately; they may also be
   removed from selected-TIRust startup. The complete final traversal (Born, finite, single/double
   poles, statuses and host estimates) must be byte identical for each backend.
4. Compile actual installed-module lifecycle fixtures with two generated clear
   wrapper prefixes. Test fresh and post-finalize no-creation, inactive diagnostic
   preservation, active diagnostic reset, an actual seeded cache cleared by the
   other prefix, public creating-clear compatibility, recreation epochs and IEEE
   flags/rounding. No native mutex poison/failure injection is attempted; the
   active path is source-identical to the original call.
5. Measure direct wrapper clearing in inactive and active states, seven
   alternating pairs of one million calls each. Label this only as event-clear
   overhead. Then use uninstrumented persistent workers for explicit CutTools
   and TIRust, all 100 points and the frozen stability policy. Calibrate equal
   batch lengths to about one second, at most one million traversals, and collect
   seven alternating original/candidate pairs for each backend. Require numerical
   byte parity after every batch and retain every raw result/failure.

All generated worker cards, immutable resource files, stdout, native provider
maps, timing/status/value outputs and compiler commands remain in the new
ignored `.tirust-build/fortran-lazy-tirust-clear-1` namespace. The root controller
must observe its actual terminal status before any result is described as done.
No retries, broad matrix expansion, CPU reservation or production promotion is
implicit. Loader traces apply their flag only at the final target execve boundary; setting the trace flag on
`taskset` would inspect the wrong program. Runtime maps from the actual warmed
workers must match all fifteen original providers exactly.

The resulting timings are shared-host observations only. Precise per-process
acceptance remains Hua-Sheng's local TODO. Retain the implementation only if
the measured direct lifecycle benefit and full-amplitude parity justify it;
report plain full-amplitude effects separately from instrumented FFI costs.

Pure checks, without compilation or numerical execution:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/fortran-lazy-tirust-clear-proposal-2 -p test_source.py -v
python3 -B docs/benchmarkfolder/experiments/fortran-lazy-tirust-clear-proposal-2/run.py verify
```

After independent review and root admission only:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-lazy-tirust-clear-proposal-2/run.py run
```
