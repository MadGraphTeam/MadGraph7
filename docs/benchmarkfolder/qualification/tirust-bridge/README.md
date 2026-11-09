# TIRust bridge qualification

This directory records native interface correctness tests. The TIRust shared
libraries used in the original tests were **debug builds**; NinjaRS in the mixed
tests was a release static library. The later `stability-audit/release-*` records
replay the retained stress point against the installed release bridge. Neither
set qualifies runtime performance. Release installation is separately recorded
in `../tirust-install`. No executable or library binary is committed here.

Numerical acceptance remains open: one served DP stress fixture exceeds the
campaign's independent symmetric `3e-6` criterion, while QP resolves it. The full classification
and failing coefficient are retained below and in `qualification.json`.

`rust-tests.log` contains the complete output of nine bridge tests. They cover
binary128's last bit and signed zero, a complex-mass rank-four box at 256 and 384
working bits, a failed overflow reduction followed by a successful valid call,
Cartesian rank-six ordering, split UV/IR poles, and the bounded QP tensor cache.
The precision convergence bound is `2^-180 * (1 + |reference component|)`.

`gate-passed-classified.jsonl` and `method-choice-classified.jsonl` retain every fixture result through
the actual C ABI, with the supplied invariant matrix. The source fixtures are
the pinned TIRust checkout's NinjaRS reference data; their paths and SHA256
hashes are recorded in `qualification.json`. Every QP lane must succeed. DP
refusals are recorded and allowed by these upstream stress tests. Served DP
lanes must meet the source test's error bound (`1e-5` for gate-passed lanes,
`1e-6` for method-choice lanes); QP uses `1e-8` against the binary64 fixture
truth. The error is the largest Laurent-coefficient difference divided by the
largest reference coefficient. This comparison accounts for MadLoop's odd-rank
sign and uses no tensor multinomial factors.

`fortran-smoke.log` checks the compiled ISO C binding, true binary128 transport,
finite-skipping pole evaluation, cache hits and independent checks, lifecycle
generations, errors, and directed rounding. `exports.txt` lists every defined
dynamic export; all names must begin with `ml_tirust_`.

`mixed-link/` contains the exact driver, commands, logs and binary hashes for
both library link orders and both runtime initialization orders. All four
serial runs passed. An initial run hit Symbolica's restricted concurrent-thread
allowance while another process evaluated integrals; the identical binary
passed on retry and in the serial four-case matrix. The original failed log is
retained. These tests used one unchanged Symbolica license configuration and
lock directory. Numerical test processes must be serialized in that environment.

Run commands from the MadGraph repository root. `commands.sh` documents the
bridge build, source override, fixture tests and Fortran compile/link flags.
The mixed test can be replayed by copying `mixed-link/smoke.f90` and
`mixed-link/run.sh` into `.tirust-build/mixed-link/`, then running the latter from
the repository root. It consumes the installed NinjaRS compiler-runtime
manifest, including the compiler-resolved `libgcc.a` before the Rust archive.
The earlier mixed matrix and final fixture matrix have separately recorded DSO
hashes: the final bridge adds failed-DP cache cleanup without changing successful
numerical results.

The stress fixtures are **not all accepted by the independent truth criterion**. Of 66
DP/QP evaluations, 52 were served and 14 DP calls were refused. Among served
results, 51 pass the diagnostic application of the campaign's per-coefficient
symmetric `3e-6` criterion; one DP result fails. The failing lane is the
synthetic massive `N=4`, rank-five LI-32 row 57 lane 44. Its finite coefficient
has a whole-lane relative error of `6.958943498811447e-6`; QP resolves it to
`3.5148270968783686e-15`. The detailed symmetric ratio and each complex
coefficient are retained in `qualification.json` and the classified logs.
The upstream gate test permits `1e-5`, so passing that test is a weaker statement.
The [native stability audit](stability-audit/README.md) demonstrates that the
default `MLStabThres=1e-3` can miss correlated DP errors at this synthetic lane,
even with reversal and two ordinary rotations. Explicit QP resolves it; the
tighter `1e-6` threshold requests rescue in the actual native estimator. This
was followed by an observed execution of the verbatim production rescue block
with the real Fortran tensor callback: all four tighter-threshold cases enter
QP and return successful-rescue code `323`, matching the QP reference. The
full physical helicity/filter pipeline is outside that experiment. These complex integral fixtures
are not physical squared matrix elements, and the physical campaign's
independent truth gate remains necessary. The initial logs, before component-level
classification was added, remain in `gate-passed.jsonl` and `method-choice.jsonl`.

`dso-dynamic-section.txt` and `dso-runtime-dependencies.txt` record the native
runtime dependencies of the exact final debug DSO. GMP and MPFR are contained
in it; the runtime needs compatible system `libgcc_s`, `libm`, `libc`, and the
GNU/Linux dynamic loader. Generated Fortran executables also need the selected
compiler's Fortran runtime. Relocation copies the bridge DSO and adjusts its
search path within this qualified environment; it does not bundle glibc or
promise portability to an incompatible system ABI. The release installer must
check its final DSO dependencies and SONAME independently.

`evidence-index.json` hashes the packaged text/source evidence independently of
the recorded native binary identities. Regenerate it with
`python docs/benchmarkfolder/qualification/tirust-bridge/write_evidence_index.py`
after adding evidence.
