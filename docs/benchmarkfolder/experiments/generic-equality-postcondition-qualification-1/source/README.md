# Standalone exact geometry helper: qualification proposal

This is a dormant **source-only** BUILD/RUN proposal. No compiler, loader or
native driver has run. The helper is an exact copy of frozen source
`3d2e4b775334374082771f670996e67138205b66cf7d5a7ff084013235046f4c`, Fortran
`4dfedb9f940dd8e0565ff1ff34b9aa0d56635a5e5d8c52b328359fbfc754afda`.
It contains no changed helper arithmetic or scientific dispatch.

The question is narrow: does the actual compiled integer128 helper agree with
independent rational answers and preserve observed FP controls/flags while
processing supplied bit patterns? There is no model, scalar library, Rust
bridge, physical numerator or amplitude in this standalone lane. Snapshot
ownership, current-point lifecycle, final physical dot arithmetic and runtime
performance remain separate requirements.

## Frozen inputs and oracle

`fixtures.hex` contains 1,698 cases: the earlier 500 exact random inputs,
500 deliberate perturbations and 600 arbitrary finite inputs, followed by
98 explicit controls/witnesses. The independent Fraction oracle and a separate
exact signed128 capacity calculation predict 526 accepts and 1,172 refusals.
Ten mathematically equal inputs deliberately exceed capacity and must refuse;
cancellation never relaxes product/alignment/absolute-sum safety guards.

Controls include signed zeros, IEEE extrema/subnormals, signaling/quiet NaNs,
nonreal values, all malformed array slices, N=0/64/65, the default integer
minimum, coefficient product and shift/capacity boundaries, binary64 forbidden
high bits/sign extension, and positive/negative distinctions at binary128's
113th significant bit. All fixture words travel as full32-digit hex to Fortran
INTEGER128. The driver echoes every input word and coefficient once; the parser
reconstructs each complete canonical case and authenticates its recorded hash.
This tests integer bit transport, not a newly generated REAL16 numerator.

The four CMS20/21 DP1/QP4, route3 pair(2,4) cases retain exact source/trace pins
and the external ALOHA sign convention. Their q-component words are **conditional
nearest-even reconstructions**, not words directly observed at the old ABI.
The old streams contain no owned point snapshot. Their usefulness is prospective;
this standalone test cannot promote them to physical/lifetime qualification.

The native helper returns one logical answer. Expected refusal reasons,
capacity metadata and component-error semantics are offline oracle fields;
they must not be reported as additional observed native API outputs. A true
answer authorizes zero construction error under the declared mathematical
reference, but no generated caller error array is exercised here.

## Separate bounded stages

Both stages have an all-in180s cap, including60s reserved cleanup,4GiB address
space,256MiB output limit, nonexclusive CPU12 and the shared numerical mutex.
Preflight, source checks, child execution and administrative closure share the
same budget. No phase retries, chains or subsequent stage starts automatically.
An existing work directory cannot be reused to hide a failed attempt.

**BUILD** compiles one C driver observer and two private executable variants:
the unchanged helper at `-O0 -fcheck=all -ftrapv` and at `-O3 -fno-fast-math`.
Both use a checked, unoptimized Fortran driver. The compiler is pinned GNU15.2
from the successful O3 source2 environment. Required tools, executable bits,
PATH resolutions, wrappers/frontends, inherited-injection removal and original
compiler policy are verified using file reads, without Nix store globs.

Before linking, the link helper binds objects, tools, CRT/linker inputs and
libraries. An explicit five-provider standalone-runtime join is applied and
recorded **before** linking; it is not the full-amplitude fifteen-provider set.
Actual linker LOAD records, a pure unit-specific DT_NEEDED/RUNPATH closure and
the actual bound PT_INTERP `--list` must agree. Helper symbol ownership is unique
and its undefined symbols must contain no floating/environment helpers. BUILD
does not run the Fortran main.

**RUN** requires a separately reviewed successful BUILD terminal hash, rehashes
both executables, objects, link proofs and providers, and starts two fresh
workers serially. READY/GO and DONE/EXIT handshakes permit actual `/proc` maps
and CPU-affinity checks before input evaluation and after all helper calls.
The strict parser checks complete input echoes, all expected answers, mode
coverage, restoration and raw/API control consistency. Checked and optimized
decisions must agree. Any mismatch/refusal of an expected accept, unexpected
accept, setup/cleanup error, unsupported dependency or malformed evidence fails
the stage and retains its raw artifacts.

## Explicit driver environment controls

The mathematical helper contains only integer operations and cannot call any
environment setter. Setters exist solely in `fenv_observer.c`, linked into the
standalone driver. There are20 frozen cohorts for each case:

- An untouched baseline and all four rounding modes.
- Each of five sticky flags and their union.
- Each corresponding enabled halting mask and their union, with zero pending flags.
- FTZ, DAZ and both, only when the x86_64 architectural mask supports them.

The driver saves its original environment for each cohort. It masks all six
hardware exceptions before clearing flags, installs requested sticky flags
through the opaque flag API, and enables halting only with no pending flags.
It observes actual rounding, sticky/halting flags, MXCSR raw/control and x87
control/status immediately before and after the helper. All input/output and
formatting occur outside the altered interval after observed restoration.
If restoration fails, a bounded integer-only raw stderr write and `_exit(86)`
run before any Fortran formatting. This failure path cannot become a pass.

Only optional FTZ/DAZ cohorts16–18 may emit explicit SKIP records, which receive
no environment qualification. The GNU/x86_64 rounding, flag and halting cohorts
must execute. Setup or restoration errors fail. The parser verifies requested
versus actual x87/SSE rounding and masks, FTZ/DAZ, observed status preservation,
restored fields and identical mode support across the corpus. It does not infer
unobserved complete FPU register preservation from these seven fields.

## Source tests and future admissions

Pure tests regenerate every fixture/oracle, check conservative capacity
refusals and the bit113 controls, reject corrupted input echoes, answers,
rounding/flags/restoration and incomplete transcripts, verify unsupported-mode
disclosure and compiler-injection replacement, and enforce separate RUN
admission/budgets. A synthetic transcript tests the parser only, not native
environment behavior. Independent static review caught and fixed formatted
failure reporting before restoration; no executable had been built.

Commands, from the repository working directory with the pinned Python:

```sh
python3 -B -m unittest discover -s .tirust-build/generic-equality-postcondition-qualification-proposal-1 -p test_proposal.py -v
python3 -B -O -m unittest discover -s .tirust-build/generic-equality-postcondition-qualification-proposal-1 -p test_proposal.py -v
python3 -B .tirust-build/generic-equality-postcondition-qualification-proposal-1/controller.py --preflight --phase build --source SOURCE_SHA256
# Only after separate source review/admission:
python3 -B .tirust-build/generic-equality-postcondition-qualification-proposal-1/controller.py --execute --phase build --source SOURCE_SHA256
# Only after separate closed BUILD review/admission:
python3 -B .tirust-build/generic-equality-postcondition-qualification-proposal-1/controller.py --execute --phase run --source SOURCE_SHA256 --reviewed-build BUILD_TERMINAL_SHA256
```

Success would qualify only the standalone integer helper and the observed
executed driver cohorts. The source1 freshness plumbing, an owned repaired
point snapshot, genuine same-kind numerator construction, actual q-at-use
observation, structural/mass selection and final-dot contract remain outside
this lane. Production integration, physical agreement and precise local
performance acceptance remain open.
