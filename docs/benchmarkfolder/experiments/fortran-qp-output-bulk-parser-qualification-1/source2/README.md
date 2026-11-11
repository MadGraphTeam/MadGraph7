# Bounded public QP parser qualification — source only

This is the completed successor to the paused, unsealed four-file unit draft.
It does not patch production. The exact original module and exact candidate
come from source proposal `600fb55c…`; candidate SHA256 is `256f0d87…`, original
is `3cb9cf98…`. The original proposal's `EQUIVALENCE.md` remains the full later
promotion contract. This first gate qualifies only public decimal transport
with a recording C ABI, not a real Rust evaluator or amplitude.

BUILD compiles the recording mock and four separate GNU Fortran module/driver
executables. Original and candidate use their exact frozen sources. Observer
and forced-failure modules are separately generated, pinned test-only variants:
the observer records the actual selected branch; the forced variant injects an
error before an otherwise eligible bulk read. Neither variant is scientific
candidate evidence. No Rust, reduction archive, generated process or model is
linked. Actual link maps, unique symbols and the prebound ELF interpreter and
six runtime providers are checked before a program may run. Compiler/loader
discovery occurs inside BUILD and is recorded, rather than source preparation.

RUN has 149 deterministic corpus cases: 133 ordinary public cases run original,
candidate and observer; two dedicated forced-hook cases run only that hook;
14 undefined or platform-specific parser-syntax cases are lexical-only and
never compared using uninitialized public output. The corpus covers all 42
qualified N/rank pairs, the 21 first/middle/last malformed record positions,
39-significant-digit normals, exact signed zero, binary128 tie and last-bit
boundaries, scalar exponent/subnormal/nonfinite fallback, partial/pole modes,
cache bypass/check passes, clear failure, four rounding modes, five entry sticky
flags and five halting modes. A compact pattern expands to the exact complete
64-byte-record batch for every call. No corpus reduction or automatic retries.

Every normal invocation makes five public calls with scale keys 1,1,2,2,1.
Key1 is an intentionally noncanonical valid scalar record batch; key2 carries
the case's frozen wire batch. The provider inserts before public validation.
The analyzer compares statuses, exact error messages, all binary128 words,
rounding probes, pre/post IEEE flags and halting states, cache entries/bytes,
hits/misses/bypasses/clear calls and final context destruction. Defined success
words in nearest mode are independently computed by integer rational conversion
directly to binary128, never through binary64. For non-nearest modes, default
internal-file READ has no explicit ROUND= promise in this experiment. Exact
original/candidate/observer word equality remains a hard gate, but comparisons
with directional integer references are retained separately as unqualified
diagnostics; no directional-input-correctness claim is made. Arithmetic rounding
probes, pre/post modes and IEEE state still have independent hard gates. This
policy was fixed before any compiler/native execution, not selected from an
observed result. Initially sticky flags may not be cleared.
Unexpected read-error injection must retain zero outputs, exact local error,
one cache-clear attempt per failed call and valid-call behavior afterward.

Actual `ABI_ENTER` and observer branch markers use integer-only byte writes.
Halting cases must have matching original/candidate/observer terminal states
and retained structured output. A trap before ABI entry cannot qualify output
fallback; a retained observer `SCALAR_ENTER` identifies actual decoder entry.
Signals are retained and never changed into successes. Partial mock modes
qualify parser/options dispatch only, not physical Laurent pole zeroing.

The source tests and preparation are pure Python file operations. Exact
source, tool/wrapper, compiler environment and closed profile-runtime pins are
bound before BUILD. Current production module, Rust bridge, installed eb3d DSO
and configuration are protected before/after. One BUILD has an all-in 180s cap;
one RUN has 600s. Each reserves 60s inside that cap for identity-checked cleanup
and administrative failure closure. Preflight and reviewed BUILD verification
are inside these deadlines. Shared 8/9 numerical mutex, nonexclusive CPU12,
one-thread environment, 16GiB address-space, 512MiB per-file and zero core are
fixed. Owned cleanup records observed start identities; it does not claim a
cgroup-wide census. Any unresolved cleanup stops dependent stages. A raw child
handle is retained before ownership construction, with a failure-only TERM/KILL
path; ownership uncertainty never qualifies a subsequent run.

After source review, commands are separate (exact hashes must be supplied):

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/fortran-qp-output-bulk-unit-controller-2/controller.py --preflight --source SOURCE_SHA256
python3 -B .tirust-build/fortran-qp-output-bulk-unit-controller-2/controller.py --build --source SOURCE_SHA256
# Root reviews the successful closed build and pins its terminal.json first.
python3 -B .tirust-build/fortran-qp-output-bulk-unit-controller-2/controller.py --run --source SOURCE_SHA256 --reviewed-build BUILD_TERMINAL_SHA256
```

No parser parity or speedup is claimed before those stages close. Real wire
eligibility counts, actual unchanged-full100-learning amplitude parity,
full100 physical observation, and paired descriptive end-to-end timing remain
separate later stages under the unchanged original contract. IEEE modes outside
the four compiler-supported modes, exhaustive NUL-position native mechanics,
legacy undefined-null/slash mechanics, and real-cache failure behavior are not
qualified by this limited unit corpus. The optional optimization must still
show measured benefit before production consideration. Precise performance
acceptance remains Hua-Sheng's local TODO.
