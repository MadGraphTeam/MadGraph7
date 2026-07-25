# Native compiled DirectApplication prototype

Status: real C++ producer slice validated; not wired into production.

Base: `56feb0c30141b88ab49561630b07d739c9e98aa4`

## Existing boundary

Compiled C++ and ASM stage leaves currently share one target-native dynamic
library envelope, but expose only Symbolica's scalar-row ABI:

```text
<name>_complexf64(params, buffer, out)
<name>_complexf64_get_buffer_len()
```

Rusticol's `CompiledComplexF64Evaluator` calls that function once per point.
The compiled engine gathers stage inputs into dense point-major rows before the
call and scatters dense outputs afterwards.

The prepared-model native recurrence exporter is not an acceptable shortcut
for compiled stages. Its generated direct entry point still constructs
`split_params` and `split_out` for one row, calls the scalar raw evaluator, and
stores the result. That is appropriate evidence for the recurrence transition
contract, but remains a dense-row wrapper under the compiled Direct-Arena
acceptance rule.

## Prototype ABI

`native_compiled_direct.rs` adds a backend-neutral v1 loader contract that a
C++ or ASM library can export:

- a metadata symbol certifying split-complex, component-major,
  point-contiguous, factor-free overwrite, no-output-alias and non-throwing
  semantics;
- fixed arrays of input-plane, scalar-pointer and output-plane descriptors;
- one `point_start, point_count` native call returning an integer status;
- logical bindings resolved and validated once when persistent storage is
  pinned;
- unchecked native plane execution after bounds validation, with no packet
  buffer, gather, scatter or remap in the Rust adapter.

The original loader test library is a **synthetic scalar plane-loop fixture**
implementing a small complex multiply/add leaf against the planes themselves.
It remains useful for bounded loader/alias/status tests, but is no longer the
only producer evidence.

The prototype validates metadata/counts, portable C symbol names, input/output
alias rejection, arena/momentum/parameter bounds, nonempty point ranges,
odd/tail point ranges, checked `u32` descriptor counts, and returned native
statuses. The target-native producer must mark both C entry points `noexcept`
and convert every recoverable failure to a status code. Rust does not and
cannot contain foreign unwinding across an `extern "C"` boundary, nor can it
recover from a foreign signal or process termination.

Traffic counters cover only packet/gather/scatter/remap work performed by the
Rust adapter; they cannot see hidden work inside a native function. Allocation
counts likewise use Rust's global allocator and do not observe foreign
`malloc`/`new`. Source inspection establishes that this synthetic fixture does
neither, but real generated libraries require their own audit and profiler
evidence.

## Measurement

macOS AArch64, release mode, 129 points, nine interleaved samples, 10,000
repetitions per sample:

| path | median | MAD | warmed Rust allocations |
|---|---:|---:|---:|
| native plane call | 55 ns/call | 1 ns | 0 |
| gather + scalar native calls + scatter | 343 ns/call | 2 ns | 0 |

The isolated synthetic fixture ratio is `6.236x`. This remains raw
scalar-plane-loop/ABI evidence, not an end-to-end process claim.

## Genuine Symbolica C++ producer slice

Symbolica 2.2.0 exposes a documented, reusable expression boundary:
`Evaluator.get_instructions()`. It returns the optimized evaluator
instructions, the exact temporary-slot bound, and constants specifically for
code generation in other languages. `native_direct_cpp.py` now lowers that
instruction stream directly to a split-plane C++ entry. It does not parse the
generated dense source, invoke `<name>_complexf64`, or construct dense
parameter/output rows.

The risk-first producer:

- accepts explicit semantic storage classes for every source parameter
  (`complex-plane`, `real-plane`, `complex-scalar`, or `real-scalar`);
- emits fixed descriptor indices and factor-free output overwrites;
- uses portable Clang/GCC vector extensions for explicit two- or four-double
  SIMD bundles and a scalar remainder loop for odd tails;
- exports the target triple, sorted CPU-feature set, retained evaluator-state
  SHA-256, and logical stack bound;
- derives its stack contract from Symbolica's temporary count plus local
  outputs and refuses more than 64 KiB by default;
- rejects unknown instructions, inconsistent real-value annotations,
  uninitialized reads, descriptor aliases, invalid counts, null descriptors,
  and overflowing/empty point ranges;
- is `noexcept` and maps recoverable failures to status codes.

The current fail-closed instruction subset is `add`, `mul`, `assign`, and
`pow(-1)`. No fallback exists for a rejected operation.

`_CompiledComplexEvaluatorAdapter.compile_native_direct_application_prototype`
is the cold integration seam. It requires stage lowering to supply semantic
storage kinds rather than guessing scalar inputs from arity. The developer
probe reads those kinds from a genuine compiled stage manifest, loads the
matching retained evaluator state from PACBIN, and compiles an independent,
direct-only translation unit. The original Symbolica dense library is loaded
separately solely as the same-expression oracle. The direct library exports no
`*_complexf64` symbol.

### Real-process evidence

Source base: `56feb0c30141b88ab49561630b07d739c9e98aa4`.

Artifact:

- process: built-in `d d~ > Z g`, compiled C++ O3, native target;
- artifact ID:
  `2cc523f424722a4763853c8b7c585a40af0ef78a524b91d50fea44a33006b094`;
- normal generation: 6 s under the 30 GiB watchdog, 0.197 GiB peak RSS;
- payload: 46 files, 1.47 MiB.

Selected real leaf:

- stage 1, chunk 0;
- 19 complex source parameters and four complex outputs;
- ten complex point inputs, eight real point inputs, and one fixed real model
  scalar;
- 77 optimized Symbolica instructions and nine temporary slots;
- 28 input planes, one scalar descriptor, and eight output planes;
- 416-byte logical SIMD scratch bound;
- evaluator-state SHA-256:
  `9dbe892d68a3babff4e81aee82bf964528f1aea6971427f699639f97e8b06bce`.

Parity passed at ranges `0+1`, `0+2`, `0+3`, `0+127`, `1+127`, `0+128`,
and `0+129`, including inactive-output sentinels, with `rtol=1e-12` and
`atol=1e-15`. The maximum absolute difference was
`7.687954195124218e-12`; input/output aliasing returned status 4 and a
descriptor-count mismatch returned status 2. Disassembly contains native
AArch64 `ldr q`, `str q`, `fmul.2d`, `fadd.2d`, and `fsub.2d` operations in
the direct symbol, while the same entry's scalar remainder handles the odd
point.

Producer compilation took 0.192 s. Replacing the selected dense source/library
pair with the direct-only pair changes:

| payload | dense | direct-only | delta |
|---|---:|---:|---:|
| source | 3,584 B | 18,087 B | +14,503 B |
| library | 34,064 B | 18,368 B | -15,696 B |
| selected pair / artifact | | | -1,193 B / -0.077% |

Seven interleaved 129-point Python-FFI probe samples (2,000 repetitions) gave:

| path | median | MAD | median per point |
|---|---:|---:|---:|
| plane-native direct | 1,406 ns/call | 9 ns | 10.899 ns |
| prepacked dense evaluator | 4,530 ns/call | 62 ns | 35.116 ns |
| NumPy gather + dense + scatter | 5,786 ns/call | 21 ns | 44.853 ns |

The ratios are `3.222x` and `4.115x`. These timings establish that the real
emitter and ABI do not lose the copy-removal opportunity. They are not an
end-to-end Rust engine claim: the dense Python API allocates its result and the
full comparator uses NumPy materialization.

Reproduction:

```console
SYMBOLICA_LICENSE=... XDG_CACHE_HOME=.cache \
  python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  pyamplicol generate 'd d~ > z g' /private/tmp/native-direct-zg \
  --model built-in-sm --backend cpp --execution-mode compiled \
  --workers 1 --cores 1 --output-chunk-size 64 --cpp-optimization O3 \
  --cpp-native-arch --no-validation --no-post-build-validation \
  --no-emit-api-bundle --progress log

SYMBOLICA_LICENSE=... PYTHONPATH=src \
  python tools/ci/memory_watchdog.py --limit-gib 30 -- \
  python tools/developer/native_direct_cpp_probe.py \
  /private/tmp/native-direct-zg --points 129 --stride 136 \
  --samples 7 --repeats 2000
```

## Producer work still required

The loader ABI can be shared by C++ and ASM. The producer implementation cannot
be shared completely with today's exporters:

1. **C++:** invoke the validated producer for every retained compiled leaf from
   semantic stage lowering; expand the fail-closed instruction subset for any
   real process that needs more than add/multiply/assign/reciprocal; publish
   only these direct-only libraries in production artifacts.
2. **ASM:** the current inline-ASM exporter also targets the scalar-row ABI. A
   genuine direct implementation needs upstream Symbolica support for the
   plane-loop ABI (including odd tails and target feature metadata), or a
   separate architecture-specific direct kernel emitter. A wrapper repeatedly
   invoking the scalar ASM function does not qualify.
3. Production artifact manifests must carry the direct ABI, state digest,
   target/CPU metadata, scratch bound, and semantic bindings. Runtime loading
   must select the direct symbol without retaining a packet fallback.
4. The production compiler path must prove that custom headers, allowlisted
   flags, target features, and C++ exception settings exactly match ordinary
   C++ generation.
5. Real retained compiled artifacts on AArch64 and x86-64 must prove parity,
   zero traffic/allocation, and end-to-end gain before cutover. The current
   evidence is AArch64 only.
6. The Rust compiled engine must bind this produced symbol to its authoritative
   arena and run the complete selector/tail/API matrix. The developer probe is
   a leaf-level validation boundary, not production wiring.

The prototype deliberately leaves public APIs, stage fusion, artifact
production, and the production evaluator selection unchanged.
