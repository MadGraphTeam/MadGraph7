# Native compiled DirectApplication prototype

Status: isolated risk-first prototype; not wired into production.

Base: `95e4f6ef85a0daecaa8d4d19c185a9b44ef169c4`

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
  point-contiguous, factor-free overwrite and no-output-alias semantics;
- fixed arrays of input-plane, scalar-pointer and output-plane descriptors;
- one `point_start, point_count` native call returning an integer status;
- logical bindings resolved and validated once when persistent storage is
  pinned;
- unchecked native plane execution after bounds validation, with no packet
  buffer, gather, scatter, remap or warmed allocation.

The fixture's direct function implements the fused complex multiply/add leaf
against the planes themselves. It does not call the dense function and never
constructs a dense `params/buffer/out` row. The same library also exports the
old dense function solely to provide an exact parity and timing oracle.

The prototype validates metadata/counts, portable C symbol names, input/output
alias rejection, arena/momentum/parameter bounds, nonempty point ranges,
odd/tail point ranges, returned native statuses, and zero warmed allocation.
The target-native producer must mark the C entry point non-throwing and convert
all recoverable failures to status codes; Rust can contain a returned failure
or Rust unwind, but cannot recover from a foreign exception, signal, or process
termination.

## Measurement

macOS AArch64, release mode, 129 points, nine interleaved samples, 10,000
repetitions per sample:

| path | median | MAD | warmed allocations |
|---|---:|---:|---:|
| native plane call | 55 ns/call | 1 ns | 0 |
| gather + scalar native calls + scatter | 343 ns/call | 2 ns | 0 |

The isolated fixture ratio is `6.236x`. This is raw ABI evidence, not an
end-to-end process claim and not evidence that a real large fused expression
will retain the same ratio.

## Producer work still required

The loader ABI can be shared by C++ and ASM. The producer implementation cannot
be shared completely with today's exporters:

1. **C++:** emit each fused stage expression directly inside a point-outer
   function whose leaves and destinations are plane lvalues. The existing
   Symbolica complex exporter only emits the scalar-row function, so this needs
   a plane-aware Symbolica export API or a maintained pyAmpliCol C++ expression
   emitter. Merely adding a C++ wrapper around the existing function is
   forbidden.
2. **ASM:** the current inline-ASM exporter also targets the scalar-row ABI. A
   genuine direct implementation needs upstream Symbolica support for the
   plane-loop ABI (including odd tails and target feature metadata), or a
   separate architecture-specific direct kernel emitter. A wrapper repeatedly
   invoking the scalar ASM function does not qualify.
3. Production artifact manifests must carry the direct ABI and metadata symbol,
   stage lowering must derive the same logical bindings used by the SymJIT
   DirectApplication lane, and runtime loading must select the direct symbol
   without retaining a packet fallback.
4. Real retained compiled artifacts on AArch64 and x86-64 must prove parity,
   zero traffic/allocation, and end-to-end gain before cutover.

The prototype deliberately leaves public APIs, stage fusion, artifact
production, and the production evaluator selection unchanged.
