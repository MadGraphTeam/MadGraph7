# Raw P-kernel plane-descriptor patch

Base: `siravan/symjit-crate` 2.22.0, commit
`77789ff0f78232b1ea4608aceb397058df50b06d`.

Apply
`0001-Expose-a-stable-raw-P-kernel-plane-descriptor.patch`.

## Generic rationale

The existing indirect P-kernel function type receives an array of Rust mutable
slice fat pointers. Constructing that array is unsound when a consumer needs
the same plane in more than one input slot or intentionally binds one plane as
both input and output, because it would create overlapping `&mut` references.
Those layouts are valid for a column-oriented kernel and are useful to any
arena consumer, independent of its scheduling model.

The patch exposes the same machine-code entry point through:

- a `#[repr(C)] PlaneDescriptor<T>` with a raw data pointer and length;
- an unsafe `CompiledPlaneFunc<T>` callable which preserves the existing
  SymJIT compiled-function Rust ABI;
- scalar and SIMD accessors on `Applet` which are available only for indirect
  direct-arena P-kernels.

The caller must keep the `Applet`, executable mapping, descriptors, parameters,
and backing allocations alive; validate lengths, alignment, and kernel ranges;
and synchronize all accesses. The API deliberately remains unsafe because
generated code cannot validate those conditions.

The pinned base already defines the P2 index as an actual row number and
implements parameter-scaled output when identity output is disabled. This
patch preserves both upstream semantics.

No prologue, body, epilogue, operation, or serialization format changes. The
standalone `kernels.rs` case binds one allocation into duplicate input slots
and an exact output alias, checks both SIMD row-indexed and scalar
invocations, and confirms that ordinary non-arena applications do not expose
raw-plane accessors.
