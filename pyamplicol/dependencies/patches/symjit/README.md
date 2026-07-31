# SymJIT 2.22.0 upstream patch

The contributor and release installers apply the ordered patches in
`upstream/` to the immutable `siravan/symjit-crate` 2.22.0 source at revision
`77789ff0f78232b1ea4608aceb397058df50b06d`.

Installation authenticates the archive, its pristine source tree, every patch
byte, forward and reverse applicability, and the configured post-patch tree.
The authoritative order and hashes live in both
`dependencies/contributor-lock.toml` and `dependencies/release-lock.toml`.

The current closure contains one generic patch:

- `0001-Expose-a-stable-raw-P-kernel-plane-descriptor.patch` exposes a
  `#[repr(C)]` pointer-and-length descriptor plus unsafe scalar and SIMD
  P-kernel accessors which preserve SymJIT's existing compiled-function Rust
  ABI. This permits duplicate plane bindings and intentional input/output
  aliases without constructing overlapping Rust mutable references.

The patch does not change the generated kernel body or add application-specific
schedules, output policies, factors, or table concepts. Its public safety
contract and standalone duplicate/alias test are documented in the patch and
summarized in `upstream/README.md`.

The base revision itself supplies P2 actual-row indexing and optional
parameter-scaled output. Neither behavior is implemented or altered by this
patch.
