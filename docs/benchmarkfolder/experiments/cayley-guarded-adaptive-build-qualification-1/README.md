# Guarded adaptive private release build: closed compilation evidence

The private release build completed successfully. Cargo compiled the guarded
TIRust core, adaptive bridge, scalar-master adapter and pole-kernel crate
anew (`fresh=false`, optimization level 3, no test profile). It reused 140
unchanged dependency artifacts. All 1,193 protected input hashes match before
and after the build. These are compilation and source-provenance findings.
The new shared library has not been loaded, installed or numerically exercised.
Precision, physical accuracy, stability, floating-point trap behavior and
runtime performance remain unqualified by this package.

The recorded executable banners are Cargo 1.97.0 and rustc 1.97.1. The
Nix Cargo package directory is labelled 1.97.1; the literal executable banner
is preserved. An initial portable-verifier assertion confused these labels;
its failure, source and initial inventory remain in `retention-failure-1/`.
Correcting this metadata assertion required no new build or DSO read.

The shared library is retained only by its hash and pure ELF metadata:

- SHA256: `69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b`.
- Size: 69,162,336 bytes; ELF64 little-endian x86-64 shared object.
- Native binaries, object files and compiler modules are excluded from this package.

The successful source proposal is copied byte-for-byte under `proposal3/`.
Its source-index SHA256 is
`39d6ba2bd3c7aef5c8f5008360661c988a93d86ab1265cb3bc815707233b9e72`.
The successful execution receipt has SHA256
`59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632`;
the uncompressed Cargo log has SHA256
`7ea452558263c98b91781e979f843c8cf236d951b89450cbbc436275945e8a3d`.
The copied plan and receipt preserve the complete environment, compiler/tool
hashes, lockfile identities, dependency pins and original cache-seed provenance.
They also preserve both earlier failures: LLVM allocation failure under a
4 GiB address-space limit, then the separate 600-second build timeout.
Their original incomplete receipts remain unchanged; later read/hash-only
closures establish 1,163 and 1,186 unchanged inputs respectively.

For this successful attempt the original release cache was copied into its
own namespace. No failed guarded-build artifact was reused. Cargo ran offline
with the existing lockfile, one job, a nonexclusive CPU11 affinity and an
8 GiB virtual-address-space limit per build process. Cache copying had a
4 GiB limit. The single attempt had a 1,800-second deadline, including source
preflight, copying, compilation, output hashing, input closure and publication.
Both child commands exited zero and were reaped. The Cargo process took
593.614 seconds and the whole attempt took 608.861 seconds. These administrative
build durations are not reducer or amplitude performance measurements.
All 41 compiler warning diagnostics remain in the full Cargo log.

The private source preserves all 499 source-file identities. Focused UTF-8
archives additionally retain 133 code and license files: bridge/core/master/
pole-kernel source trees, their manifests, both lockfiles, the workspace
manifest and toolchain file, and the ABI header. Deterministic independently
compressed text pieces stay below 1 MiB each. Unrelated examples, reports,
render assets, fixtures and unused workspace crates remain explicitly hash-only.
The original and guarded Gram source permit an offline exact comparison:
only the previously reviewed signed-zero denominator guard differs. Cargo
manifests, lockfiles and NinjaRS dependency pins are unchanged. This source
comparison does not establish the guard's numerical or trap behavior in the
complete library.

`elf_metadata.py` is the unchanged existing pure byte reader, authenticated
against its retained origin index. Retention read the new DSO once to obtain
its hash, literal dynamic dependencies, version requirements and 13 exported
`ml_tirust_*` symbols. Metadata describes the file; it does not establish
runtime loader resolution, ABI calls or precision behavior.

The portable verifier reads only this folder's text and UTF-8 gzip payloads:

```sh
python3 -I -B verify.py
```

It checks the inventory, re-parses Cargo output, verifies the complete
before/after closure, preserves both failed-attempt classifications, and
reconstructs the retained code for the sole-source-change comparison.
Historical native paths in receipts are provenance strings; they need not
exist. Copying this folder elsewhere preserves verification. The ELF reader
is retained for provenance and is never invoked by the portable verifier.
`retention-script.py` records the one-time packaging procedure and requires
the original private build tree; it is not part of the portable verification
command. No precision, physical, stability or performance acceptance follows
from a passing verifier.
