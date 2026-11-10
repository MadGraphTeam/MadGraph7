# Private cache-observer build qualification

The root-authorized CPU 12 / one-job / 180-second build completed successfully.
All seven owned children exited with code zero and were reaped: three Fortran
compilations, two links, and two ELF-interpreter dependency listings. No Rust
build or amplitude evaluation ran. Protected input identities were unchanged.

Both executables use the same installed `eb3d67fd…` TIRust DSO, common private
driver and observer objects. Their original numerical objects differ only at
the explicitly instrumented `TIRUST_interface.o`. Each link resolves 59 verified
input files, and each dependency listing resolves the same 15 library hashes.
This establishes compile/link provenance; actual numerical-process mappings,
original/disabled/enabled bit parity, and cache occupancy remain unmeasured.

The complete build manifest, two linker maps, and seven command logs are retained
as deterministic gzip text with raw SHA-256 and byte-count bindings. Executables,
objects, modules and libraries are represented only by hashes. Run `package.py`
without arguments for a read-only portable verification; it does not invoke any
compiler, loader, or worker. The earlier source-only index remains unchanged.
