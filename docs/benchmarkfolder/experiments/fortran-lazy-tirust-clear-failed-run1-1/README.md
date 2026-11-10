# Closed first preparation failure

The source2 controller exited 1 after 4.61008516093716 seconds. All four
preparation commands succeeded: unchanged worker compilation, integer observer
compilation, unchanged observed linking, and an exact fifteen-provider loader
trace. The first native CutTools worker then exited 0 before READY because
`SETPARA('param_card.dat')` found a copied cwd-local parameter card. The unchanged
MODEL reader assumes a directory prefix when deriving `ParamCardPath`; the bare
filename produced an empty prefix and lookup of `/ident_card.dat`, followed by
its existing STOP. This is a fixture-layout failure.

No amplitude, TIRust context creation, cache clear or plan registration occurred.
No candidate source or candidate binary was constructed. There are no timing,
lifecycle, numerical parity or performance results. No native map was captured;
the successful loader trace is distinct from observing the live amplitude.

All 28 regular work artifacts are inventoried; all 24 nonbinary files are copied
byte for byte. Four build artifacts are bound by size/hash without committing
binaries. All fifteen provider symlinks retain literal/resolved targets and
hashes. Original stdout, command/terminal records, failed worker initialization,
cards and original model reader sources are included. Nothing in the original
run was changed or retried. A later source successor may repair only the fixture
layout and reuse the successful hashed unchanged preparation artifacts.

Portable verification, without reading the original workspace or executing
compilers/native libraries:

```sh
python3 -B verify.py
```
