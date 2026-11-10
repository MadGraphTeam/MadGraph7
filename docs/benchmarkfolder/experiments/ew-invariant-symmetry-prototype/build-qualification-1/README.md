# EW adapter compile/link evidence

The private build completed on CPU 12 with one job and a 180-second cap. Its three
owned commands—one adapter compilation, one link and one dependency listing—all
returned zero and were reaped. All 899 selected protected inputs remained
unchanged. The original build proposal and canonical source index remain frozen.

Only `TIRUST_interface.o` was replaced. The qualified raw-word `check_sa.o`,
`host_observer.o` and `observer_noop.o` were reused unchanged, as were the other
numerical objects and archive inputs. The actual linker LOAD closure passed its
one-object substitution check. The dependency listing matched the original
closure, including the installed `eb3d67fd…` bridge. Link-time inputs and loaded
dependency resolution remain separately recorded.

This is **build qualification only**. No amplitude was run. Numerical parity,
live numerical runtime mappings, compiled IEEE/input behavior and physics
qualification remain pending. There is no timing or performance conclusion.

The package retains exact source, derivation patch, build plan/controller,
commands, three logs, actual linker map and completed manifest as deterministic
gzip text (empty filename, mtime zero, level six). Every payload records its raw
SHA-256/size and compressed SHA-256/size. Executables, objects, libraries and
caches are identified in metadata but are not copied. The original cause-build
package and six-arm numerical top record are referenced through their existing
portable archives and immutable repository indexes.

`package.py --package` creates this folder's index once and checks the current
protected inputs before and after copying. Default `package.py` verifies only
retained text and repository-relative source/archive bindings, so it does not
need `.tirust-build`, installed libraries or binaries. Historical absolute paths
remain record identifiers and are not dereferenced by the default verifier.
Verification does not recompile, relink, load a library or replay a numerical
parser. Partial packaging failures are retained without automatic retries.

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/build-qualification-1/package.py
```

The completed build manifest has SHA-256
`971e185d42eaf2513ac16849e8b481e0eedda34f034b9a81b043f28e4ba131c3`.
Its candidate executable is `c4e28fa35ac464f5234f4d73a4685e85346b90a93b2bcca942ece9880dc9e83e`;
the new adapter object is `a9de994ab844d6532c4a69eeb71539ba220b54a48dd348e09a9ea63712d437fe`.
