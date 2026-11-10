# V2 adapter compile/link evidence

The private v2 build completed on CPU 12 with one job and a 180-second cap.
Its three owned commands—one adapter compilation, one link and one dependency
listing—all returned zero and were reaped. All 899 selected protected inputs
remained unchanged. The frozen v2 source index and build proposal remain intact.

Only `TIRUST_interface.o` was replaced. The qualified raw-word `check_sa.o`,
`host_observer.o` and `observer_noop.o` were reused unchanged, along with the
remaining numerical objects and archive inputs. The actual linker LOAD closure
passed its one-object substitution check. The dependency listing matched the
original closure, including the installed `eb3d67fd…` bridge. Link-time inputs
and dependency resolution are recorded separately.

This package establishes **build qualification only**. This build invoked no
amplitude. Its record leaves numerical parity, live numerical runtime mappings,
compiled IEEE/input behavior and physics qualification pending. Separate
fixture or runtime evidence is outside this package. No timing or performance
conclusion follows from the build.

The package retains the exact completed manifest, commands, three logs, linker
map, candidate source, build plan/controller, derivation patch, v1-to-v2 patch
and controller delta as deterministic gzip text (empty filename, mtime zero,
level six). Payload metadata records original and compressed hashes and sizes.
Executables, objects, libraries, core dumps and caches are not copied. The
prior cause-build and six-arm numerical records use their existing portable
archives and immutable repository indexes; v1 evidence remains unchanged.

`package.py --package` retains this completed attempt once and checks protected
inputs before and after copying. Default `package.py` verifies retained bytes,
build-record joins and repository-relative source/archive bindings. It needs
neither `.tirust-build` nor installed libraries or binaries. Historical absolute
paths remain record identifiers and are not dereferenced by default verification.
Neither mode invokes a compiler, linker, loader or numerical workload. Partial
packaging failures remain on disk without automatic retries.

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/build-qualification-1/package.py
```

Completed build manifest SHA-256:
`bd0b33f69a552851feac60e61573442bfefad5c17f45dd9713d40d6a9ac6094a`.
Candidate executable:
`542452c183adcaee341b569f83cf963cf768a29fccb251ba5a0f9b9a75085674`.
Replacement adapter object:
`e88c135942a8d5761ee67161d3618325a4735fd10971776503fa30443a368fe4`.
