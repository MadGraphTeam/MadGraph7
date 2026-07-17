# Spenso patch series

This development-only series targets GammaLoop revision
`db79edc84f6a1580decbcc4ede7ea0b1c79d9a08` and its `spenso` crate.

## Patches

1. `0001-respect-symbolica-restricted-mode.patch` keeps every internal
   Symbolica `Atom` reduction sequential when Symbolica has no valid license.
   Spenso's public sequential execution path previously used hidden Rayon
   iterators for sparse symbolic sums; a worker-thread `Atom` operation then
   triggered Symbolica's restricted-mode process abort. Licensed execution
   retains the parallel reductions.

The patch is a candidate for upstreaming. It is not included in a release
wheel or sdist.

## Handoff and regression evidence

The original failure was a process abort in Symbolica restricted mode: the
caller selected Spenso's sequential path, but sparse tensor accumulation still
entered hidden Rayon workers and performed `Atom` operations there. The patch
checks Symbolica's effective license state at each affected reduction,
preserving the existing Rayon implementation for licensed users and using the
same operation sequentially otherwise. It does not change the mathematical
contraction or licensed scheduling.

`tests/integration/test_spenso_restricted_mode.py` runs the restricted-mode
regression through pyAmpliCol. The exact GammaLoop base revision and patch
digest are recorded in `dependencies/contributor-lock.toml`. GammaLoop/Spenso
is contributor-only reference tooling, is not a pyAmpliCol runtime dependency,
and neither this patch nor GammaLoop is included in release wheels or sdists.
