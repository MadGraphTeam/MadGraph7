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
