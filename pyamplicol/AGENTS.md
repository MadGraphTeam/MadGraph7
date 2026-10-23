# Agent policy: keep development and release checks lean

Release and dependency validation must be deliberately minimal. Complexity, cache invalidation, and wall time are costs, not signs of rigor.

- Start with the smallest check that prevents a demonstrated failure. Do not add attestations, secondary manifests, duplicated hashes, pristine-tree proofs, repeated dependency scans, or extra provenance layers merely because they are possible.
- Trust standard package-manager guarantees where they already cover the invariant: pinned Git revisions, lockfiles, archive checksums, and Cargo/pip resolution must not be reimplemented by repository-specific ceremony.
- Keep identities narrowly scoped to inputs that can change the artifact they identify. Python, documentation, tests, reports, and staging metadata must not invalidate a native build; validation settings and provenance-only files must not change runtime artifact identity.
- Check each invariant once, at the cheapest authoritative boundary, and reuse that result. Do not repeat fresh installs, builds, source scans, artifact audits, or platform matrices unless the changed code can affect them.
- Prefer focused local tests plus one authoritative CI run. Do not run duplicate full suites or performance campaigns during ordinary finalization unless the user explicitly asks or a relevant performance-sensitive path changed.
- Backward-compatibility machinery is opt-in, not automatic. This repository does not preserve old internal artifact formats unless the user explicitly requests it.
- Before adding a release/authentication check, state the concrete failure it prevents, why an existing check is insufficient, and its expected runtime/cache cost. If that case cannot be made, do not add it.
- When existing ceremony is encountered, simplify or remove it when safe. Never broaden validation as an incidental part of an unrelated change.

Correctness, dependency integrity, and publishability remain required; the rule is to establish each with the least duplicated work and the smallest maintainable mechanism.
