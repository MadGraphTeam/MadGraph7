<!-- SPDX-License-Identifier: 0BSD -->
# Architecture Performance Reports

The repository retains four rendered snapshots from separate manual
measurement campaigns:

- [Consolidated report](../arxiv/pyAmpliCol.pdf)
- [MacBook M3 report](macbook_M3/pyAmpliCol.pdf)
- [MacBook M3 Z-process subset](macbook_M3/z_table/z_table.pdf)
- [x86 EPYC report](x86_EPYC/pyAmpliCol.pdf)

These PDFs are historical measurements, not release-CI results. Raw JSON,
generated TeX, build workspaces, attempts, logs, locks, and coordination state
are not kept in the source tree.

Create or reset a self-contained campaign from any installed wheel with:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
```

Runtime attempts, artifacts, locks, and coordination state live visibly in
`pyamplicol-profiling-campaign/campaign_artifacts/`. Moving or renaming the
whole campaign directory moves that state with it; campaigns with the same
basename in different parents remain independent. Historical repository-global
`.artifacts` state is intentionally ignored.

`--force` replaces the managed template files and resets only
`campaign_artifacts/`, the managed PDF and summary IDs, measurement lineage,
and known LaTeX byproducts. It preserves unrelated destination files and the
recorded local-AmpliCol checkout. Stop an active campaign before resetting it;
an active controller makes the command fail without removing state.

All pyAmpliCol backends use installed resources. Measuring the optional
original-AmpliCol reference additionally requires
`--original-amplicol PATH_TO_COMPLETE_CHECKOUT`; without that checkout only
that backend is reported unavailable. pyAmpliCol and the supported patched
comparison checkout have no LHAPDF dependency.
