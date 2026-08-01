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

Create a fresh, empty campaign from any installed wheel with:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
```

All pyAmpliCol backends use installed resources. Measuring the optional
original-AmpliCol reference additionally requires
`--original-amplicol PATH_TO_COMPLETE_CHECKOUT`; without that checkout only
that backend is reported unavailable. pyAmpliCol and the supported patched
comparison checkout have no LHAPDF dependency.
