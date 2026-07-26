<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol performance report: `x86_EPYC`

This directory is the self-contained publication workspace for the
`x86_EPYC` measurement environment. It contains the report's LaTeX sources,
canonical raw JSON measurements, generated table TeX, and the PDF when it has
been compiled. Large evaluator artifacts, worker logs, locks, and coordination
state are deliberately stored outside this tracked directory.

`TABLE_FILLING.md` is the sole authoritative campaign procedure. Follow its
phase ordering, resource limits, user-approval pauses, numerical gates,
visual-review cadence, branch coordination, and publication allowlist. This
README intentionally contains no shortened measurement recipe.

Compile this publication folder on any machine with Python and pdfLaTeX:

```bash
cd docs/performance_reports/x86_EPYC
python3 build_pdf.py
```

From a pyAmpliCol source checkout, regenerate tables and run the cache audit
with:

```bash
python3 docs/performance_reports/x86_EPYC/result_tables.py render --compile
python3 docs/performance_reports/x86_EPYC/result_tables.py audit
```

Create a portable copy, including raw data, TeX, and the reviewed PDF, from a
source checkout with:

```bash
python3 docs/arxiv/result_tables.py export-profile x86_EPYC /absolute/output/path
```

The copied entry point selects this profile automatically. Measurements still
require the exact pyAmpliCol source checkout and authenticated native runtime.
Never commit evaluator artifacts, candidate wheels, prepared models, attempts,
logs, locks, coordination state, page images, or LaTeX auxiliary files.
