# Admitted raw stability report proposal 3

This additive proposal is based on the current 25-page retrospective report at
commit `ba047d9165`, PDF SHA-256
`45e9a5f24c9cf1bfca73b67370003db522644d4c987cf97c21cf1eedd84b20b0`.
Its exact nine canonical baselines are retained. Canonical files/PDF/ledger,
frozen source6, both prior raw drafts, the retrospective proposal and NinjaRS
report remain unchanged. No native worker, compiler, plot renderer or Typst
has run. This packet reads only root-admitted closed evidence and executes the
exact hash-pinned pure retained verification APIs in temporary roots.

The four-path `candidate.patch` proposes new `campaign_report_appendix_c.py`
plus `campaign_report_refresh.py`, `campaign_report.py` and `campaign-report.typ`.
The separate `README.patch` updates the reproduction path and links from
`README.proposed.md`. `benchmark.typ`, `build_report.py` and `style.typ` need no
change. Current typography, navy/colors, source6 acceptance gates and both
optional retrospective/raw selection fields are preserved.

The exact admitted closure is:

- Data `stability_appendix_c/raw-dp-pilot-5-1/index.json`,
  `ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e`.
- Figures `stability_appendix_c/raw-dp-pilot-5-figures-1/index.json`,
  `b131e6294fb3c6900e6776339948928ae71518092cc99de59a8bb0d39fdabcfd`.
- Plotter `stability_appendix_c/plotter-source-6/index.json`,
  `a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae`.

The data verifier source is `efac5f8debda3513c935e1f2cd24de82ad1e8a6ff8561452fbff4ba3220186e0`;
the figure verifier source is `9f3141c7cc1e0b92daf891a19c624ba23b2bd654643af51bf7b302f18f26e166`.
`admitted-inputs.json` records these exact contracts. The helper accepts only
this one explicitly admitted figure pin. Missing/null preserves the current
report's output and read set, including the retrospective appendix. Unknown
pins fail before checkpoint or raw-evidence reads. No latest-directory,
historical-host or sibling discovery is permitted.

Every indexed physical byte is bound through `Evidence.add`: 266 data payloads,
16 figure payloads, four plotter payloads and their three indices, or 289 paths.
The complete original status/error/native-flag records, raw values/stdout/maps,
physical inputs/cards, lossless journals, point dataset and all four PNG/PDF/JSON
figure sets enter the dossier/ledger closure. Logical gzip aliases stay inside
the retained data verifier. One authenticated payload is copied at a time into
three temporary roots; no duplicate permanent 95 MB evidence archive is added.
The data API `verify.py::verify(root=None)` returns verified/index/summary/scope.
The figure API `figures.py::verify(figure_root, *, data_root, plotter_root)` returns
verified/index/dependencies/cases/artifacts=12, without a summary. It recomputes
all four statistical JSON files without plotting. Both returned proofs must be
finite JSON. The full data summary and complete figure receipt remain in the
report. Existing source/physical-input build-ledger checks and the post-compile
rehash remain intact; the optional helper is bound when selected.

All twenty requests retain 100 intended physical point slots and 300 intended
primary probe outcomes each, with bootstrap return codes separate. There are
1,824 available diagnostics and 176 unavailable outcomes, not 176 measured
point failures. Eighteen lanes have 100 available diagnostics. TIRust QCD Zgg's
request completed but has only its first 24 diagnostics available; the following
76 become unavailable after the point-25 group failure and persistent DP
floating-point latch. Its `1e-3` tail bounds are `[0,0.76]`, with fraction null,
not a measured zero tail. Fifteen active-group failure calls and 228 latched
calls do not establish 76 independent exceptions. TIRust QCD ZZgg failed with
0/100 diagnostics available: tail unknown `[0,1]`, fraction null. Its retained
254 primary unchecked-110 returns and 46 unexecuted primary calls accompany
212 active-group failure calls and 254 latched calls. The diagnostic deadline
and SIGINT cleanup do not provide precise benchmark timing. Nineteen completed
requests are not nineteen accuracy passes. All errors and native flags remain
unchanged in the normalized dossier rows.

The captions distinguish **host forced-DP/card-selected backend** from observed
actual reducer/internal arithmetic precision. Status 110 is unchecked. The
finite/Born three-probe range/absolute-mean metric is internal consistency, not
independent true error; AO2PI cancels algebraically without a bitwise-rescaling
claim. The 100 cut-qualified points provide 1% empirical tail resolution.
Dotted upper bounds account for unavailable outcomes, not measured failure
fractions. Zero lower bounds are not drawable on a log axis; ratios over Ninja
require complete paired populations and a positive Ninja tail, without floors
or pseudocounts. Valid infinite metrics remain distinct from nonfinite failures.
All five configured reducers stay visible. Full return-code, bootstrap, primary
outcome and overlapping group/latch counts have separate tables. Four raw plot
pages and the count tables are intended layout only; final pagination and
legibility require an admitted render and visual review.

The ordinary full11 checkpoint, lifecycle, failed partial stress, incomplete
CMS, precision cohorts and Hua-Sheng's deferred local performance gates remain
exactly unchanged. Neither this pilot nor its root integrity proof promotes
scientific, independent-truth or performance acceptance; delivery stays false.
The existing eight post-rescue retrospective figures remain selected separately.

## Actual private artifacts and checks

`selection.proposed.json` SHA-256
`7970d67ddf68547612edf6c384d1c4928c84cc695a8b7c3949acdbd1426b178b`
(7,460 bytes) adds only the raw figure pin to canonical selection `afa84ff0…`.
`dossier.proposed.json` SHA-256
`b69312c77ef5563eb9f99e7ed145dcef199b49bc507645cf68dd41ae162a0133`
(4,940,803 bytes) binds 10,143 evidence paths. Its complete previous report
content and common file identities equal canonical dossier `0c9a7971…` after
removing only the new raw appendix, changed selection path/file binding and
expected 289-path closure. The selection replacement adds one path and removes
one path. All twenty normalized rows preserve every retained summary field,
error/native flag and histogram; the primary outcome population is exactly
6,000. Independent source/pin/status/artifact/caption reviews passed.

Private collection and exact frozen `campaign_report.validate()` both passed
with `Popen` and `ctypes.CDLL` blocked. The frozen replay binds 10,160 inputs,
including the actual proposed helper sources. These checks do not render or
execute numerical science. `actual-collection.json` records the complete exact
20-row table and file delta; `actual-frozen-validation.json` records the source
bindings. Seventeen invented raw tests pass normally and optimized. The prior
ten retrospective synthetic tests also pass against this successor's refresh.

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3 \
  -p 'test_*.py'
python3 -O -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3 \
  -p 'test_*.py'
git apply --check --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3/candidate.patch
git apply --check --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3/README.patch
```

## Root promotion and fresh canonical collection

Only root should perform these separately reviewed/admitted steps. Apply the
four-file core patch and separate README patch, copy the reviewed selection to
a new append-once path, and collect with the promoted canonical helpers:

```sh
git apply --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3/candidate.patch
git apply --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-raw-pilot-integration-proposal-3/README.patch
source tools/tirust_env.sh
python3 -B - <<'PY_COPY_SELECTION'
from pathlib import Path
import hashlib
root = Path("docs/benchmarkfolder")
raw = (root / "experiments/report-appendix-c-raw-pilot-integration-proposal-3/selection.proposed.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != "7970d67ddf68547612edf6c384d1c4928c84cc695a8b7c3949acdbd1426b178b":
    raise ValueError("Raw-pilot selection differs from its reviewed pin")
with (root / "campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-selection-1.json").open("xb") as stream:
    stream.write(raw)
PY_COPY_SELECTION
python3 -B docs/benchmarkfolder/campaign_report.py \
  --campaign invariant-symmetry-1 \
  --selection campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-selection-1.json \
  --output campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-1.json
```

Both output names must be absent. The fresh canonical dossier will bind the
new selection path and must get its own actual hash; do not copy the private
preview to a different path. The dossier binds scientific inputs; validation
and the build ledger bind the actual promoted report helpers. After root's
separate rendering admission, build, inspect all new plot/table pages and run
matching verify-only:

```sh
source tools/tirust_env.sh
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-1.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier"
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier"
```

The preceding 25-page report remains reproducible at commit `ba047d9165`.
No extra PDF/archive is copied, and no later experiment is selected automatically.
