# Closed retrospective SM plot report proposal 1

This additive proposal targets the published 17-page milestone `4d24cebce1`.
It reads only the admitted closed full11 selection and the retrospective figure
capsule at `stability_appendix_c/retrospective-main-closed11-1/index.json`, SHA-256
`1138e7f644aec5a0eda96959b47eb17694344a2ad92caa09b2abee6202e23dab`.
No canonical files, existing snapshots, frozen source6, numerical protocols or
NinjaRS report have changed. No compiler, native worker, adapter, plotter or Typst
has run. The separate future raw-pilot proposal remains unselected and unchanged.

`candidate.patch` changes exactly two core paths: `campaign_report_refresh.py`
and `campaign-report.typ`. The separate `README.patch` proposes only the current
report's documentation update, from `README.proposed.md`. `campaign_report.py`,
`build_report.py`, `benchmark.typ` and `style.typ` need no edit. The existing
Libertinus typography, navy/colors, report layout conventions and all report
gates remain in force. The eight plots are each assigned a separate page; final
page count, legibility and caption layout require the parent's admitted render
and visual review. No rendered 25-page output is claimed by this source packet.

The optional selection key `appendix_c_retrospective` accepts only the exact
admitted capsule and requires the selected full11 checkpoint `8f5f7110…`.
Absent/null leaves the previous read set and result shape unchanged. A new or
unknown pin fails before any main checkpoint reads; there is no latest-directory,
active-run or sibling discovery. All 43 payloads plus the index are authenticated
through `Evidence.add`, including eight PNG/PDF/JSON/point-sidecar sets, the
root's portable proof and comparison summary, and all 2,000 accepted status rows.
The inner figures inventory and proof dependencies are pinned. The report binds
the root-admitted proof; it does not rerun its adapter or underlying numerical
science. Existing `campaign_report.validate()` and the builder transfer every
bound physical path into the build ledger and rehash them after compilation;
an image reference alone is not treated as an evidence binding.

All four cases retain all five candidate reducer streams, with 100 unique saved
points per lane. Accepted statuses, backend/precision aliases, finite coefficients,
accuracy diagnostics and recomputed precision/status/zero histograms are checked.
Full accepted outcomes, comparison summary and portable proof stay in the dossier.
The plots add no independent-truth or performance qualification and modify no
main, lifecycle, incomplete CMS, failed partial stress or delivery gate. The full11
`3e-6` ordinary pass and retained `1e-8` target misses remain distinct.

Fresh captions distinguish accepted post-rescue composite host estimates from
finite disagreement against current candidate CutTools. The ordinary 100-point
streams are not raw-DP probes and are not asserted to satisfy the paper's cuts;
1% empirical tail resolution does not qualify rarer population tails. Estimator
definitions differ, including COLLIER's, so curves do not rank true accuracy.
Exact-zero estimates do not certify digits or infinite precision. In QCD n4,
TIRust has 86 zero estimates alongside 14 DP / 86 QP final acceptances; those zeros
are associated with accepted QP diagnostics. The CutTools reference is not known
truth, and its 100 self-comparisons are zero. Zero tails are omitted from log axes;
ratios over Ninja are masked when its denominator is zero, with no floor or
pseudocount. The paper's `1e-3` marker is not an independently byte-bound host
policy marker in this delivered package.

The private actual `selection.proposed.json` has SHA-256
`afa84ff092e8b98d9b0071fe3c4d462523ef674e24cfdc5f934bcd0d05a8c15c` (7,276 bytes).
It adds only the admitted retrospective pin to selection `04f7f07b…`.
The private `dossier.proposed.json` has SHA-256
`d13c1bef33b87d03c642539c463d1753850a050c61b908c1a0a5a0c227a30a49`
(4,814,865 bytes; 9,854 bound evidence paths). Excluding its changed selection
path/file binding and the new appendix, its complete report content and every
common evidence identity equal the prior `closed11-coverage-2.json` dossier.
The ledger delta is 44 capsule paths plus the new selection replacing the old
selection. Existing delivery and formal performance acceptance remain false.
The private preview uses staged copies of the canonical helpers with only the
proposed refresh source overlaid. It never claims a canonical render. Root should
collect a new canonical dossier after promotion, as documented below.

## Source checks

Run from the repository root:

```sh
source tools/tirust_env.sh
python3 -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1 \
  -p 'test_*.py'
python3 -O -B -m unittest discover \
  -s docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1 \
  -p 'test_*.py'
git apply --check --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1/candidate.patch
git apply --check --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1/README.patch
```

The ten tests use clearly invented in-memory capsules. They cover optional
routing, exact admission, complete figure/status coverage, inventory/hash drift,
missing sidecars, strict boolean/numeric types, accepted-status aliases, diagnostic
histograms, proof dependencies, and unchanged report gates. Actual private
collection and exact frozen replay separately block `Popen` and `ctypes.CDLL`.
Independent source/caption and private dossier audits passed. Canonical baseline
hashes are retained in `canonical-baselines.json`.

## Promotion and fresh collection

Only root may execute these steps after reviewing the proposal. First apply the
two core-file patch and the separate README patch:

```sh
git apply --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1/candidate.patch
git apply --whitespace=error \
  docs/benchmarkfolder/experiments/report-appendix-c-retrospective-integration-proposal-1/README.patch
source tools/tirust_env.sh
python3 -B - <<'PY_COPY_SELECTION'
from pathlib import Path
import hashlib
root = Path("docs/benchmarkfolder")
raw = (root / "experiments/report-appendix-c-retrospective-integration-proposal-1/selection.proposed.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != "afa84ff092e8b98d9b0071fe3c4d462523ef674e24cfdc5f934bcd0d05a8c15c":
    raise ValueError("Retrospective selection differs from its reviewed pin")
with (root / "campaigns/invariant-symmetry-1/report-snapshots/closed11-retrospective-selection-1.json").open("xb") as stream:
    stream.write(raw)
PY_COPY_SELECTION
python3 -B docs/benchmarkfolder/campaign_report.py \
  --campaign invariant-symmetry-1 \
  --selection campaigns/invariant-symmetry-1/report-snapshots/closed11-retrospective-selection-1.json \
  --output campaigns/invariant-symmetry-1/report-snapshots/closed11-retrospective-1.json
```

Both new output names must be absent. The new canonical dossier will differ from
the private preview because it binds the new canonical selection path; record
its actual hash after collection rather than copying the preview under a new
name. The collector binds scientific evidence; `campaign_report.validate()` and
the build ledger bind the promoted report helper source bytes. Root can then
perform the separately admitted render, inspect all eight figure pages and run
matching verify-only:

```sh
source tools/tirust_env.sh
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-retrospective-1.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier"
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier"
```

The preceding 17-page PDF and ledger remain available at commit `4d24cebce1`;
this proposal does not add a duplicate archive. No raw Appendix C pilot or later
stress packet is selected automatically by these commands.
