# TIRust (MadLoop backend 9) benchmarks and qualification

Start with [`CURRENT_STATUS.md`](CURRENT_STATUS.md). It names the current
campaign, its results and the open acceptance items. The user guide is
[`docs/tirust.md`](../tirust.md), and the overall handoff is
[`HANDOFF.md`](../../HANDOFF.md).

## Layout

| Path | Content |
|---|---|
| [`benchmark.pdf`](benchmark.pdf) | Report: 48 preserved pages plus the current-source pages |
| [`campaigns/tirust-9a781998-1/`](campaigns/tirust-9a781998-1/) | Current campaign. Registration (manifest, frozen inputs, points), [`summaries/`](campaigns/tirust-9a781998-1/summaries/), [`stability/`](campaigns/tirust-9a781998-1/stability/), report sources and the [local timing TODO](campaigns/tirust-9a781998-1/timing-handoff-1/LOCAL_TIMING_TODO.md) |
| [`campaigns/entry-latch-cache-guard-1/`](campaigns/entry-latch-cache-guard-1/) | Measured predecessor; its physical cards and frozen inputs are inherited |
| `campaigns/{invariant-symmetry-1,split-dp-terminal-1}/`, `runs/qcd_n1_uux_z/tirust-first-attempt-3/` | Only the ancestor manifests, summary snapshots and retained failure that campaign verification checks |
| [`cases.json`](cases.json), [`points/`](points/), [`inputs/`](inputs/) | The 11 processes, frozen 100-point streams and stress points, baseline and model pins |
| [`qualification/comparison-backends/`](qualification/comparison-backends/README.md) | How the CutTools/Ninja/COLLIER/NinjaRS comparison libraries are installed |
| [`history/report-before-current-source-1/`](history/report-before-current-source-1/) | The 48-page report the current pages are appended to |
| `experiments/{complex-mass/points,…}`, `stability_appendix_c/raw-dp-pilot-5-1/points` | Fixtures still read by tools and tests (CMS stress points, Appendix-C sample points) |
| [`PLAN.md`](PLAN.md), `PLAN-AMENDMENT-*.md`, [`POLES.md`](POLES.md), [`OEM_SIGNING.md`](OEM_SIGNING.md) | Goals, agreed amendments and feature notes |

Raw lane evidence (full residual lists, worker logs, call traces) is not
committed. It stays in the ignored `.tirust-build/` and `.benchmark-work/` of
the producing host and can be reproduced with the commands below.

## Reproduce

Run from the MadGraph root with `source tools/tirust_env.sh` (outside
`nix develop`) and `FFLAGS='-w -fPIC -O3'`:

```bash
REASON='…' tools/tirust_requalify.sh tirust-<pin8>-1     # install … stability, provenance, summaries
python3 -B tools/tirust_current_report.py render --campaign tirust-<pin8>-1 [--publish]
python3 -B tools/tirust_current_report.py verify --campaign tirust-9a781998-1
```

The individual tools (`tools/tirust_benchmark*.py`, `tools/tirust_capture/`,
`tools/tirust_stress.py`, `tools/tirust_stability.py`,
`tools/tirust_provenance.py`) are documented in their headers. Precise timing
acceptance is performed locally ([`LOCAL_FINAL_VALIDATION_TODO.md`](LOCAL_FINAL_VALIDATION_TODO.md)).

## History

Earlier campaigns, the audit-style evidence capsules (~52,800 files), the
stability pilots and the earlier report builders are preserved at the tag
[`madloop_tirust-evidence-2026-11-13`](https://github.com/MadGraphTeam/MadGraph7/tree/madloop_tirust-evidence-2026-11-13/docs/benchmarkfolder).
Historical notes inside the predecessor campaign link to paths that exist only
at that tag.
