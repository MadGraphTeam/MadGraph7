# MadLoop NinjaRS benchmark report

[Read the final measured report](benchmark.pdf) or its [Typst source](benchmark.typ).

The final study compares Ninja6 and NinjaRS8 across twelve actual MG7 CLI profiles, including both SM2→4 processes and H/HH/HHH/HHHH. It also reports the separately scoped six-process original-code timing comparison, eight-process normal-rescue correctness, native-DP random stability and installation/precision/software controls.

[Data, exact scopes and offline reproduction](final-report-v1/README.md) are retained with the report. The two on-shell CLI cases keep NA ratios where timing helicity/filter scope changed; mixed and uncertain results are not described as a blanket speedup.

The complete scientific evidence, authority readers and prior partial report remain available in the [immutable measurement archive](https://github.com/MadGraphTeam/MadGraph7/tree/19d67c8605337c176680878a521b3a014176a7fb/docs/benchmarkfolder). This folder retains the requested small PDF, its Typst source, numerical summaries and vector figures. Original evidence paths in its data identify files in that archive, rather than files in the cleaned checkout.

The seven font binaries and compressed raw CLI materialization are stored in the
[immutable report archive](https://github.com/MadGraphTeam/MadGraph7/tree/cc6f6fe6dd344a0a767dcfe52341c147e39a193a/doc/benchmarkfolder).
They are excluded from new source checkouts. To reproduce the PDF, first fetch
these eight files into ignored local paths (3.51 MB total):

```sh
python -B doc/benchmarkfolder/fetch_artifacts.py
python -B doc/benchmarkfolder/final-report-v1/reproduce.py --output .benchmark-work/reproduced-benchmark.pdf
```

The first command requires network access and verifies every file against the
retained size and SHA256 before installing it atomically. Reruns reuse verified
files and refuse to overwrite modified inputs. The second command is unchanged:
after acquisition it runs entirely offline and requires byte-identical PDF
output using Typst 0.15.1. The CLI materialization supports auditing; it is not a
Typst input. Font licensing remains in `final-report-v1/fonts/OFL.txt`. No
download is required to read the PDF or run the NinjaRS regression suite.
