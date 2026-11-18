# MadLoop NinjaRS benchmark report

The report, Typst source, measurements, figures, reproduction scripts and
optional Nix development shell are distributed in the
[benchmark release](https://github.com/alphal00p/ninjars/releases/tag/madloop-ninjars-benchmarks-v1).
The [PDF](https://github.com/alphal00p/ninjars/releases/download/madloop-ninjars-benchmarks-v1/benchmark.pdf)
is available separately for reading.

Download
[`madloop-ninjars-benchmark-9a9b28d62ef2.tar.gz`](https://github.com/alphal00p/ninjars/releases/download/madloop-ninjars-benchmarks-v1/madloop-ninjars-benchmark-9a9b28d62ef2.tar.gz).
Its SHA256 is:

```text
524cf05598652253a60b5e57525e989f748c447e83cedc588acb2b19cd2b8d59
```

After verifying the download, extract it and open
`doc/benchmarkfolder/benchmark.pdf` inside the extracted directory. The bundle
preserves the report and development shell from source revision
[`9a9b28d62ef2`](https://github.com/MadGraphTeam/MadGraph7/commit/9a9b28d62ef20b0687770f76e7d1dc2fc517dddf).
Each measurement retains its original dependency revisions and timing scope.

From the extracted directory, verify its contents and run the standalone
acquisition tests:

```sh
sha256sum -c SHA256SUMS
python3 -m unittest discover -s tests -p test_report_artifacts.py
```

The reproduction instructions remain in `doc/benchmarkfolder/README.md`.
They acquire checksum-pinned optional fonts and raw inputs once; reproduction
then runs offline with Typst 0.15.1:

```sh
python3 -B doc/benchmarkfolder/fetch_artifacts.py
python3 -B doc/benchmarkfolder/final-report-v1/reproduce.py --output benchmark-reproduced.pdf
```

The [complete measurement archive](https://github.com/MadGraphTeam/MadGraph7/tree/19d67c8605337c176680878a521b3a014176a7fb/docs/benchmarkfolder)
remains available at its immutable revision. The bundle's
`RELEASE-PROVENANCE.json` records source hashes, licensing and the relocation of
the acquisition tests. Source relocation did not recompile the PDF.
