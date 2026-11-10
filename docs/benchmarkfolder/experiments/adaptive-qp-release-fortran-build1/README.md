# Private adaptive release Fortran link

The single admitted relink completed successfully in 1.5653 seconds under the
120-second/4-GiB bound, with one job on CPU 12 nonexclusively. No CPU was reserved.
This is compilation provenance, not a timing or numerical result. No loader,
Fortran worker or reducer was executed.

The corrected source proposal is `adaptive-qp-release-fortran-qualification-v2`,
index `f85e28784762a70806398ab5b878f93f01c0a1ca51475d960c50036a9ad7995f`.
All five numerical/observer objects and the legacy ABI1 executable are exact
copies of the frozen debug qualification build. The legacy executable retains
its observed `0700` mode and execute permission. No Fortran/C source was
recompiled; the one command changes only the declared private paths and the
adaptive library selection.

The new adaptive executable is
`6625b3088f1b2edb78a9d4cadc2a453e1ff66a47d406e419220a9b96a5545b1c`.
It links the exact private optimized DSO
`c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7`.
The link map records this library and the same five object bytes. The full
command, environment, resource limits, owned-process completion and all
protected before/after hashes are retained in `build.json`; maps are losslessly
compressed. Installed and frozen libraries remain unchanged. Actual runtime
mapping, genuine binary128 transport, optional telemetry and numerical parity
remain the later native qualification's responsibility.

`python -B docs/benchmarkfolder/experiments/adaptive-qp-release-fortran-build1/verify.py --live`
checks the portable inventory, command and object bindings, executable mode,
protected live files and selected library without loading any executable.
Omit `--live` to check retained evidence only. Final precise timing remains
Hua-Sheng's local TODO.
