This package contains the frozen observational design, an explicit target-leg correction, and the separately authorized Born-only original-versus-repaired kinematics diagnostic. No per-group loop trace has been run and no production code, threshold, physical input record or reduction policy has changed.

Read `design-amendment-1/README.md` together with the frozen `design/README.md`: the retained soft leg is 4, so the tiny composite channel is `p1-p4`. The original design's `p1-p5` source routing is correct as a hard-channel example, but its identification as the stress channel was wrong and remains visible in the preserved history.

`born-consistency/README.md` reports the finite Born comparison and its limits. The actual repair changes the Born by about 0.0353%; this measured normalization change alone does not explain the approximately 34.6% deficit in the previously rejected loop double poles. The replay does not bound the loop's sensitivity to the changed kinematics. The original/promoted and repaired coordinates are distinct effective inputs; neither a finite Born nor this diagnostic qualifies a checked physical amplitude.

Run `PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/verify.py` from the MG root to verify the complete portable inventory and available source/provenance bindings without compiling or executing native code.
