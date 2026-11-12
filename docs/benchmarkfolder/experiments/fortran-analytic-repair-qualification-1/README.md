# Analytic Fortran repair qualification

This is a portable text record of a private, unapplied change to the generated Fortran momentum repair routine. It retains the exact scientific sources, patches, 136-fixture corpus, qualification source, closed BUILD/REPAIR/AMPLITUDE/TIMING evidence, independent reviews and required parent text. Native executables and objects are excluded; their identities, link maps and runtime provider records are retained.

The candidate is `84e8caf51078c3fb5817582e94e9a6a62f1ad22fc4bf7909f5034fce23dbc31c`; qualification source is `bac02ef904383e8215e40c6e8306f00b0b3ca46f9ea5233e94b4ac0e76e125cf`. The bridge and scientific compiler flags are unchanged. BUILD records 218 commands: two compilations, four links, 204 compiler metadata commands, four symbol listings and four interpreter provider listings.

Two standalone mains run 136 source-labelled fixtures in DP and QP for each arm, giving 544 outer wrapper invocations. The DP wrapper internally calls the MP routine. The fixture labels (106 eligible, 26 local-predicate refusals and four caller refusals) describe source expectations; they are not instrumented fast-branch coverage. DP outputs are word-identical. QP outputs differ in 94 fixture results and 233 component words; retained exact rational checks enforce the documented shell, conservation and closeness limits. Legacy refusals, diagnostics and observed IEEE state agree. The initialized MP model mass is a binary64 lift. The massive-vector input fixture does not qualify a model with massive incoming particles.

The full-amplitude check uses 100 saved physical points and two fresh workers. Their coefficients and accuracy words are identical, with reported status 219. Overall and sole split-order outputs duplicate each point: 200 rows contain 800 coefficient words and 200 accuracy words. Each fresh amplitude or timing worker makes 101 learning calls including bootstrap, followed by 100 exact calls.

Seven alternating timing pairs contain 14 batches, each longer than one second. All 179,200 timed calls have accepted status 219. Numerical records cover only the final 100 points of each batch, not every timed call. The paired median candidate/original ratio is 0.952790310414709 on this shared host. These are descriptive N=1 DP timings; formal baseline/default-chain, eight-process and local-machine runtime acceptance remain open. Cold first-call timers exclude model setup and executable startup and establish no initialization gain.

Unexecuted source1 history is preserved. Scientific source2 adds incoming finite guards. Qualification source2 corrects the ZERO_POINT caller classification after binding the positive initialized model mass; source1 was never run. A zero-byte timing owner-readout artifact records an administrative serialization mistake; the official native data were unchanged and the experiment ran once. The timing review also discloses correction of an overly strict offline raw-map equality assertion: anonymous mappings differ, while each saved snapshot has the exact file closure and PID.

This record makes no independent full-amplitude truth, broad model/backend/stress, software-latch, trap-freedom, production-promotion or formal performance claim. No binaries or native execution are needed for replay. Historical controllers are retained as text and are never invoked by the replayer.

From this directory, with Python 3.11 or newer:

```sh
python3 -B replay.py --index-sha256 <published-index-sha256>
python3 -B test_replay.py
python3 -B -O test_replay.py
```

`index.json` authenticates every physical file and decompressed logical text member. Gzip containers have zero timestamps. Replay uses the frozen parsers and exact comparator definition with external file, process, library, signal, network and mutation effects blocked. Tests cover relocation, altered raw text, rehashed false claims, map/identity corruption and symlinks.
