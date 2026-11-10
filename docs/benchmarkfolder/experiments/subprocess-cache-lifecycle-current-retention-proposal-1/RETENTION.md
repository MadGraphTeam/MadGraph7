# Closed current-source lifecycle retention

This is the offline producer/test history for `../subprocess-cache-lifecycle-current-qualification-2/`. It contains no compiler, loader or reducer execution. The scientific stages were executed once by the parent coordinator and closed before capture.

`capture.py --execute` captured the closed proof. Its initial index `b3eea1677e6b6c988ce4fbb0daf448e2176cccc88bff7fe15794e29bdbce74c6` and exact verifier/summary remain in `provisional-first-capture/`. The independent verifier review requested explicit original-file and generated-source bindings, plus a global UTF-8/NUL-free rule. `finalize.py --execute` added those retention checks and preserved the provisional packaging outputs under the final packet's `retention-history/`. It changed no scientific logs, inputs, results or build files.

The first test run had one test-fixture error: it treated compressed `results.json` as a directly indexed file. Its source/log remain `test_verify.attempt1.py` and `tests-attempt1.log`. The corrected test removes the real gzip index entry; all 14 tests pass. These tests cover actual raw derivation, strict summary, indexed-only relocation, missing inputs, gzip corruption, path traversal, accidental binary payloads, generated source coverage/content, original logs, incomplete-sum W03 warnings, unexpected mapped libraries, scope promotion, source substitution and the unchanged report loader.

The finalized proof index is `212b6046a48030f26e12facfbd85b4179df0036633f14bbbc38c2788df376964`. Its verifier is `f972cf33e455df9ca2d2bc24e9bed50c6ba582b59dd41c099c92fa2f4d94122a`. The unchanged report `validate_candidate()` independently passes all 2,664 registered production entries and the installed native identities. Report ingestion remains a coordinator action.

Replay only, from the repository root after `source tools/tirust_env.sh`:

```sh
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-qualification-2/verify_evidence.py
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-retention-proposal-1/test_verify.py
```

The producers deliberately refuse re-execution over the already finalized proof. Native binaries and historical symlink targets remain metadata only. The failed first preparation remains at `../subprocess-cache-lifecycle-current-failed-preparation-1/`, index `22deeb09e727fb4f687e3c78c2bc215afd86cca81a933b74cf259597453c296b`.
