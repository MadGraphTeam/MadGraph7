# Closed continuation verifier

This pure verifier has replayed retained evidence for six closed cases:
`li_gg_h`, `qcd_n2_gg_ttx`, `ew_n2_epem_mupmum`, `mixed_uux_ga`, and
`li_gg_hh`, plus `qcd_n3_uux_zgg`. Their tested checkpoint pins are recorded
in `bindings.json`.

The API is `verify(repo_root, checkpoint_path, checkpoint_sha256)`. Supply all
three arguments explicitly, including the expected checkpoint SHA-256. The
verifier does not discover campaigns, select the latest result, or inspect
active cases. The caller must use a separate strict whitelist of root-admitted
checkpoint paths and expected hashes. The tested-pin list is evidence of the
checks performed, rather than automatic admission of another case.

It checks indexed-file containment, sizes and hashes; uses the copied pure
parser and comparison code to replay the saved 100-point streams; and joins
the retained card, point, configuration, compilation, runtime and cleanup
records. A recorded failure at `3e-6` or the `1e-8` target remains a numerical
failure in the result, rather than becoming an evidence-integrity exception.

These portable checks validate retained records. Complete source bytes and
external generated objects were separately audited on the original host;
those checks cannot be recreated from this capsule alone. The helper does not
itself qualify the full source tree or campaign. Runtime evidence consists of
`ldd`-resolved file hashes, not observations from `/proc/<pid>/maps`.

Native executions are zero, and performance acceptance remains false. FPE
flags and internal retry attempts were not observed. Replaying a closed
record does not establish those observations or qualify unrecorded cases.

Run the bounded pure check with `python3 -B check.py REPOSITORY_ROOT`. It replays
the six explicit tested pins, tests an indexed-files-only temporary copy, and
checks rejection of a wrong checkpoint hash and an escaping payload symlink.
