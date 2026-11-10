# QCD n3 genuine-QP diagnostic: closed build2 static audit

The four serial compile/link commands succeeded in 6.137125 seconds, with
CPU12 affinity and a 4 GiB address-space cap. Root reported terminal session
63632 exited zero; the four retained command receipts independently show zero
returns and reaped children. This packet contains static inspection only: the
diagnostic executable and its loader were not executed during this audit.

The source index is `44039ba25e451910d1ab7c264ac1f397437ce41f965d8e381b539dd9baf44046`.
The executable is `ed6a51fff0312b06a45cd97a409aafa7d2447eef5ae7800579b44c0c904b8474`;
the map is `28baebd869a17b26e08a35f3f331679703c0f80ce0f244f0af46adc2b93401f2`.
Its PT_INTERP matches the donor's pinned glibc 2.40 interpreter. The original
twenty numerical objects, all source dependencies, and linked input bytes were
checked after inspection. No claim is made about historical per-object hashes
before the source proposal froze the current objects.

There are fourteen unique wrapper definitions. Seventeen undefined-symbol
relocations cover all fourteen targets; the final executable has incoming
instructions targeting every wrapper and each wrapper forwards to the real
symbol. The CutTools QP numerator callback address also targets its wrapper.
One additional relocation is defined within `improve_ps.o`: its internal
DP-repair to QP-repair call resolves directly. The separate public QP-repair
call from `loop_matrix.o` is wrapped. Static interception does not demonstrate
runtime traversal, initialized inputs, plain/observed parity, or numerical
agreement; those remain strict gates of the unexecuted run2 protocol.

The complete link map is retained with deterministic, lossless gzip compression
and raw/compressed hashes. Its 81 LOAD records resolve to 55 inputs. Eleven
compiler startup/script/support inputs are explicitly labeled post-link
snapshots, rather than prebound dependencies. The link-time `libquadmath` file
differs from the intended runtime path; the existing exact loader/maps gates
must verify the runtime choice. The original linker warning about a dependency
with no GNU-stack note is retained in `command-4.log`.

`collect.py` records the one-shot static inspection recipe. It invokes only
`nm`, `objdump`, and `readelf` on data files; its output is filtered to the
relevant symbols/edges, with command output hashes retained. Archive members
were extracted only into the ignored private audit directory. No compiled
binary, object, module, or archive is packaged here. `link.map.gz` is compressed
text, not a compiled artifact.

Replay the packet's internal checks without opening workspace binaries:

```sh
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/build2-static/verify.py
```

Static audit PASS is not numerical acceptance. Native run2 remains a separate
root admission under the previously reviewed one-shot protocol and common lock.
