# Current-source relocated deployment: build-only evidence

The two actual MG CLI exports and all requested builds passed at revision
`0fcd2cd873d61d134837e1d9e87afb13a47824fd`, exporter inventory `afb6e0be…`.
The selected standalone and FKS/NLO virtual diagnostic executables link against
the relocated TIRust DSO `eb3d67fd…` and the unchanged NinjaRS static archive
`0edfa89d…` (298,293,528 bytes). **No amplitude or Monte Carlo evaluation was
performed in this packet.** Runtime qualification is deferred.

The original sealed proposal is retained byte for byte in `proposal/` (seal
`3b8ffa3555351ce73e98d1103514944bce6d1e9eff4c6d519b609526649782bd`).
`build/build-result.json` is the exact closed receipt
`8b54b3ecd0e8daf7f670fc883b26913ccde810dc39a3d6c819cba975c9ffbd58`;
`build/build-admitted.json` is the exact artifact receipt
`75ab86f7a9acc9c0a0359b3aa2a7c9293a8d36ef6fd74f8f1768cfe8970da49b`.
Their 15 stage records comprise 12 successful subprocess stages and three pure
validation/move stages. All observed owned children were cleaned up. The receipt
records production configurations, source and installed dependencies unchanged.

The actual commands, resource limits and compiler paths are in
`proposal/commands.json` and the individual stage receipts. Console logs are
retained with their original bytes under the portable `.txt` extension. The CLI
generated `loop_sm` `u u~ > z [virt=QCD]` standalone output and
`u u~ > z [QCD]` FKS output. Each used an isolated configuration, explicit
`--no_save` settings and internal dependencies. After copied artifacts were
verified, only the private copied dependency prefix was removed and both outputs
were moved. Normal generated makefiles then built the standalone `check`, NLO
CutTools/model/DHELAS libraries, NLO virtual library and NLO diagnostic `check`.
The NLO diagnostic link uses the retained `adapter-check.mk` extension, which
includes the generated makefile and links its normal virtual library.
The `check` make target compiled executables; neither executable was launched.

This is a backends 1/8/9 deployment check. The optional IREGI directory was
deliberately omitted from the private source deployment and inventoried separately,
because its historical makefile can use inherited `PWD` to reach the original
vendor directory. No original vendor or installation was modified. Internal
Ninja/COLLIER availability and full Monte Carlo output hermeticity are outside
this check. Inventory link spellings are retained; a null file hash can also
describe a directory symlink and must not be read as proof of a dangling link.

The retained `readelf -d` logs show `$ORIGIN` search paths, and the `ldd` logs
resolve the relocated local TIRust DSO plus pinned Nix providers. These are
diagnostic resolution records, **not runtime initialization evidence**. In this
environment `ldd` substitutes its own interpreter, so the executables' actual
`PT_INTERP` requires a separate additive runtime admission binding. The original
proposal and build receipts are not rewritten to add that later observation.
`provider-spellings.json` separately records a pure post-build inspection of
symlink aliases printed by `ldd`, checked against the closed provider hashes.

`generated/` preserves selected exact Fortran sources, cards, makefiles and
dependency metadata. The runtime card locations are standalone
`Cards/MadLoopParams.dat` and NLO `SubProcesses/MadLoopParams.dat`; both initial
cards hash to `e782503a…`. All 1,871 bound artifact path hashes, both executable
hashes and full export link inventories remain in the artifact receipt. These are
path entries, not counts of unique retained binaries. Native binaries, objects,
modules and the full source archive are omitted. `origins.json` maps retained
copies to original locations. Absolute paths are provenance; portable replay
does not open them.

Run from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-deployment-qualification-1/verify.py
```

This verifies retained bytes, source seals, log/receipt consistency and recorded
copy/link identities only. It does not rehash omitted artifacts or imply a new
build. On the original host, the following additionally rehashes the external
build artifacts and recorded providers without loading or executing them:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-deployment-qualification-1/verify.py \
  --external-build-root .tirust-build/current-source-deployment-1 \
  --external-providers
```

The verifier reports these scopes separately. Diagnostic elapsed times do not
establish performance acceptance. The sealed build controller is historical
execution provenance and must not be rerun in the already-used directory.
