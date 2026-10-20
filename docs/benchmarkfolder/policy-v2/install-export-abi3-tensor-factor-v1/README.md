# Relocated internal-copy validation after the QP factor port

The actual CLI exported `u u~ > z g g [QCD]` with internal dependencies from
committed numerical source `f4bffc61ed67e5d083b80ff9901768d4d5779989` on CPU2.
It selected all 136 private QP factor call sites. Each affected coefficient
batch has its IEEE imports and fresh entry-flag query. Generation took 184.15 s
and the relocated subprocess build 17.18 s; these are diagnostic wall times.

The test copied the existing installed ABI3 prefix to its own private prefix,
exported against that copy, relocated the process output, then renamed only the
private prefix before compiling and evaluating the relocated output. The main
installation remained present and byte-identical throughout. There was no
installer or Cargo rebuild. The exported archive, module, Fortran source,
header and three metadata files are regular internal copies with matching
hashes. The link recipe and executable imports retain the versioned GNU
quadruple-precision runtime. Legacy Ninja is absent in this internal export,
as recorded by the availability array; NinjaRS is available in slot 8.

All three numerical controls completed successfully after the MP slot was
explicitly handed over:

| Control | Return code | Result |
|---|---:|---|
| Forced DP, mode 1 | 110 | Pass; unchecked numerical control |
| Forced genuine QP, mode 4 | 120 | Pass; unchecked numerical control |
| Automatic stability, freshly forced check | 218 | Pass; checked NinjaRS DP |

The diagnostic driver reads the same retained first QCD3 point on each of its
20 calls; it does **not** cover 20 distinct points. It records the last Born,
finite, single-pole and double-pole result for both the total and physical order.
All 48 pairwise coefficient comparisons among these three outputs and the bound
Ninja6 reference pass `1e-7 * max(abs(x),abs(y)) + 1e-10 * point_scale`.
The forced-QP control proves that the relocated archive can execute genuine MP;
it does not establish stability-checked QP rescue or a count of accepted factor
helper calls. These controls are not final benchmark acceptance or timing data.

`prepared.json` retains the pre-generation tree identity, full generated-source
manifest, executable hash, regular-copy hashes, call-site counts and build
recipes. `runtime-results.json`, the three exact MadLoop cards and raw logs
retain the numerical evidence. Key generated Fortran and copied interface
assets are under `inputs/`; the 299 MB archive and executable remain in the
isolated work directory and are represented by their exact hashes here.
The globally ignored `.log` files are intentional evidence and belong in the
commit. Python bytecode caches are excluded from the seal.

Offline verification (no compiler or amplitude launch):

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/policy-v2/install-export-abi3-tensor-factor-v1/verify.py
python docs/benchmarkfolder/policy-v2/install-export-abi3-tensor-factor-v1/verify.py --live
```

The second command additionally rehashes the live relocated sources and
executable, hidden private prefix and unchanged main installation. Both commands
are read-only and do not rewrite retained results. The original `validate.py`
commands refuse to overwrite this completed namespace.
