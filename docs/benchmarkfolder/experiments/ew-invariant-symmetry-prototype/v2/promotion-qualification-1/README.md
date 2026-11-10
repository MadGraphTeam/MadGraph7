# Applied production promotion

The reviewed four DP/QP declaration and matrix-normalization blocks are now
applied to `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc`.
The production comments omit private-prototype wording; the qualified code is
unchanged. All 316 canonical statements in the rendered EW adapter match the
qualified v2 adapter. Removing those four exact insertion blocks recovers the
original production template bytes.

All four tests in `tests.unit_tests.loop.test_tirust_fortran` passed, with no
skips. These compile and execute local recording fixtures; they do not invoke a
real reducer. The added test checks the actual production blocks in DP and QP
against an independent integer-word oracle, including finite unequal pairs,
equal signed zeros, nonfinite entries, rounding modes, sticky flags and invalid
traps. The original tensor, failure-lifecycle, rescue-list and permutation tests
also passed. `unittest.log` is the actual output and `tests.json` records the
command, compiler, source hashes and reaped exit-zero process. The test command
ran once on CPU 12 under the shared nonblocking numerical lock and a 180-second
deadline. The lock was released. Log elapsed time is not a performance claim.

The applied hashes are:

| File | SHA-256 |
| --- | --- |
| `TIRUST_interface.inc` | `4ccd4456d0b28f39185e10645e34b6ed2423cfbe2897c2620712e5f9432f461b` |
| `test_tirust_fortran.py` | `74b02f5d373fa515c440b0d59a3ad1654c7c7fcb15c7707f5c5b8aa00b486d1a` |

`source/` contains the exact applied bytes. `source-comparison.json` records the
actual production render, its context (`AmplitudeReduction=false` for the
qualified EW export), and the statement comparison. `rendered-TIRUST_interface.f`
contains that render. The reviewed patch and comparison helper remain in the
unchanged sibling `promotion-proposal/` directory. Its proposal-stage statements
about work not yet executed describe the earlier stage; this directory records
the completed promotion and tests.

The preceding physical qualification is retained separately in
`../runtime-1/qualification-1/`: one bootstrap and all 100 frozen EW points, all
100 accepted at DP status 219, all six references passing the existing comparator
at both 3e-6 and 1e-8, exact original inputs and formerly accepted DP answers and
accuracy words, and no IEEE-state differences. That comparator retains its
existing small-pole exception. No physical replay was repeated for this
promotion. The component fixture and adapter-build packages also remain frozen.

This is a source promotion with local compiled test evidence and a statement
identity link to the qualified prototype. It does not requalify the historical
11-case campaign or any earlier generated exports. The promoted production
version requires its own freshly registered candidate campaign. Installer and
tool files were outside this change.

`index.json` binds every retained file and the referenced repository evidence.
To check the retained bytes from the repository root without compiling or
executing numerical code:

```sh
source tools/tirust_env.sh
python3 -I -B - <<'PY'
import hashlib, json
from pathlib import Path
root = Path.cwd()
base = root / 'docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/promotion-qualification-1'
index = json.loads((base / 'index.json').read_text())
for home, field in ((base, 'files'), (root, 'repository_bindings')):
    for name, expected in index[field].items():
        data = (home / name).read_bytes()
        assert len(data) == expected['bytes'], name
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
assert set(index['files']) == {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p.name != 'index.json'}
print('Retained promotion files and repository bindings verified.')
PY
```

This inventory check verifies bytes; the retained source-comparison and test
receipts describe the semantic validation actually performed.
