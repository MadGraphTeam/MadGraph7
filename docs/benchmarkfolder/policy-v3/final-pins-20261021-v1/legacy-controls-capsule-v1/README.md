This curated snapshot preserves the accepted legacy references and actual
rotation/scale controls for the two additional challenging processes. It
contains 171 selected original JSON, text, Fortran/Python source, card, point,
registry, and audit records as 116 content-addressed gzip payloads (5,373,993
compressed bytes). Every payload decompresses to the exact original bytes and
hash. Compiled archives, modules, executables, and workspace caches are not
embedded.

For `g g > t t~ g g [virt=QCD]`, the accepted twenty-point CutTools1, Ninja6,
and Collier7 calibration references retain their actual historical source and
runtime identities and failed parent. The independent audit permits their
numerical reuse; it does not relabel them as current optimized-code timings.
The old NinjaRS800 measurements are excluded.

For `u u~ > e+ e- mu+ mu- [virt=QED]`, fresh CutTools1 and Ninja6 twenty-point
calibrations agree in all 160 Born/Laurent total/split comparisons, with a
maximum difference of 0.000201379 times the fixed tolerance. The actual
Collier7 forty-row stream is complete and finite: nineteen points returned217;
point19 returned410 with estimated accuracy1.54055483808898249e-9 above the
unchanged1e-9 MadLoop target. The parent remains failed, no accepted Collier
record was created, and there were no ordinary timing repetitions. A separate
coefficient-only diagnostic passes all480 three-tool scalar comparisons,
including the rejected row; this does not override its failed stability gate
or establish a native library defect.

Actual Ninja6 controls ran on CPU12 with one thread, full poles, and unchanged
stability settings. Both four-point rotation streams completed calibration and
two commands; all192 invariance comparisons passed, with maximum tolerance
fraction0.002040103. Both eight-call scale sequences also completed calibration
and two commands; all320 initial/restored-scale comparisons were exact. The
intermediate scale factors1,2,0.5,1 and actual coefficient variation were
independently audited. All controllers and registered child groups were reaped;
CPU12 was released. Their diagnostic traversal times are ineligible for final
performance acceptance.

Fresh exports and legacy controls bind numerical source856e, generator77b,
archiveb0d and full runtimea757 through the complete hashes in `manifest.json`.
The four audit receipts are included unchanged. New NinjaRS8 stress evaluations
remain outstanding; this snapshot neither forms a completed four-tool dataset
nor supplies a NinjaRS correctness or performance pass.

`manifest.json` maps each original repository path to its lossless payload.
The retained original records still describe their real source/build/runtime
and parent states. Binary hash bindings remain in those records for rebuilding
and live verification; no binary image is substituted. To verify the curated
payload bytes from the repository root using only Python's standard library:

```python
import gzip, hashlib, json
from pathlib import Path
root = Path.cwd()
folder = root / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/legacy-controls-capsule-v1"
manifest = json.loads((folder / "manifest.json").read_text())
for record in manifest["records"].values():
    payload = root / record["payload"]["path"]
    packed = payload.read_bytes()
    assert hashlib.sha256(packed).hexdigest() == record["payload"]["sha256"]
    data = gzip.decompress(packed)
    assert len(data) == record["original"]["bytes"]
    assert hashlib.sha256(data).hexdigest() == record["original"]["sha256"]
```
