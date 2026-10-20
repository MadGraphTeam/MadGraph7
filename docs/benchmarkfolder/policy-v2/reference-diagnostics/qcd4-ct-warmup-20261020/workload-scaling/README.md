# Static QP workload comparison

This source-only inventory explains why the strict QCD4 CutTools reference can
take much longer than QCD3. It does not inspect the live point index or change the
worker. `analysis.json` binds the generated source files and retained helicity
filters that were read.

| Quantity | QCD3 | QCD4 |
|---|---:|---:|
| Generated helicities | 48 | 144 |
| Active helicities in the retained filter | 24 | 72 |
| QP updater calls per complete helicity pass | 388 | 2,146 |
| Complex products per complete QP point, after literal B-zero skips | 20,186,856 | 595,899,288 |

The final count includes all generated updater ranks and multiplies by active
helicities. Literal zero B entries are discounted only where the original
compact updater actually tests and skips B; the original expanded kernels still
multiply their zero entries. Nonliteral B values may also become zero at runtime.
The inventory excludes ALOHA construction, merging, clearing and reduction.

The resulting static ratio is **29.52**. QCD3's retained CutTools measurement is
1.9393 seconds per point with 52 QP and 48 DP outcomes. Assigning all that time to
its QP points and scaling by the static ratio gives an illustrative 110 seconds
per all-QP QCD4 point, or about 3.1 hours for 100 such points, before startup.
This is neither a measured QCD4 time nor a bound or completion forecast: the
rescue fraction, dynamic zeros, initialization and non-updater costs differ.
It establishes that a 7,200-second protocol timeout can be too short for a
legitimate full warm pass.

`analyze.py` reproduces the inventory from the retained generated outputs and
refuses to overwrite `analysis.json`. No amplitude evaluation, build, debugger
or additional profiler was used.
