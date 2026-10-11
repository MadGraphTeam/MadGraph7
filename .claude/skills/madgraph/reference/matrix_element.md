# The matrix element at given phase-space points

Use this when the user wants |M|² at momenta they choose: one point to
check, a list of points, or millions of points from their own code (a
reweighting, an analysis, an ML training set). There is no `launch` here:
generate an output, compile it, and call it.

## Choosing the route

| Need | Output | Call |
|---|---|---|
| one or a few points from Python, any α_s, single helicities | `output standalone_fortran DIR --prefix=int` | `all_matrix2py.smatrixhel(...)`, one call per point |
| many points at once (numpy arrays), GPU | `output DIR` (mg7) | madspace `MatrixElement` on a batch |
| C++ only, or a file of points | `output standalone DIR` | `check_sa.exe perf --momenta ... --dump-me ...` |

Measured cost per point (Apple M5 Pro, 18 cores, double precision):

| | g g > t t~ | g g > t t~ g |
|---|---|---|
| Fortran module, one Python call per point | 4.7 µs | 35 µs |
| madspace, SIMD build, all cores, batches of 1e5 | | 0.43 µs |
| madspace, scalar build, all cores, batches of 1e5 | 0.14 µs | 0.72 µs |
| madspace, scalar build, 1 thread | 1.4 µs | 9.5 µs |
| madspace, a batch of 1 point | 80-100 µs (call overhead) | |
| C++ `check_sa.exe`, simd_128, one thread | 1 µs | |

The cost grows quickly with the number of legs. Time a small batch first,
then extrapolate.

- **Batch the madspace calls**: each call costs about 100 µs of overhead,
  so evaluate points in arrays of thousands, not one at a time. For a single
  point, the Fortran module is the quickest and simplest.
- **Agreement**: in double precision the routes agree to about 1e-13,
  occasionally up to 1e-10 for SIMD builds on 2 → 3 (checked on e+ e- >
  mu+ mu-, e+ e- > a a, u u~ > t t~, g g > t t~, d d~/u u~ > e+ e-,
  g g > t t~ g, and against analytic results).

## Conventions (identical for the three routes)

- **What is returned**: |M|² summed over final-state helicities and colours,
  **averaged** over initial-state helicities and colours, and **divided by
  the identical-particle factor** of the final state. e+ e- > a a gives
  exactly ½ of the textbook 2e⁴(u/t + t/u). To compare with a textbook
  |M̄|², multiply by n! for n identical final-state particles. In a
  cross-section code, do **not** apply the factor again. Units:
  GeV^(8-2N) for N external legs: dimensionless for 2 → 2, GeV⁻² for
  2 → 3.
- **Momenta**: (E, px, py, pz) per leg, legs in the order of the process
  definition, incoming first, all as physical momenta (positive energies).
  They must be **exactly on shell** with the param-card masses and conserve
  momentum. Nothing checks this, and a bad point silently gives a wrong
  number. Any Lorentz frame works for unpolarised processes.
- **Masses and widths** are those of `Cards/param_card.dat`. Check the
  masses there (`grep -n "# MT\|# WT" Cards/param_card.dat`) before building
  the momenta. The widths enter **every** propagator of that particle,
  t-channel ones included (e.g. the top width in g g > t t~ g).
- **Parameters**: read from the param card when the library is loaded or
  initialised. To change a mass, coupling or width, edit the card (or point
  to another one) and reload.
- **α_s** is an input of every call (Fortran argument, madspace array).
  The C++ `check_sa.exe` takes it from `aS` in the param card. Before PR
  #258, its `perf` mode ignored the card and used 0.118: on an older
  installation, check by doubling `aS` (a pure-QCD |M|² must scale by
  2^qcd_power).
- **Precision**: the C++ backends (mg7 and `output standalone`) build in
  **mixed precision by default**, which gives about 1e-7 relative error (e+
  e- > mu+ mu-: 1.1242219363e-02 against the exact 1.1242220471e-02). For
  precision work, build with `FPTYPE=d`. If the directory was already
  built, run `make cleanall` first; otherwise make refuses ("Cannot build for
  tag ... as old builds exist"). The Fortran route is always double.

## Subprocesses and flavours

An output has one `SubProcesses/P*` directory per subprocess, with the light
quarks **grouped**: `p p > t t~` gives `P1_gg_ttx` and `P1_QQx_ttx`.
`SubProcesses/subprocesses.json` (mg7 and C++ outputs) lists each
subprocess's flavours. Print them with:

```bash
python3 -c "import json; [print(s['path'], [(f['index'], f['options'], f['mirror']) for f in s['flavors']]) for s in json.load(open('SubProcesses/subprocesses.json'))]"
```

which gives, for `p p > e+ e-`:

```
P1_QQx_epem   flavor index 0: [1, -1, -11, 11], [3, -3, -11, 11]   (d, s)
              flavor index 1: [2, -2, -11, 11], [4, -4, -11, 11]   (u, c)
```

- Options sharing an index have **identical** matrix elements. Pass the
  index: madspace `flavor_in` (a 1-D **int32** array; without it, index 0)
  and C++ `-f <index>`.
- The Fortran `all_matrix2py` takes the real PDG codes instead, and finds the
  subprocess and flavour itself. Its `get_pdg_order()`, like the `incoming`
  field of `subprocesses.json`, shows **81/-81** for a grouped light quark
  (82 for charged leptons, 83 for neutrinos). Still pass the real codes
  (`[2, -2, ...]`).
- `mirror: true` means the reversed initial state (q~ q) is the same
  subprocess with the two incoming legs swapped. All routes evaluate the
  order listed in the options. **For q~ q, swap the two incoming momenta
  (and PDG codes) yourself.** The Fortran module returns exactly 0 for
  `[-2, 2, 6, -6]`, because no subprocess has that order. Alternatively,
  generate the order you need explicitly (`generate u~ u > t t~ g`), which
  gives a subprocess in that order (and `mirror: false`).
- **A result of exactly 0 is suspicious**: an unknown PDG order or the
  wrong `nhel` (below) both give 0 silently.

## Route 1: Fortran, one point per call (Python)

```
# gen.mg7
set crash_on_error True --no_save
import model sm
generate p p > t t~
output standalone_fortran TT_F --prefix=int
```

```bash
cd TT_F/SubProcesses && make f2py      # a few seconds
```

```python
import sys, numpy as np
sys.path.insert(0, "TT_F/SubProcesses")
import all_matrix2py as am
am.initialise("TT_F/Cards/param_card.dat")
print(am.get_pdg_order())                 # the PDG orders it knows
P = np.array([[E1, px1, py1, pz1], ...])  # (legs, 4)
me = am.smatrixhel([21, 21, 6, -6], -1, P.T.copy(), 0.118, 0.0, -1)
#                  pdgs           procid  p is (4, legs): TRANSPOSE   alpha_s scale2 nhel
```

- The module is `all_matrix2py.cpython-<version>-<platform>.so`. The empty
  `all_matrix2py.so` next to it is only a make timestamp.
- **`nhel = -1` sums over helicities. `nhel = 0` returns 0.**
- `nhel = k` (1, 2, ...) gives the single helicity combination k, ordered as
  the `NHEL` table in `P*/matrix.f`, **without** the initial-spin average:
  summing over all k gives (number of initial helicity states) x the
  averaged value. This is the only route with fixed helicities; madspace
  rejects a fixed helicity ("unsupported input key").
- `procid = -1` matches any process; `scale2 = 0` means α_s is used as
  given.
- Without `--prefix=int`, every `P*` directory has its own module, all named
  `matrix2py`. Only one can be imported per Python process, and those
  modules take a 1-based flavour index with **`nhel = 0` as the helicity
  sum**. Prefer `--prefix=int` and `all_matrix2py`. (Before PR #258, a plain
  `make` in `SubProcesses` built only the first module; `make all` builds
  them all.)
- There is no batching: loop in Python (about 5 µs per point for 2 → 2).

## Route 2: madspace, batched (mg7 output)

```
# gen.mg7
set crash_on_error True --no_save
import model sm
generate p p > t t~
output TT_MG7
```

```bash
cd TT_MG7/SubProcesses && make BACKEND=scalar FPTYPE=d -j8
```

- `BACKEND=scalar` keeps the library name portable (the same on every
  machine). For speed, use a plain `make FPTYPE=d -j8` instead: it picks the
  best SIMD backend of this CPU (1.6-1.7x faster here). Its name is in the
  build log (`BACKEND=simd_128 (was auto)`) and in `lib/`
  (`..._simd_128.so`, `..._simd_256.so`, ...). Two backends can be built
  side by side without `make cleanall`; only a change of `FPTYPE` needs
  it. Use `BACKEND=cuda` for an NVIDIA GPU.
- The libraries are in `TT_MG7/lib/libmadmatrix_P1_gg_ttx_scalar.so`, and in
  `subprocesses.json` (`me_path`).

```python
import sys, numpy as np
sys.path.insert(0, "<MG7>/madspace/install")      # the madspace MadGraph7 built
import madspace as ms
ctx = ms.default_context()
api = ctx.load_matrix_element("TT_MG7/lib/libmadmatrix_P1_gg_ttx_scalar.so",
                              "TT_MG7/Cards/param_card.dat")
MEI, MEO = ms.MatrixElement.MatrixElementInput, ms.MatrixElement.MatrixElementOutput
me = ms.MatrixElement(api, [MEI.momenta_in, MEI.alpha_s_in, MEI.flavor_in],
                      [MEO.matrix_element_out], True)
P = ...                                   # shape (npoints, legs, 4), float64
values = me(P, np.full(len(P), 0.118), np.zeros(len(P), dtype=np.int32))
```

- A single point is a batch of one: `me(P[None], np.array([0.118]), ...)`.
- **Threads**: `ms.default_context()` uses every CPU core. To use a fixed
  number of threads, build a context and a runtime explicitly (calling the
  `MatrixElement` object directly would use the default context and fail
  with "Matrix element index out of bounds"):

  ```python
  ctx = ms.Context(device=ms.cpu_device(), thread_count=4)
  api = ctx.load_matrix_element(lib, card)
  gen = ms.MatrixElement(api, [MEI.momenta_in, MEI.alpha_s_in], [MEO.matrix_element_out], True)
  run = ms.FunctionRuntime(gen.function(), ctx)
  values = run(P, alpha_s)            # numpy in, numpy out
  ```
- `flavor_in` must be int32 and 1-D; int64, uint64 or 2-D arrays are
  rejected.
- The last argument (`True`) lets the library draw its own random helicity
  and colour for its event-level outputs. The returned |M|² is still the
  full helicity sum.
- `me_frame` (an optional argument) only matters for polarised processes.
- **Interpreter lock-in**: the compiled modules (madspace's
  `_madspace_py.cpython-<version>-*.so`, and `all_matrix2py` for route 1)
  only load in the Python version that MadGraph7 uses. The user's code must
  run with that interpreter, and it depends on the absolute path
  `<MG7>/madspace/install`. Tell the user both.

## Route 3: C++ standalone, points from a file

```
# gen.mg7
set crash_on_error True --no_save
import model sm
generate p p > t t~
output standalone TT_CPP
```

```bash
cd TT_CPP/SubProcesses/P1_gg_ttx && make FPTYPE=d -j8
./check_sa.exe perf -f 0 --momenta pts.bin --dump-me me.bin 1 8 1
```

- `pts.bin` holds raw doubles (E, px, py, pz) for each leg of each point,
  with no header: `P.astype(np.float64).tofile("pts.bin")` for an array of
  shape (points, legs, 4). `me.bin` comes back as raw doubles, one per
  point: `np.fromfile("me.bin")`.
- The trailing numbers are blocks, threads per block and iterations. Points
  are processed in batches of blocks × threads, which must be a **multiple
  of the SIMD width** (2 for `simd_128` in double, up to 8), or it aborts
  with "nevt should be a multiple of neppM". Pad a single point by
  repeating it 8 times and use `1 8 1`.
- `-e file.lhe` reads the points from an LHE file instead.
- `./check_sa.exe` with no argument evaluates one random point at 1000 GeV
  for every flavour, which is a quick sanity check. That mode does use the
  card's α_s.
- α_s comes from `aS` in `Cards/param_card.dat` (see Conventions for older
  versions).

## Checking a result

- Reproduce a known analytic case first. For example, `generate e+ e- >
  mu+ mu- / z` at √s = 100 GeV and θ = 60° gives 2e⁴(t² + u²)/s² =
  1.1242220471e-02 with α = 1/132.507.
- Two routes on the same point should agree to 1e-13 in double, or about
  1e-7 if one of them is a default (mixed-precision) C++ build.
- `check gauge`, `check lorentz` and `check permutation <process>` at the
  MadGraph prompt test the internal consistency of a matrix element, at
  random points. They are not a way to evaluate the user's points.
