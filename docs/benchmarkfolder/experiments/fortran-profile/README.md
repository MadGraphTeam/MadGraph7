# Generated Fortran profile

Run `python docs/benchmarkfolder/experiments/fortran-profile/run.py` from the
repository's pinned development environment. The script copies the untouched
baseline's complete generated standalone outputs into `.benchmark-work`, compiles
the generated Fortran with `-g -pg`, retains existing `-O3` for polynomial routines,
and profiles original Ninja on CPU 2. It never edits production templates or the
original baseline.

The clocks target approximately ten seconds of warm evaluation. `gprof` also
samples initialization, the warm pass and the separate coefficient-output pass.
Dynamic reducer and master-library samples are excluded; percentages below are
fractions of sampled executable self time, not fractions of complete wall time.
Other builds were active. This profile locates candidates and is not a performance
acceptance measurement.

| Generated routine | `gg → tt~` | `uu~ → Zgg` |
| --- | ---: | ---: |
| `UPDATE_WL_2_1` | 40.08% | 32.29% |
| `UPDATE_WL_3_1` | not generated | 18.41% |
| `UPDATE_WL_1_1` | 20.74% | 12.61% |
| `SLOOPMATRIX` self | 3.18% | 4.01% |
| `BUILD_KINEMATIC_MATRIX` | 1.40% | 0.32% |
| `NINJA_LOOP` self | 1.27% | 0.25% |

Line sampling in `uu~ → Zgg` assigns 27.48% to `polynomial.f:786`, the compact
rank-two/rank-one complex coefficient update, and 16.05% to its rank-three analogue
at line 960. Counterterm/color multiplication around `loop_matrix.f:1003–1004`
accounts for approximately 2.36%; coefficient clearing at line 916 is about 0.32%.
This motivates an arithmetic-order-preserving loop-layout experiment before
further work on interface allocation or repeated momentum construction.

Full flat/call-graph and line profiles, a CSV of the flat profile, executable
hashes, timed status histograms and clock boundaries accompany this note.
