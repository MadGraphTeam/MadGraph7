# Private precision and guard experiment: not qualified

The private bridge built successfully with 512 working bits and guard
`2^-392`, preserving 113-bit input/output transport and the original
`2^-100` lightlike policy. Its DSO SHA-256 is
`e8b65f86b78f77d8090e7c406274da32920d89279ee4d5d3ef40e518fb05afdc`.
Both bounded native replays exited zero and passed source, input, runtime,
cache-disabled and output-coverage validation. No installed library,
production policy, original physical input or existing evidence was changed.

The new threshold group-5 tensor returns native success, but it fails the
cross-reducer diagnostic. Its scalar component agrees exactly with NinjaRS
in all three Laurent slots. Of the 35 components, 31 finite and 31 simple-pole
components differ by more than `3e-6` under the complex max-norm diagnostic
metric. The maximum finite relative difference is `1.00596` (unit 7), and
the maximum absolute difference is `1602.10` (unit 27). The maximum simple-pole
relative difference is `1.00017` (unit 9), and the maximum absolute difference
is `44.2362` (unit 27). Both reducers return zero double poles for this tensor.
Differences between tensor degrees have different units; absolute maxima are
descriptive and cannot be interpreted as amplitude errors.

The 512-bit TIRust forward/reversed-propagator outputs are identical in all
105 recorded component/slot pairs, including their printed tokens. NinjaRS
forward/reverse poles agree exactly; its maximum finite relative difference
is `8.18734e-9`. Every NinjaRS output is unchanged between the processes loading
the historical 256-bit bridge and the private 512-bit bridge. Order stability
therefore does not resolve the disagreement.

The ordinary group-5 control remains identical to historical TIRust and agrees
with NinjaRS to `9.47e-17` in the finite term and `1.57e-16` in the simple pole.
The massless threshold group-24 control retains its previous finite agreement
(`2.30e-11` maximum relative); near-zero pole discrepancies remain retained
with their absolute magnitudes. These controls constrain normalization,
component mapping and library coexistence, but do not qualify group 5.

The original authoritative invariants and Cartesian shifts are not mutually
exact: the separate input/Gram audits quantify that mismatch. The experiment
does not decide which higher tensor moments represent a common physical
integral under those conventions. Native success, higher precision, order
agreement and reducer voting are each insufficient. No full threshold
amplitude was evaluated with this private bridge, and no production guard
change is proposed. The historical 256-bit refusal remains intact.

`summary.json` retains all component comparisons, both native executions,
ordinary controls, unchanged-NinjaRS checks and source hashes. The initial
forward lock-admission failure is separately preserved; it launched no native
subprocess. Native elapsed values are diagnostics only, never benchmark data.

Reproduce the analysis from the repository root after the frozen replays:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/higher-precision/analysis/summarize.py
```
