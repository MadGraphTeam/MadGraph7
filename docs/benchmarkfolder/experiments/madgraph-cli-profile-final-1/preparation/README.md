# Final-source stock CLI profile

This capsule is prepared from final source
`77eb57b82d4858ff302fa9870778e01cc86d7326` and the qualified final TIRust
installation with DSO
`d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052`.
Preparation checked the committed Rust bridge, Fortran adapter and C header
against the delivery receipt. It also checked the corrected monotonic
ProcessTimer and truthful configured-helicity labels. No compiler, CLI, SDK or
numerical calculation ran during preparation.

Actual execution waits for the coordinator's numerical window, after the
seven-cycle first-point pilot and all owned cleanup have closed. Backend 8 and
9 run sequentially with identical final dependency prefix, full finite plus
UV/IR pole mode 0, QP integrand enabled and cache -1. The stock command requests
ten separate stability samples. N2 command files are prepared but require
separate scheduling after a smooth N1 comparison.

Run N1 backend 8:

```
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/madgraph-cli-profile-3/run_cli.py --case n1 --backend 8
```

Then run backend 9, cloning the first receipt's saved export to preserve both
observations and reuse generated/compiled model information:

```
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/madgraph-cli-profile-3/run_cli.py --case n1 --backend 9 --reuse-export <saved_export_from_backend_8>
```

The runner rejects changed-prefix or changed-DSO reuse. It uses the established
owned-process supervisor, numerical mutex, allowed nonexclusive CPU, 3,600s cap
with 60s cleanup reserve, and shared/source hash guards. All failures and raw
observations are retained.

Headline values are the full helicity-sum FULL wall time per point, with SKIP
and FULL−SKIP shown separately; the difference is an estimated reduction time.
The retained last all-helicity result is inspected for finite/nonzero Born and
finite coefficients, without claiming a complete timed point/status stream.
The configured single-helicity result has unverified contribution and remains
diagnostic. Its potentially zero value is never a contributing-helicity
performance headline.

The corrected unpolarized FULL clock uses ProcessTimer's monotonic completion
waiter, independent of the RSS polling interval. Other stock stages retain
Python time.time subprocess durations. Stock calibration still chooses unequal
point counts from cold initialization/estimation, so exact batch lengths and
raw durations must be reported. A single shared-host stock run is descriptive;
it does not provide paired confidence acceptance or final precise local timing.

Stock timing uses a DP interface. The final bridge permits local 106/256/512-bit
tensor rescues while retaining MadLoop's independent checks and genuine QP path.
Installed route-statistics capability is recorded, but the unmodified stock
driver does not emit the counters. Missing timed routes are unknown, not zero;
no all-native-DP claim follows from this timing or an amplitude return code.
The separate ten-point stability analysis also retains its own scope.

First backend initialization is its fresh stock bootstrap. Reusing the export
for the second backend leaves its separate Initialization field null; this
means unmeasured, never zero. Boot/estimate stages remain separately available.
The fixed-cohort initialization improvement and earlier coarse stock controls
belong to their own source-bound observations and are not relabeled here.
