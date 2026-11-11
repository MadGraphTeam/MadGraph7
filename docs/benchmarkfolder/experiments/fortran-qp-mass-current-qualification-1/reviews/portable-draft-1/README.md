# Independent numerical-retention draft review

This record preserves the two actual pure audit attempts against draft manifest
`464199630c9aaa43edef10026041777af5b55be09cc8655431fe0c06b1296bb4`.
The first wrapper omitted a standard-library preload; its audit hook correctly
rejected the resulting `argparse` import outside the capsule before numerical
replay. This was an audit-wrapper setup mistake, not a native or source failure.

The corrected wrapper preloaded `argparse`, `contextlib` and `fractions`. The
observed pure replay passed in 1.763 seconds with 1,710 capsule-only file reads,
zero external/mutating/process/ctypes attempts, zero nested forbidden attempts,
and identical manifest hashes before and after verification.

Only BUILD2, recording units and CACHE0 were closed and replayed. CACHEALL and
both observers were pending. The retained scripts bind that exact old draft;
executing them against a later refresh must reject the changed manifest.
They are provenance, not permission to run any native program or requalify a
later package. The reported duration is a pure verification duration and is not
a physics runtime benchmark. No new replay was performed while recording this
auditable history.
