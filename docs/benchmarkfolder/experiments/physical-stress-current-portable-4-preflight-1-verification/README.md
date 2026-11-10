# Portable stress preparation/preflight verification

The original capsule remains unchanged. Pure Python verified it in place and after copying it to a temporary directory, with all Path.open reads outside the relocated capsule rejected. Four negative checks rejected an unsafe path, changed payload bytes, changed gzip raw binding, and an index-consistent forged physical-stress success summary. Five checks passed; no subprocess, compiler, loader or amplitude was invoked.

The sibling capsule index is 49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22. This supplement retains the exact check source, actual capture and check stdout, and results. Both pure-Python commands terminated0.

From the repository root after sourcing tools/tirust_env.sh, the completed commands were:

```sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-portable-4-preflight-1/capture.py --execute
python -B docs/benchmarkfolder/experiments/physical-stress-current-portable-4-preflight-1-verification/check.py
```

Do not repeat capture: it is append-only and the capsule is sealed. The portable verifier can be repeated safely; check.py writes this supplement's result and therefore is retained as a reproduction script, not an instruction to overwrite sealed evidence. For a fresh reproduction, copy the two sibling directories elsewhere first.
