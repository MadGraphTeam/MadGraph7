# ABI 3 standalone and NLO validation preparation

This is an unexecuted validation preparation. It preserves the earlier
successful `install-export-fixed` evidence and will test a new archived,
committed source revision containing both production ports.

From the repository root, with the shared Symbolica slot reserved:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/prepare.py --revision HEAD
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/export.py
python docs/benchmarkfolder/policy-v2/install-export-abi3-terminal/runtime_validator.py \
  --build --hide-installation --run
```

The installer uses pinned downloads through the actual MG5aMC CLI, checks ABI 3
metadata and requires the compiled native/compensated/native DP and genuine
binary128-input smoke test. It does not use sibling source overrides.

Standalone and NLO exports copy their dependencies internally. The validator
relocates them, hides the private installation, removes external library-path
overrides, and checks standalone DP/QP/stability results, the full NLO build,
20 pole checks, soft/collinear tests and GNU binary128 symbol resolution.
Success requires the numerical markers and output files, rather than only a
zero CLI exit code. Work products stay in
`.benchmark-work/policy-v2/install-export-abi3-terminal`.
