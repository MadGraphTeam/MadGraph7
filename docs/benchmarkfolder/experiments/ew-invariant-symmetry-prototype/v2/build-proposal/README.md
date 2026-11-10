# Revision 2 adapter-only build proposal

No compiler or executable has been run for this proposal. `controller-delta.patch` contains the complete delta from reviewed revision 1 controller `0e42cc72bd7d19d6b5a84b047d8dbee147166bd66cb70290791cd8417e4acf9e`: repository depth, candidate hash, two schema labels, and the fresh ignored output namespace.

The original object, archive, generated-source, input, compiler, and runtime bindings are unchanged. The only replacement is the newly derived `TIRUST_interface.o`; the qualified raw-word bootstrap-plus-100 driver, host observer, and no-op shim are retained exactly. The pending stage has one adapter compilation, one link, and one loader dependency listing, with CPU12 nonexclusive affinity and a 180-second whole-stage bound. It performs no Rust build, reducer evaluation, or physical test.

The read-only plan command is:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/build-proposal/run.py plan
```

The following command is prepared, **not executed or admitted**:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/build-proposal/run.py build \
  --label attempt1 --cpu 12 \
  --self-sha256 c442152caf1f7d29cd9511eba259cc442461730baeab17fb19ee4d3126d5593d \
  --plan-sha256 74cc1ba21a3f2da38b6f26d7f80d6fa443a9be08b2955032aeb292d2c698e184
```

Any failure must be retained in its own attempt. A successful compile/link would not establish live numerical runtime, coefficient accuracy, or performance equivalence.
