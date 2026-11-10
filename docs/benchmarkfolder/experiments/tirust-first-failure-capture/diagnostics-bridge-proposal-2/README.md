# Private diagnostics build resource amendment V2

This is a separate build-only proposal. It preserves the failed first attempt,
its source seal755a2d95 and its closed failure capsule. No build, loader or
tensor evaluation has been performed under this proposal.

The exact controller derivation is recorded in resource-delta.json. Only the
exclusive destination changes to .tirust-build/tirust-private-diagnostics-2
and the per-child address-space limit/recorded cap changes from4GiB to16GiB.
All other controller bytes are unchanged: two jobs, nonexclusive CPUs16–17,
release opt-level3/codegen-units1, pinned sources/lock/tools, offline locked
Cargo, fresh target/build caches,900-second deadline, no automatic retry.
No numerical option, precision, guard, optimization level or source changes.

The predecessor's post-terminal host facts report about238GiB MemAvailable
and no memory maximum in the session/user cgroups, comfortably exceeding
two16GiB address spaces at observation time. This is not a reservation or a
guarantee of future availability. oneloop explicitly reported allocation
failure under4GiB; the precise earlier Symbolica EAGAIN cause is unproven.

The copied source payloads, bindings and18pure tests are identical to V1;
only this README, the controller, resource-delta.json and source index differ.
The full private-API design and separate runtime/replay conditions remain in
../diagnostics-bridge-proposal-1/README.md. No private library may be loaded
until actual post-link artifacts/runtime closure receive separate admission.

Run verify.py and run_pure.py here from the required checkout using the
sourced tools/tirust_env.sh environment. Proposed build-only entry, requiring
root admission before one attempt:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-proposal-2/build.py build
```
