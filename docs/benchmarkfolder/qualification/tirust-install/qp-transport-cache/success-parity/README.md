# Actual installed Fortran success parity

Both separately linked installations completed all 65 calls successfully. Their
entire frame files match byte for byte, including every coefficient bit, status,
context generation and before/after cache counter. The common frame SHA-256 is
`7d37b7e40832c08c355be8b94e66c83c06fea2395ee766725a2b321b58f19f9c`.

The cohort exercises DP and QP rank-2 tensors, full and direct-pole evaluation,
uncached UV/IR/NONE selections where supported, BOTH-mode cache hits and bypass,
zero-budget insertion refusal, clearing and context recreation. The exact last
binary128 tadpole bit also passes its analytic check. Full coefficient bytes,
statuses and cache/lifecycle counters are the measured scope; generated physical
amplitudes, floating-point environment state and runtime performance are outside
this diagnostic.

The two raw-API massive-triangle DP calls also succeeded. The preparation document
under `protocol-sources` mentions a historical generated-adapter refusal and allows
such a refusal; that branch was **not exercised** in this run. No native refusal
coverage is claimed here. `post-analysis.py` independently reparses the retained
frames and confirms all 15 counters are zero before and after each of these two
cache-disabled calls, for both installations.

`results.json` records the completed parity and provenance guards. Installed
archives, modules, shared libraries, configuration and Cargo files, compiler-built
workers, all resolved shared-library dependencies, and protocol/fixture inputs
retain their recorded identities. `execution.json` records the commands, CPU
affinity, shared-lock interval and individual worker deadlines. The frozen
executed protocol remains unchanged in `protocol-sources`; `post-analysis.json`
is explicitly later offline analysis and requires no new numerical execution.
