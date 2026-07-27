# Proposed SymJIT upstream series

Base: SymJIT 2.21.1, commit
`48197f32536c894b51ef25b2cf05ddd05c22675f`.

Apply the numbered patches in order. Each patch is independently reviewable
and the series is deliberately layered:

- `0001`: test/build plumbing;
- `0002`: AArch64 compressed-funclet support;
- `0003`: generic direct plane applications;
- `0004`: optional generic table-driven applications built on `0003`.

These files are review artifacts for the original four-commit API series. The
contributor lock consumes the later fork revision
`60a9d66fbfb2181d36a5747c389714eccc187244`, which also contains the two direct
output-lowering fixes and the rlib-only packaging cutover. None of these files
is replayed during installation.

These patches make no assumptions about amplitudes, currents, recurrences, or
pyAmpliCol artifact roles. Bytecode is trusted input; the APIs retain ordinary
memory-contract validation but do not attempt malware or hostile-payload
vetting.
