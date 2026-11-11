# Validation-only successor to the retained failed BUILD

Original SOURCE1 is4f496ebf…; its BUILD terminale1d442db… remains failed and
unchanged. All54 completed compiler/tool commands returned0, including the
single driver compilation and relink. The original loader guard then stopped
before any loader query or numerical main because one expected transitive
runtime provider was absent from its helper's `prelink.bound` dictionary.

This source-only successor performs no compilation, linking or tensor evaluation.
It reuses the exact compiled ELF38fe2ed4…, driver object872655c3… and link
map3ccbbb57…. No failed stage file is rewritten. The430 protected files include
all172 original failed-BUILD artifacts/logs, both original source manifests and
every original source/header/tool/object/library dependency.

The scoped missing binding is
`/nix/store/chqq8mpmpyfi9kgsngya71akv5xicn03-gcc-15.2.0-lib/lib/libquadmath.so.0.0.0`,
SHA78d384074a271a26fd8bd466aedb4f285e56547daf596aa898c3bfff4bcaa6b5.
It was already in the original SOURCE1 protected input manifest. The immutable
original controller verifies that whole manifest before entering its build
function; its successful subsequent compiler commands establish that guard
completed. Original source/admission/lock/command hashes and ordering are checked
explicitly. This is evidence of the existing source-protection guard, not a new
measurement of historical file contents.

Validation writes a **new, explicitly post-failure provenance join** under a new
work directory. It preserves the original helper record and says that record did
not include the extra provider. Only this single path/hash may be added, and it
must match the original protected manifest and current bytes. No compiler flag,
link input, ELF, object, model, fixture, provider or oracle changes.

The existing profile loader helper first verifies every original actual LOAD
and PT_INTERP against this joined record. It then performs the previously planned
actual-loader `--list` query and requires the entire provider set to equal the
original pinned expected set. A mismatch remains a new failed validation; no
subset/fallback provider policy is permitted. One `nm --defined-only` query then
checks unique compiled DP/QP diagnostics, binary128 module and forwarding
observer symbols. The Fortran main remains unexecuted and all24 numerical rows
remain pending. A successful validation cannot reclassify the original failed
BUILD or imply numerical, physical-amplitude, stability or performance success.

The reviewed StageDeadline/raw-cleanup/common-owned-child machinery is reused.
One nonexclusive CPU12,4GiB address space,512MiB file size, core0 and the common
numerical mutex apply. The sole validation stage has a180s all-in deadline,
including120s work and60s cleanup. Original-source/artifact rehashing, loader and
symbol queries, metadata checks and administrative closure share that deadline.
There is no automatic retry, output overwrite, BUILD option or RUN option.

After separate root/audit source admission, the proposed command is:

```sh
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/rank6-ir-triangle-validation-controller-1/controller.py --validate --source REVIEWED_VALIDATION_SOURCE_SHA256
```

`--preflight --source SHA` is file-only. The validation outputs belong under
`.tirust-build/rank6-ir-triangle-validation-work-1/validate`; original failed
artifacts stay read-only. RUN requires a distinct later controller/admission and
a reviewed successful validation hash. No loader or numerical command has run
for this source proposal. Precise local performance remains Hua-Sheng's TODO.
