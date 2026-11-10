# QCD n3 run2: independent static runtime-closure audit

This audit inspected the closed run2 loader trace and the existing build2 ELF,
link map, plan and donor files. It did not invoke a compiler, dynamic loader or
amplitude executable. Original inputs were hashed before and after inspection
and remained unchanged. Source2, build2 and run2 were not edited.

The strict source2 closure guard correctly rejected three compiler-library
substitutions before any numerical worker started. The trace resolves exactly
15 shared ELF providers. All physics libraries, libgfortran and the glibc 2.40
loader and libraries retain their expected resolved paths and hashes.

| SONAME | Expected SHA-256 | Actual SHA-256 |
|---|---|---|
| libquadmath.so.0 | `78d384074a271a26fd8bd466aedb4f285e56547daf596aa898c3bfff4bcaa6b5` | `6f9f9a2633ff0f9c9c2c8a9f6df3a145be3233d9037526a7dd8ac3cd1c7e46e6` |
| libgcc_s.so.1 | `528a3ea63aa4c25bf9ba5cd8d14c8bdeab3c5aaea16dc9cee00d50fdd8565a53` | `ae2a657067061213fe4233f0c1c1fe4f77d7d5913f6f3dc6a3fd323f138b71b8` |
| libstdc++.so.6 | `07853a5af9ef85b9d180efe4a795db0023e40ec9168872095580aef556f9e0ec` | `ce5851a9d16f70802f67bd7397d3cab4b78173cb604318f3e35d8b2d835fed4e` |

The preserved target `LD_LIBRARY_PATH` puts the gfortran `cb4kg…` library
directory first. It contains all three SONAME aliases, including a libgcc_s
symlink to `wy205…`. Those aliases therefore take priority over the historical
`chqq…`/`xc0…` alternatives. The fresh executable also has DT_RUNPATH, rather
than DT_RPATH, with `cb4kg…` as its first compiler-library directory. This is an
observed trace under the explicit target environment, not an assumed future
loader choice.

The link map selected `xc0…` libgcc_s and libquadmath plus `cb4kg…` libgfortran.
Link-time inputs and runtime SONAME lookup are different records. In particular,
link-time libquadmath has a third hash, `58f1e90497fdb2fe3bfc1ea468829b2f3e469c28a5b2018b6693b3a53491dbd2`.
The fresh executable has no direct libstdc++ DT_NEEDED entry; libninja still
requires it transitively. Libgfortran requires libquadmath transitively.

All three replacement pairs have the same SONAME, version-definition names,
version-requirement names and GCC 15.2.0 comment. Across the actual 15 providers
and fresh executable, 1,142 versioned imports have a matching symbol and version
in their declared provider. No DT_NEEDED provider or required version is missing.
None of the six replacement alternatives has a GNU build-ID note. The empty
build-ID-list equality in `audit.json` denotes absent metadata; it does not
establish build identity.

The libraries are not byte-identical. Their differences include ELF layout and
dynamic metadata; libquadmath also has different `.text` bytes. Consequently,
the audit supports an isolated successor diagnostic with the same explicit
target environment, a new exact 15-path/hash binding and unchanged strict
runtime-map checks. It does not establish numerical equivalence with the
historical campaign or qualify the QP diagnostic itself.

`audit.json` records all aliases, canonical paths, hashes, ELF metadata, version
edges and original input pins. `section-comparison.json` records independent
section/header comparisons. `link-map-and-explanation.json` records the actual
link inputs and the resolution explanation. `metadata-crosscheck.json` compares
the independent parser against the previously pinned static metadata reader.

To reproduce the main inspection, copy `audit.py` into a fresh additive audit
directory at the same repository depth, then run it with the pinned Python
environment. The script refuses to overwrite its existing `audit.json`.
