# Third-Party Notices

pyAmpliCol and Rusticol source written for this project are licensed under the
BSD Zero Clause License. That license does not replace the terms attached to
third-party dependencies or bundled model data.

## Symbolica

pyAmpliCol uses Symbolica, including its idenso and spenso functionality.
Symbolica is proprietary software and is recorded in the Rust inventory as
`LicenseRef-Symbolica-Proprietary`. Its upstream terms state that copying or
distribution of any part of the Symbolica code requires express prior
permission:
<https://symbolica.io/license/>.

The pyAmpliCol project has express authorization from the Symbolica licensor
to redistribute the Symbolica components required by pyAmpliCol's binary
runtime. This project-specific permission does not authorize separate
redistribution of Symbolica. Users remain responsible for obtaining the
appropriate Symbolica use license. Attribution, upstream terms, and the scope
of pyAmpliCol's redistribution permission are recorded in
`licenses/Symbolica.txt`.

Release artifacts must record the exact Symbolica package, Rust crate,
serialization ABI, and source revision used to build them.

## SymJIT

SymJIT is a separate Rust dependency used by Symbolica's JIT functionality.
The published `symjit` crate is distributed under the MIT License, not the
Symbolica proprietary license. Its copyright notice and complete MIT License
are reproduced in `licenses/SymJIT.txt`.

## Rust Dependency Inventory

`licenses/RUST_THIRD_PARTY.toml` records every third-party package in
`Cargo.lock`, including transitive packages, with its exact version, Cargo
source, checksum when supplied by Cargo, and curated SPDX license expression.
`tools/release/check_rust_licenses.py` fails if the lock and inventory differ,
if required special notices are missing, or if any legal file is not included
in the release source.

## Native Runtime Feature Boundary

`Cargo.lock` records optional Symbolica, Rug/GMP, and Malachite dependencies
used by developer-only arbitrary-precision Rust checks. The distributed
`pyamplicol._rusticol` extension and `librusticol_capi.a` select the f64-only
Rusticol closure and link SymJIT directly; they do not link Symbolica, Rug/GMP,
or Malachite. Higher-precision Python evaluation loads the separately installed
Symbolica Python package lazily and remains subject to Symbolica's terms.

`licenses/STATIC_LINK_COMPLIANCE.toml` records this feature-separation policy.
The release gate derives each target-specific Cargo closure and scans both
native artifacts for GMP/MPFR/Malachite markers. If a future feature change
makes LGPL code statically reachable, publication is blocked until the policy
contains complete target coverage and the required relinking/source evidence.

## UFO Model Loader

The `ufo-model-loader` dependency is maintained at
<https://github.com/alphal00p/ufo_model_loader> and retains its own license.
Release metadata pins the exact compatible published version.

The loader is distributed under the MIT License. Its complete license is
shipped with the bundled dependency provenance.

## GammaLoop Model Assets

The distribution includes Standard Model, scalar contact, and scalar-gravity
UFO examples together with serialized JSON forms. Their source provenance,
authors, generators, and applicable licenses are recorded in
`src/pyamplicol/assets/models/PROVENANCE.toml`.

Publishing is blocked if that provenance inventory is absent, incomplete, or
does not match the packaged asset checksums.

The model sources were distributed with GammaLoop. GammaLoop's upstream
license says that there are no usage restrictions for GammaLoop; because that
statement has no standard SPDX identifier, these assets use the local
identifier `LicenseRef-GammaLoop-No-Restrictions`. The upstream text is
reproduced in `licenses/GammaLoop.txt`, and its asset-specific scope is
recorded in `licenses/GammaLoop-model-assets.txt`. The SM UFO identifies N.
Christensen and C. Duhr as authors. The scalar and scalar-gravity models
identify Valentin Hirschi as author and were generated with FeynRules 1.7.69.

The SM UFO's optional `build_restrict.py` originates from
MadGraph5_aMC@NLO. Its adapted University of Illinois/NCSA license is
reproduced in `licenses/MadGraph5_aMCatNLO.txt`. Generated JSON forms retain
the provenance and terms of their corresponding UFO source.
