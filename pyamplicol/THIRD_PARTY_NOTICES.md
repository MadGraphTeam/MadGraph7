# Third-Party Notices

pyAmpliCol and Rusticol source written for this project are licensed under the
BSD Zero Clause License. That license does not replace the terms attached to
third-party dependencies or bundled model data.

## Symbolica

pyAmpliCol uses Symbolica, including its idenso and spenso functionality.
Symbolica is proprietary software. Its upstream terms state that copying or
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

The native libraries statically link the Symbolica Rust crates `symbolica`,
`numerica` and `graphica` from the revision pinned in `Cargo.toml` and
`Cargo.lock`; `licenses/Symbolica.txt` reproduces that revision's complete
license, and `numerica` and `graphica` state their own MIT License
(`licenses/rust/numerica-3.0.2/`, `licenses/rust/graphica-3.0.1/`).

Release artifacts must record the exact Symbolica package, Rust crate,
serialization ABI, and source revision used to build them.

## Developer-Only Legacy AmpliCol Oracle

Contributors may opt into a pinned checkout of the original Fortran AmpliCol
repository for independent numerical and performance validation. That checkout
is not included in pyAmpliCol wheels or source distributions. The contributor
metadata records that the pinned upstream revision contains no `LICENSE` or
`COPYING` file. This observation is not a grant or a claim about the upstream
source's terms. The developer setup uses the dedicated upstream branch
`amplicol_with_patches`; no legacy-AmpliCol patch payload is
shipped in pyAmpliCol release artifacts.

## SymJIT

SymJIT is a separate Rust dependency used by Symbolica's JIT functionality.
The published `symjit` 2.28.3 crate pinned in `Cargo.lock` is distributed
under the MIT License, not the Symbolica proprietary license. Its copyright
notice and complete MIT License are reproduced in `licenses/SymJIT.txt`.
The generic unsafe raw P-kernel plane-descriptor callable was merged upstream
in `siravan/symjit-crate#1`; it does not alter generated kernels.

## Native Libraries

The distributed `pyamplicol._rusticol` extension and the C SDK archive
`librusticol_capi.a` statically link pyAmpliCol's own Rust crates (0BSD) with
their Rust dependencies: Symbolica's Rust crates, Rug with GMP, MPFR and MPC
(through `gmp-mpfr-sys`), SymJIT, NinjaRS, oneloopmaster, GammaLoop's FeynKit,
Spenso, Idenso and linnet crates, and published crates.io crates.
Higher-precision Python evaluation additionally loads the separately installed
Symbolica Python package, which remains subject to Symbolica's terms.

The license files of every linked crates.io crate are reproduced under
`licenses/rust/<crate>-<version>/`; a crate published without a license file
has a `LICENSE-NOTE` there; `tools/release/rust_licence_inventory.py` checks
this against the crates linked on every release target. GMP, MPFR and MPC,
and the Rust crates `rug` 1.30.0 and `gmp-mpfr-sys` 1.7.1 that wrap and build
them, are distributed under the GNU Lesser General Public License version 3
or later; their license texts are in `licenses/rust/gmp-mpfr-sys-1.7.1/` and
`licenses/rust/rug-1.30.0/`. The crates distributed under the Mozilla
Public License 2.0 (`colored`, `option-ext`, `smartstring`) are used
unmodified; their sources are published on crates.io.

## NinjaRS

NinjaRS (<https://github.com/alphal00p/ninjars>), the one-loop integrand
reduction library, is linked from the revision pinned in
`rust/loop-reduction/Cargo.toml` and `Cargo.lock`. It is distributed under the
BSD Zero Clause License since its relicensing commit `36e30a49` (formerly
GPL-3.0-or-later; relicensed with the agreement of all its authors). Its
license is reproduced in `licenses/NinjaRS.txt`.

## oneloopmaster

oneloopmaster (<https://github.com/alphal00p/oneloopmaster>, Rust crate
`oneloop`) supplies the one-loop scalar integrals and is linked from the
revision pinned in `rust/loop-reduction/Cargo.toml` and `Cargo.lock`. Its
manifest declares the MIT License and has no license file; its README calls
that identifier provisional, because the crate is a transcription of OneLOop
by Andreas van Hameren (GPL-3.0-or-later) and the permission for its intended
distribution or relicensing has not been recorded as granted.
`licenses/oneloopmaster.txt` records the declaration, the OneLOop credits its
README asks for and the MIT text; the GPL version 3 text is in
`licenses/GPL-3.0.txt`.

## GammaLoop Crates (FeynKit)

FeynKit and its sibling crates from GammaLoop
(<https://github.com/alphal00p/gammaloop>) are linked from the revision pinned
in `rust/loop-reduction/Cargo.toml` and `Cargo.lock`. At that revision the
FeynKit crates and Spenso declare `MIT OR Apache-2.0`, linnet declares MIT,
and Idenso, spenso-hep-lib, symbolica-utils and the workspace-hack crate
declare no license and fall under GammaLoop's top-level license, which places
no usage restrictions (`licenses/GammaLoop.txt`). linnet depends on
`dot-parser` 0.5.1 by Martin Vassor, which GammaLoop vendors and which declares
GPL-2.0-or-later (`licenses/rust/dot-parser-0.5.1/`). The crate list and
license texts are in `licenses/GammaLoop-crates.txt`.

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
