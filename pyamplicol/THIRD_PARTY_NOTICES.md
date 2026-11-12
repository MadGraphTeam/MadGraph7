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
their Rust dependencies: Symbolica's Rust crates, Rug and its `gmp-mpfr-sys`
bindings, SymJIT, NinjaRS, oneloopmaster, TIRust, GammaLoop's FeynKit, Spenso,
Idenso and linnet crates, and published crates.io crates.
Higher-precision Python evaluation additionally loads the separately installed
Symbolica Python package, which remains subject to Symbolica's terms.

The license files of every linked crates.io crate are reproduced under
`licenses/rust/<crate>-<version>/`; a crate published without a license file
has a `LICENSE-NOTE` there; `tools/release/rust_licence_inventory.py` checks
this against the crates linked on every release target. The crates
distributed under the Mozilla Public License 2.0 (`colored`, `option-ext`,
`smartstring`) are used unmodified; their sources are published on crates.io.

### GMP, MPFR and MPC

The GNU Multiple Precision Arithmetic Library (GMP) 6.3.0, the GNU MPFR
Library 4.2.2 and the GNU MPC library 1.4.1, and the Rust crates `rug` 1.30.0
and `gmp-mpfr-sys` 1.7.1 that wrap them, are distributed under the GNU Lesser
General Public License version 3 or later (GMP alternatively under the GNU
General Public License version 2 or later). Their license texts, the GNU
General Public License version 3 that the Lesser General Public License
incorporates, and the libraries' own `COPYING` files are in
`licenses/rust/gmp-mpfr-sys-1.7.1/` and `licenses/rust/rug-1.30.0/`.

The binary wheels link GMP, MPFR and MPC dynamically. They ship them as
separate shared libraries, built from unmodified sources, that can be replaced
by any interface-compatible build: in `pyamplicol.libs/` (Linux; the libraries carry
the sonames `libgmp-pyamplicol.so.10`, `libmpfr-pyamplicol.so.6` and
`libmpc-pyamplicol.so.3` plus the hash suffix of the wheel repair) or
`pyamplicol.dylibs/` (macOS) for the Python extension, and in
`pyamplicol/_sdk/lib/` next to the C SDK archive, which contains no copy of
them and links them by path (`rusticol-config --libs`). The libraries are built
from the sources bundled in the `gmp-mpfr-sys` 1.7.1 crate
(<https://crates.io/crates/gmp-mpfr-sys/1.7.1>; directories `gmp-6.3.0-c`,
`mpfr-4.2.2-c` and `mpc-1.4.1-c`, also at
<https://gmplib.org/>, <https://www.mpfr.org/> and
<https://www.multiprecision.org/mpc/>) with that crate's configure options,
shared instead of static; `build_backend/arithmetic.py` in pyAmpliCol's source
distribution is the complete build recipe.

The Rust crates `rug` and `gmp-mpfr-sys` themselves (the bindings, not the C
libraries) are statically linked into the extension and the C SDK archive.
Relinking with modified versions of them is offered under section 4(d)(0) of
the GNU Lesser General Public License version 3: pyAmpliCol's source
distribution is the complete Corresponding Application Code (every Rust and
Python source, `Cargo.toml`, `Cargo.lock` and the build backend), and the
Minimal Corresponding Source of the two crates is published on crates.io at
the versions locked in `Cargo.lock`. To produce a modified combined work,
replace the crates in the unpacked source distribution (for example with a
`[patch.crates-io]` entry in its `Cargo.toml` that names the modified crates,
then `cargo update -p rug -p gmp-mpfr-sys`) and rebuild and install from it
(`python -m pip install .` in that directory, as in "To build from the source
distribution published on PyPI" in `docs/user/installation.md`).

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
revision pinned in `rust/loop-reduction/Cargo.toml` and `Cargo.lock`. It is
distributed under the BSD Zero Clause License since its relicensing commit
`37d9a9b3` (formerly a provisional MIT identifier); the grant covers the whole
project, including the pinned revision. oneloopmaster is a Rust/Symbolica
transcription of OneLOop by Andreas van Hameren, who gave his permission for
its distribution and relicensing. `licenses/oneloopmaster.txt` reproduces the
license and the OneLOop credits its README asks for.

## TIRust

TIRust (<https://github.com/huashengshao/TIRust>, formerly
`huashengshao/iregi-rust`; Rust crates `iregi-rust` and `iregi-masters`), the
one-loop tensor-integral reduction library, is linked from the revision pinned
in `rust/loop-reduction/Cargo.toml` and `Cargo.lock` (the cargo feature `tir`,
on by default). It is distributed under the BSD Zero Clause License; its
license covers the project's own code, including the scalar masters generated
from oneloopmaster's expressions (a transcription of OneLOop by Andreas van
Hameren, distributed and relicensed with his permission). The third-party
content its license lists is not part of the two linked crates.
`licenses/TIRust.txt` reproduces the license of the pinned revision.

## GammaLoop Crates (FeynKit)

FeynKit and its sibling crates from GammaLoop
(<https://github.com/alphal00p/gammaloop>) are linked from the revision pinned
in `rust/loop-reduction/Cargo.toml` and `Cargo.lock`. At that revision the
FeynKit crates and Spenso declare `MIT OR Apache-2.0`, linnet declares MIT,
and Idenso, spenso-hep-lib, symbolica-utils and the workspace-hack crate
declare no license and fall under GammaLoop's top-level license, which places
no usage restrictions (`licenses/GammaLoop.txt`). linnet depends on
`dot-parser` 0.5.1 by Martin Vassor, which GammaLoop vendors and which declares
GPL-2.0-or-later (`licenses/rust/dot-parser-0.5.1/`). linnet's
`crates/linnet/src/parser/subgraph_free.rs` is adapted from dot-parser 0.5.1's
`src/subgraph_free.rs` (Martin Vassor, GPL-2.0-or-later); its notice is the same
dot-parser license text. The crate list and license texts are in
`licenses/GammaLoop-crates.txt`.

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
