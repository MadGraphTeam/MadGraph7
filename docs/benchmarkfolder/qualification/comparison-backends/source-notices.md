# Inspected source notices

This inventory records the notices shipped with the exact inspected releases.
[source-notices.json](source-notices.json) gives their original paths, revisions
and SHA-256s. It covers the named numerical backends and the Symbolica dependency;
it is not a complete inventory of every transitive crate.

| Source | Inspected declaration or notice |
|---|---|
| COLLIER 1.2.9 | `COPYING` contains GNU GPL version 3; `src/collier_global.F90` explicitly identifies GPL version 3 and the authors' copyright. |
| Ninja 1.2.1 | Release `COPYING` contains GNU GPL version 3; `AUTHORS` retains author attribution. |
| OneLOop 3.6 | Release `COPYING` contains GNU GPL version 3. Source headers explicitly specify version 3 or later and Andreas van Hameren's copyright. |
| NinjaRS `41a4c00e7635e3ab188fc26bacf0aae6941e8082` | Root `LICENSE` contains GNU GPL version 3; workspace `Cargo.toml` declares `GPL-3.0-or-later`. |
| NinjaRS's oneloopmaster `5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5` | Manifest says `MIT`, but the final README paragraph calls this identifier provisional and says distribution/relicensing permission has not been recorded. Preserve both facts in provenance; the manifest alone does not describe that snapshot's notice. |
| TIRust `854b79dcb7fffcb35e92b4a8a3dd443369b798c4` | Manifest and root `LICENSE` declare BSD Zero Clause (`0BSD`). |
| TIRust's oneloopmaster `bd703d09a78f7849a6b86a07096e6f222fc01193` | Manifest and root `LICENSE` declare `0BSD`; its README records original-author permission and says dependency and included patch/reproducer material retain their own licenses. This newer notice does not replace the notice retained with NinjaRS's older snapshot. |
| Symbolica `9cdc2949cdb153478c1eca0d40690dc0abfc616e` and `dfe6bd97b974b1028dc63071ea165136cd2b0ef5` | The two root `License.md` files are byte-identical Symbolica Source-Available License 1.0 texts. |
| Vendored CutTools 1.9.3 | `README` gives author attribution and citations for CutTools, OneLOop and QCDLoop. No package-wide file named `LICENSE`, `COPYING` or `COPYRIGHT` was found in this vendored tree. The inventory does not infer a license from the citation instructions. |

GNU GPL version 3 sections 4–6 address retention of copyright/license/warranty
notices, providing the license text, identification of modified source, and
corresponding-source arrangements for object-code distribution. The complete
release `COPYING`/`LICENSE` files are the source of those conditions.

Symbolica's root license section 3 expressly permits an unmodified dependency
to be linked, embedded or bundled in a distributed product. It requires each
such distribution to include the complete license and preserve copyright,
attribution, provenance and license notices. Accompanying documentation must
state that runtime rights are not included or transferred unless a separate
written agreement provides otherwise. Each recipient's installation/execution
remains governed by applicable runtime terms. These statements are recorded
from the pinned source notice; this inventory does not assert any recipient's
runtime entitlement.

For installed/internal-export packaging, retain the pinned license texts,
attribution/readme material, immutable source provenance and the Symbolica
runtime-rights statement together with the numerical libraries. Preserve the
older oneloopmaster notice accurately instead of relabeling it using the newer
snapshot's manifest. The installer and export implementation own that packaging;
this read-only inventory does not modify the reference repositories or
dependency revisions.

Pinned upstream notices can be inspected at
[NinjaRS](https://github.com/alphal00p/ninjars/blob/41a4c00e7635e3ab188fc26bacf0aae6941e8082/LICENSE),
[older oneloopmaster README](https://github.com/alphal00p/oneloopmaster/blob/5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5/README.md),
[TIRust](https://github.com/huashengshao/TIRust/blob/854b79dcb7fffcb35e92b4a8a3dd443369b798c4/LICENSE),
[newer oneloopmaster README](https://github.com/alphal00p/oneloopmaster/blob/bd703d09a78f7849a6b86a07096e6f222fc01193/README.md),
and [Symbolica](https://github.com/alphal00p/symbolica/blob/9cdc2949cdb153478c1eca0d40690dc0abfc616e/License.md).
