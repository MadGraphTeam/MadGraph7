#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build the independent reference copies of the shipped loop UFOs.

The loop tests compare the shipped ``loop_sm``, ``loop_qcd_qed_sm`` and
``loop_qcd_qed_sm_Gmu`` with reference copies built from the public
upstream sources (``src/pyamplicol/assets/models/PROVENANCE.toml``):

* UFOmodeldb (https://github.com/mg5amcnlo/UFOmodeldb) commit 6ca4b120:
  ``loop_qcd_qed_sm.tgz`` and ``loop_qcd_qed_sm_Gmu.tgz``, extracted without
  ``__pycache__`` (``ufodb-6ca4b120/``, the unpatched controls);
* with the model patch of UFOmodeldb issue 1 / mg5amcnlo#475 (gist
  ValentinHirschi/81a26a3c, applied with ``patch --fuzz=0 -p1``) in
  ``ufodb-6ca4b120-patched/``;
* ``loop_qcd_qed_sm_Gmu_py3`` (both): the Python-2 G_mu model's Python-3
  conversion (``ufo-python3-conversion-v1``: MG5_aMC's ``convert model`` with
  MG5_aMC 3.7.0's ``models/sm/write_param_card.py``, then package-relative
  sibling imports); the patched one then has the patch's module-level value
  appends folded by ``tools/developer/fold_ufo_value_appends.py``;
* MG5_aMC 3.7.0's ``models/loop_sm`` (git tag v3.7.0) in
  ``MG5_aMC_v3_7_0/models/loop_sm``.

Every download is pinned by commit and sha256 and checked before use; a
mismatch fails the build (a cached download that does not match is fetched
again, and must then match). ``--github-env`` appends the test variables
(``ENVIRONMENT``) that point at the copies.

Usage: fetch_loop_reference_ufos.py OUTPUT [--downloads DIR] [--github-env FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLD = ROOT / "tools" / "developer" / "fold_ufo_value_appends.py"

UFODB_COMMIT = "6ca4b12088b354420b1a1cadf26314d9c56980ab"
UFODB_URL = f"https://github.com/mg5amcnlo/UFOmodeldb/raw/{UFODB_COMMIT}/{{}}"
UFODB_ARCHIVES = {
    "loop_qcd_qed_sm.tgz": (
        "b6a60603d510e7ab091f2eb202076503bf0d163c599836235be45477b3664e70"
    ),
    "loop_qcd_qed_sm_Gmu.tgz": (
        "9f3668e79a8e461f6f9738a18e9b63665c7f970cdade124461867581dd3c9811"
    ),
}
# The gist file at revision d887ff80 (gists strip the trailing newline of the
# recorded ufomodeldb-issue-1.patch, sha256 4b5b1ef0...; same hunks).
PATCH_URL = (
    "https://gist.githubusercontent.com/ValentinHirschi/"
    "81a26a3c9a03ca38cd2f6d6cf6a2b75f/raw/"
    "862ff6d85debba94fb5de3c80bc07642192e31cc/ufomodeldb-issue-1.patch"
)
PATCH_SHA256 = "9d47d7a4db4c8547746e672436b28b1a6e540e2ded562a6753b666a6f2850523"
MG5_COMMIT = "adf3d62d096abc1f1512c2101b40218f7c529d4f"  # tag v3.7.0
MG5_URL = (
    f"https://raw.githubusercontent.com/mg5amcnlo/mg5amcnlo/{MG5_COMMIT}/models/{{}}"
)
MG5_WRITE_PARAM_CARD = (
    "sm/write_param_card.py",
    "e804608eedfdafbcd024762cc5540d42e03ccc2f1d8058586a6d5ee64261705c",
)
MG5_LOOP_SM = {
    ".restrict_parallel_test.dat": (
        "33abf2ee8645e2f55414bb89bb6b9359ad70384667d271d4213a846281622e0e"
    ),
    ".restrict_parallel_test_c_massive.dat": (
        "e6da223a909271a9fe596a71f3d3ee97bba783a20dda65cb3d3ae71615e1ac3d"
    ),
    "CT_couplings.py": (
        "ec0ab5fab1d693e246284c1d82d9ef5219b274dedb909bdd3f6187cbc1ed7b02"
    ),
    "CT_parameters.py": (
        "fe89b26fcc375dbffc40003d7e064b2e66316d8d4daa897cd7a9c0ae386cfb3d"
    ),
    "CT_vertices.py": (
        "6f125c6cffdd94491fb6167411b23700468663d46c60cee90cdb675ad5659b0f"
    ),
    "__init__.py": "f313e3030c3b3a5d8da2db3ed1b8ad487ce45dfb21810b15a1f717b9a4147d4b",
    "coupling_orders.py": (
        "c75540ae451aaca4c8f497aed727a252a2960c65c9d6174e94039af8a0615ae0"
    ),
    "couplings.py": "a91df246f6a68325ead56ec30d49c0586fb9b39451ae62a5c527896acb45014f",
    "function_library.py": (
        "4f61390f3cd3ba3baa6b37a908f0f3129c457bca2d4e2b2fa8806c3ac31f9ea3"
    ),
    "lorentz.py": "2552626d868a558a321c84fbafd14b43dafd2f4bbd45cf04a7693cb1caac3dc1",
    "object_library.py": (
        "a4bb6c554809ff739fea28a70e3a889408ea93f48e55cfe20a68d5664a7e16e1"
    ),
    "parameters.py": "5de44653a32f4c213b6fc0696f3585f36bc34830e64d713c7e268814c847b58f",
    "particles.py": "a437dda2285158b79b44d97830d219f16a3fd1aa864f75131acca6d826893f78",
    "restrict_c_mass.dat": (
        "11626ca4f6a8220df5ed33d530b6fec92fe69ba5f1c82f54bc17427dd3e83116"
    ),
    "restrict_ckm.dat": (
        "d51f802a2fb5f55a4be5ca0c6daaf666514757f24f42c656a3c00818afd12e44"
    ),
    "restrict_default.dat": (
        "878a7d1b110faf49824885bd340fb09d6431394d9c7f3d5b86c12d6f4ef54160"
    ),
    "restrict_lepton_masses.dat": (
        "12e95ad3b16a4d92c20c4d60068b27355021bd2966df2af081e06f017ba9bce7"
    ),
    "restrict_no_b_mass.dat": (
        "9e1839428fd3a3918fa14be2a5ba0b4907dc12134e91ff08d40cd1056208714c"
    ),
    "restrict_no_masses.dat": (
        "1f72584d6ed563aa911369719a14d8ac3c51bbe055985b3d0a07b6a7b464ad83"
    ),
    "restrict_no_tau_mass.dat": (
        "4170075aed30aa27eba1ec1533d42fa1496ff605bf666608ed31557e65d28f46"
    ),
    "restrict_no_widths.dat": (
        "05ebf8a6e774a80af3f6880346d0e313cef5aaea92f70a9ec35de68635bdfedf"
    ),
    "restrict_parallel_test.dat": (
        "33abf2ee8645e2f55414bb89bb6b9359ad70384667d271d4213a846281622e0e"
    ),
    "restrict_test.dat": (
        "2ea03c8e94750e40c403d4524bc461f9c1c5802e5f951e15abc18011513f7fe1"
    ),
    "restrict_zeromass_ckm.dat": (
        "ea454a976b28615c60a12d638be9e6896b4e29de10deb8f5ff7337b81a1794ff"
    ),
    "vertices.py": "89a4311b8305295e3019e1570c57519c2614b784b1f9ee9637a54ee9d2f07b82",
    "write_param_card.py": (
        "628c1411379454931f6a1f4f4e29d2034da8ac095cc2d838e03a24834130bf5a"
    ),
}

UNPATCHED = "ufodb-6ca4b120"
PATCHED = "ufodb-6ca4b120-patched"
LOOP_SM = "MG5_aMC_v3_7_0/models/loop_sm"
# test variable -> copy (relative to OUTPUT); tests/conftest.py reads them.
ENVIRONMENT = {
    "PYAMPLICOL_NLO_UFO_REFERENCE": LOOP_SM,
    "PYAMPLICOL_EW_UFO_REFERENCE": f"{PATCHED}/loop_qcd_qed_sm_Gmu_py3",
    "PYAMPLICOL_EW_NLO_UFO_REFERENCE": f"{PATCHED}/loop_qcd_qed_sm",
    "PYAMPLICOL_EW_UFO_UNPATCHED_REFERENCE": f"{UNPATCHED}/loop_qcd_qed_sm_Gmu_py3",
    "PYAMPLICOL_EW_NLO_UFO_UNPATCHED_REFERENCE": f"{UNPATCHED}/loop_qcd_qed_sm",
}


class FetchError(RuntimeError):
    """A download that cannot be had or does not match its pinned sha256."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, expected: str, target: Path) -> Path:
    """``target`` holding ``url``'s bytes of sha256 ``expected``: a cached
    file that matches is kept, anything else is downloaded again and must
    match (a mismatch leaves no file)."""

    if target.is_file() and sha256(target) == expected:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    partial = Path(name)
    try:
        with (
            os.fdopen(handle, "wb") as sink,
            urllib.request.urlopen(url, timeout=120) as source,
        ):
            shutil.copyfileobj(source, sink)
        found = sha256(partial)
        if found != expected:
            raise FetchError(f"{url}: sha256 {found}, pinned {expected}")
        partial.replace(target)
    except OSError as error:
        raise FetchError(f"{url}: {error}") from error
    finally:
        partial.unlink(missing_ok=True)
    return target


def extract(archive: Path, target: Path) -> None:
    """Extract the regular files and directories of ``archive`` below
    ``target``, without ``__pycache__``; links or escaping names fail."""

    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                raise FetchError(f"{archive.name}: unsafe member {member.name}")
            if not (member.isfile() or member.isdir()):
                raise FetchError(f"{archive.name}: {member.name} is not a file")
            if "__pycache__" not in parts:
                members.append(member)
        if sys.version_info >= (3, 12):
            tar.extractall(target, members=members, filter="data")
        else:
            tar.extractall(target, members=members)  # members checked above


def apply_patch(patch: str, directory: Path) -> None:
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "--no-backup-if-mismatch", "-p1"],
        cwd=directory,
        input=patch,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )


_FROM = re.compile(r"^(\s*)from (\w+) import ", re.MULTILINE)
_IMPORT = re.compile(r"^(\s*)import (\w+)((?: as \w+)?[ \t]*)$", re.MULTILINE)


def convert_python3(model: Path, write_param_card: Path) -> None:
    """``ufo-python3-conversion-v1`` in place: MG5_aMC 3.7.0's ``convert
    model`` (object_library.py's ``.iteritems()`` and Python-2 raises,
    write_param_card.py replaced by MG5_aMC's sm one, object_library and
    function_library imported by ``__init__.py``), then every import of a
    sibling module made package-relative."""

    library = model / "object_library.py"
    text = library.read_text(encoding="utf-8").replace(".iteritems()", ".items()")
    text = re.sub(
        'raise (\\w+)\\s*,\\s*["\']([^"]+)["\']', r'raise \g<1>("\g<2>")', text
    )
    library.write_text(text, encoding="utf-8")
    shutil.copyfile(write_param_card, model / "write_param_card.py")
    init = model / "__init__.py"
    text = init.read_text(encoding="utf-8")
    for name in ("object_library", "function_library"):
        if f"import {name}" not in text:
            text = f"import {name} \n" + text
    init.write_text(text, encoding="utf-8")
    modules = {path.stem for path in model.glob("*.py") if path.stem.isidentifier()}
    for name in modules:
        path = model / f"{name}.py"
        text = path.read_text(encoding="utf-8")
        text = _FROM.sub(
            lambda m: f"{m[1]}from .{m[2]} import " if m[2] in modules else m[0], text
        )
        text = _IMPORT.sub(
            lambda m: f"{m[1]}from . import {m[2]}{m[3]}" if m[2] in modules else m[0],
            text,
        )
        path.write_text(text, encoding="utf-8")


def build(output: Path, downloads: Path) -> dict[str, Path]:
    """Build every copy below the new directory ``output``; the variables."""

    if output.exists() and any(output.iterdir()):
        raise FetchError(f"{output} is not empty")
    archives = {
        name: fetch(UFODB_URL.format(name), digest, downloads / "ufodb" / name)
        for name, digest in UFODB_ARCHIVES.items()
    }
    patch = fetch(PATCH_URL, PATCH_SHA256, downloads / "ufomodeldb-issue-1.patch")
    card_path, card_digest = MG5_WRITE_PARAM_CARD
    card = fetch(MG5_URL.format(card_path), card_digest, downloads / "mg5" / card_path)
    loop_sm = {
        name: fetch(
            MG5_URL.format(f"loop_sm/{name}"), digest, downloads / "mg5/loop_sm" / name
        )
        for name, digest in MG5_LOOP_SM.items()
    }
    text = patch.read_text(encoding="utf-8")
    for flavour in (UNPATCHED, PATCHED):
        for archive in archives.values():
            extract(archive, output / flavour)
        if flavour == PATCHED:
            apply_patch(text, output / flavour)
        gmu = output / flavour / "loop_qcd_qed_sm_Gmu"
        py3 = gmu.with_name("loop_qcd_qed_sm_Gmu_py3")
        shutil.copytree(gmu, py3)
        convert_python3(py3, card)
        if flavour == PATCHED:
            # The upstream TeX names hold invalid escapes (SyntaxWarning).
            fold = [sys.executable, "-I", "-W", "ignore::SyntaxWarning", str(FOLD)]
            subprocess.run([*fold, str(py3)], check=True)
    target = output / LOOP_SM
    target.mkdir(parents=True)
    for name, path in loop_sm.items():
        shutil.copyfile(path, target / name)
    return {variable: output / path for variable, path in ENVIRONMENT.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="new directory of the copies")
    parser.add_argument(
        "--downloads", type=Path, help="download cache (default OUTPUT/downloads)"
    )
    parser.add_argument(
        "--github-env", type=Path, help="append VARIABLE=path lines to this file"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        variables = build(output, args.downloads or output / "downloads")
    except (FetchError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    lines = [f"{variable}={path}" for variable, path in variables.items()]
    if args.github_env:
        with args.github_env.open("a", encoding="utf-8") as sink:
            sink.writelines(f"{line}\n" for line in lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
