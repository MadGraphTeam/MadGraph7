"""Generate one Sphinx page per madspace class, for the C++ and Python APIs.

Run from conf.py after the Doxygen XML is produced. Writes
``<source>/madspace/cpp/<Class>.rst`` (breathe ``doxygenclass`` /
``doxygenstruct``) and ``<source>/madspace/python/<Class>.rst``
(``autoclass``); the ``*-api.rst`` landing pages pull these in with a globbed
toctree. Both output directories are wiped first so a removed class does not
leave a stale page behind.
"""

import argparse
import inspect
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Top-level classes in the madspace namespace only; nested types (Foo::Bar)
# are documented within their parent's page via :members:.
_TOPLEVEL = re.compile(r"^madspace::[A-Za-z_][A-Za-z0-9_]*$")

_CPP_OPTIONS = (":members:", ":undoc-members:")
_PY_OPTIONS = (":members:", ":undoc-members:", ":show-inheritance:")


def _page(title: str, directive: str, target: str, options) -> str:
    body = "".join(f"   {opt}\n" for opt in options)
    return f"{title}\n{'=' * len(title)}\n\n.. {directive}:: {target}\n{body}"


def _replace_dir(path: Path, pages: dict[str, str]) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for name, text in pages.items():
        (path / f"{name}.rst").write_text(text)


def _cpp_classes(xml_dir: Path):
    index = xml_dir / "index.xml"
    if not index.is_file():
        print(f"generate_api_pages: {index} missing; skipping C++ API")
        return []
    root = ET.parse(index).getroot()
    found = []
    for compound in root.findall("compound"):
        kind = compound.get("kind")
        name = compound.findtext("name", "")
        if kind in ("class", "struct") and _TOPLEVEL.match(name):
            found.append((name.split("::")[-1], name, kind))
    return sorted(found)


def _python_classes():
    try:
        import madspace
    except Exception as exc:  # noqa: BLE001 - want the doc build to continue
        print(
            f"generate_api_pages: cannot import madspace ({exc}); skipping Python API"
        )
        return []
    names = {
        name
        for name, obj in inspect.getmembers(madspace, inspect.isclass)
        if getattr(obj, "__module__", "").startswith("madspace")
        and not name.startswith("_")
    }
    return sorted(names)


def generate_api_pages(source_dir, xml_dir) -> None:
    base = Path(source_dir) / "madspace"

    cpp_pages = {
        short: _page(
            short,
            "doxygenstruct" if kind == "struct" else "doxygenclass",
            full,
            _CPP_OPTIONS,
        )
        for short, full, kind in _cpp_classes(Path(xml_dir))
    }
    _replace_dir(base / "cpp", cpp_pages)

    py_pages = {
        name: _page(name, "autoclass", f"madspace.{name}", _PY_OPTIONS)
        for name in _python_classes()
    }
    _replace_dir(base / "python", py_pages)

    print(
        f"generate_api_pages: wrote {len(cpp_pages)} C++ and "
        f"{len(py_pages)} Python class pages"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="source", type=Path)
    parser.add_argument("--xml-dir", default="build/doxygenxml", type=Path)
    args = parser.parse_args()
    generate_api_pages(args.source_dir, args.xml_dir)
