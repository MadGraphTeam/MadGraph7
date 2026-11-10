#!/usr/bin/env python3
"""Read-only path/layout check; never invokes Cargo, a compiler, or a reducer."""
import ast
import json
import os
from pathlib import Path
import runpy


def check():
    here = Path(__file__).resolve().parent
    builder = runpy.run_path(str(here / "build.py"))
    native = json.loads((here / "native-inputs.json").read_text())
    selected = builder["locked_cache_paths"](native)
    home = Path(native["environment"]["CARGO_HOME"])
    missing, symlinks = [], []
    count = size = 0
    for rel in selected["trees"]:
        root = home / rel
        if not root.is_dir():
            missing.append(rel)
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(base) / name
                if path.is_symlink():
                    symlinks.append(str(path.relative_to(home)))
            for name in files:
                path = Path(base) / name
                if path.is_file():
                    count += 1
                    size += path.stat().st_size
    for rel in selected["files"]:
        path = home / rel
        if not path.is_file():
            missing.append(rel)
        elif path.is_symlink():
            symlinks.append(rel)
        else:
            count += 1
            size += path.stat().st_size
    origins = {rel: (home / rel).read_text().count("url = file://" + str(home / db))
               for rel, db in selected["private_git_origins"].items()}
    covered = all(any(Path(pkg["root"]).is_relative_to(home / root)
                      for root in selected["trees"])
                  for pkg in native["external_packages"])
    unrelated = any("gammaloop" in path for path in selected["trees"] + selected["files"])
    if missing or symlinks or not covered or unrelated or any(n != 1 for n in origins.values()):
        raise ValueError("Locked-cache selection failed its path/layout checks")
    ast.parse((here / "build.py").read_text())
    return {"external_packages": len(native["external_packages"]),
            "all_package_roots_covered": covered, "selected_tree_count": len(selected["trees"]),
            "selected_file_count": len(selected["files"]), "selected_regular_files": count,
            "selected_bytes": size, "missing": missing, "symlinks": symlinks,
            "origin_matches": origins, "unrelated_gammaloop_present": unrelated,
            "selection": selected, "compiler_cargo_or_native_invoked": False}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True))
