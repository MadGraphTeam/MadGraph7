"""Prove historical objects still reproduce their recorded executable."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _link_recipe(text):
    recipes = [shlex.split(line) for line in text.splitlines() if line.strip()]
    if len(recipes) != 1:
        raise ValueError("Object replay requires exactly one linker recipe")
    command = recipes[0]
    if ("-c" in command or command.count("-o") != 1
            or command[command.index("-o") + 1:] == []
            or command[command.index("-o") + 1] != "check"
            or any(token in {";", "&&", "||", "|", ">", ">>", "<"} for token in command)):
        raise ValueError("Object replay refuses compilation or compound recipes")
    if not any(Path(token).suffix == ".o" for token in command):
        raise ValueError("Object replay recipe has no recorded objects")
    return command


def prove_object_relink(directory, expected_sha256, destination, env):
    """Link unchanged objects with original paths, writing only to destination.

    Exact executable equality binds the existing object files to the historical
    executable, even when the old build predates individual object inventories.
    A nonreproducible link is not accepted; use a fresh source build instead.
    """
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original = directory / "check"
    if _digest(original) != expected_sha256:
        raise ValueError("Historical executable changed before object replay")
    objects = {path.name: _digest(path) for path in sorted(directory.glob("*.o"))}
    if not objects or "check_sa.o" not in objects:
        raise ValueError("Historical process objects are incomplete")
    # Suppress every existing numerical dependency, including the generated
    # COLLIER symlink recipe. Never execute make itself beyond its dry run.
    frozen = [path.name for path in sorted(directory.iterdir())
              if path.is_file() and path.suffix in (".o", ".so", ".a", ".dylib")]
    frozen += [str(path) for path in sorted((directory.parents[1] / "lib").glob("*.a"))]
    make = ["make", "--no-print-directory", "--dry-run", "--always-make",
            *("--old-file=" + name for name in frozen), "check"]
    prepared = subprocess.run(make, cwd=directory, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    (destination / "make.log").write_text(prepared.stdout + prepared.stderr)
    command = _link_recipe(prepared.stdout)
    recorded_command = list(command)
    command[command.index("-o") + 1] = str(destination / "check")
    record = {"expected_executable_sha256": expected_sha256, "objects": objects,
              "protocol_sha256": _digest(__file__),
              "working_directory": str(directory), "make_command": make,
              "original_link_command": recorded_command, "proof_link_command": command}
    with (destination / "link.log").open("w") as log:
        subprocess.run(command, cwd=directory, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    record["relinked_executable_sha256"] = _digest(destination / "check")
    record["objects_unchanged"] = objects == {
        path.name: _digest(path) for path in sorted(directory.glob("*.o"))}
    record["original_executable_unchanged"] = _digest(original) == expected_sha256
    record["passed"] = (record["objects_unchanged"] and record["original_executable_unchanged"]
                         and record["relinked_executable_sha256"] == expected_sha256)
    (destination / "proof.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if not record["passed"]:
        raise ValueError("Historical objects do not reproduce the executable; require a full source rebuild")
    return record
