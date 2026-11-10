#!/usr/bin/env python3
"""Private observer builder proposal. --plan is read-only; --build is explicit.

No amplitude is called here. No installer, publication, production make, shared
Cargo target, or numerical cache is modified. Native execution belongs to the
separate controller after review of a complete build manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
SOURCE_INDEX = "b0b0af81739be886420450459eeed94a839c71e8aeef22b17c79d3b510c2cd1b"
SNAPSHOT_SHA = "2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759"
DSO_SHA = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
CASE_NAMES = ("qcd_n3_uux_zgg", "ew_n3_epem_ttxa")
CHANGED = ("check_sa.f", "loop_matrix.f", "TIR_interface.f", "TIRUST_interface.f")
MAX_SECONDS = 1800


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).absolute()
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "sha256": digest(path), "size_bytes": path.stat().st_size,
            "symlink": os.readlink(path) if path.is_symlink() else None}


def check_pin(value):
    p = Path(value["path"])
    if not p.is_absolute() or not p.is_file() or digest(p) != value["sha256"]:
        raise ValueError("Changed or absent input: " + str(p))
    current = {"resolved_path": str(p.resolve()), "size_bytes": p.stat().st_size,
               "symlink": os.readlink(p) if p.is_symlink() else None}
    for key in ("resolved_path", "size_bytes", "symlink"):
        if key in value and current[key] != value[key]:
            raise ValueError("Input identity changed: " + str(p))
    return p


def gather_pins(value):
    found = {}
    def visit(x):
        if isinstance(x, dict):
            if {"path", "sha256"} <= x.keys():
                old = found.setdefault(x["path"], x)
                if old["sha256"] != x["sha256"]:
                    raise ValueError("Conflicting input identities")
            else:
                for v in x.values():
                    visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(value)
    return found


def tree_map(path, ignored=(".git", "target", "references", "__pycache__", ".direnv")):
    path = Path(path)
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*"))
            if p.is_file() and not any(x in ignored
                                       for x in p.relative_to(path).parts)}


def verify_tree(record):
    root = Path(record["root"])
    for p in root.rglob("*"):
        if (not any(x in record["ignored_names_at_any_depth"] for x in p.relative_to(root).parts)
                and p.is_symlink()):
            raise ValueError("Unexpected source symlink: " + str(p))
    actual = tree_map(record["root"], record["ignored_names_at_any_depth"])
    if actual != record["files"]:
        raise ValueError("Source tree changed: " + record["root"])
    if hashlib.sha256(json.dumps(actual, sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest() != record["tree_sha256"]:
        raise ValueError("Source tree receipt is inconsistent")


def load_inputs(expected_plan=None):
    if digest(PROPOSAL / "index.json") != SOURCE_INDEX:
        raise ValueError("Frozen source proposal changed")
    index = read(PROPOSAL / "index.json")
    for name, record in index["files"].items():
        p = PROPOSAL / name
        if p.is_symlink() or digest(p) != record["sha256"] or p.stat().st_size != record["bytes"]:
            raise ValueError("Frozen proposal payload changed: " + name)
    snapshot_path = PROPOSAL / "preparation/source-1/snapshot.json"
    if digest(snapshot_path) != SNAPSHOT_SHA:
        raise ValueError("Snapshot changed")
    if expected_plan is not None and digest(HERE/"build-plan.json") != expected_plan:
        raise ValueError("Build plan differs from explicitly selected source receipt")
    snapshot, native, plan = read(snapshot_path), read(HERE / "native-inputs.json"), read(HERE / "build-plan.json")
    if digest(HERE / "native-inputs.json") != plan["native_inputs_sha256"]:
        raise ValueError("Native input metadata changed")
    if digest(HERE / "run.py") != plan["lifecycle_helper_sha256"]:
        raise ValueError("Reviewed lifecycle helper changed")
    if tuple(snapshot["cases"]) != tuple(sorted(CASE_NAMES)):
        raise ValueError("Unexpected case set")
    for record in gather_pins(snapshot).values():
        check_pin(record)
    for record in native["source_trees"].values():
        verify_tree(record)
    for record in native["external_packages"]:
        verify_tree(record)
    for record in native["bridge_files"].values():
        check_pin(record)
    check_pin(native["installed_lock"])
    for record in native["tools"].values():
        check_pin(record)
    for path, expected in native["archived_records"].items():
        if digest(path) != expected:
            raise ValueError("Archived build record changed: " + path)
    for record in plan["toolchain_link_inputs"].values():
        check_pin(record)
    for record in plan["compiler_support_files"].values():
        check_pin(record)
    return snapshot, native, plan


def native_pins(native):
    result = {}
    for record in list(native["source_trees"].values()) + native["external_packages"]:
        for rel, expected in record["files"].items():
            p = str(Path(record["root"])/rel)
            result[p] = {"path": p, "sha256": expected}
    result.update(gather_pins(native["bridge_files"]))
    result.update(gather_pins(native["tools"]))
    result.update(gather_pins(native["installed_lock"]))
    result.update({str((ROOT/path).resolve()): {"path": str((ROOT/path).resolve()), "sha256": expected}
                   for path, expected in native["archived_records"].items()})
    return result


def locked_cache_paths(native):
    """Select only sources needed by the frozen lock and patched manifest."""
    home = Path(native["environment"]["CARGO_HOME"])
    trees, files, origins = set(), set(), {}
    for package in native["external_packages"]:
        rel = Path(package["root"]).relative_to(home)
        if package["source"].startswith("registry+"):
            if len(rel.parts) != 4 or rel.parts[:2] != ("registry", "src"):
                raise ValueError("Unexpected locked registry source layout")
            registry, directory = rel.parts[2:]
            name = package["name"].lower()
            shard = ("1/"+name if len(name) == 1 else "2/"+name if len(name) == 2
                     else "3/"+name[0]+"/"+name if len(name) == 3
                     else name[:2]+"/"+name[2:4]+"/"+name)
            trees.add(rel)
            files.update((Path("registry/cache")/registry/(directory+".crate"),
                          Path("registry/index")/registry/"config.json",
                          Path("registry/index")/registry/".cache"/shard))
        elif package["source"].startswith("git+"):
            if len(rel.parts) < 4 or rel.parts[:2] != ("git", "checkouts"):
                raise ValueError("Unexpected locked git source layout")
            checkout = Path(*rel.parts[:4])
            database = Path("git/db")/rel.parts[2]
            trees.update((checkout, database))
            origins[checkout/".git/config"] = database
        else:
            raise ValueError("Unexpected locked external source")
    # This manifest origin is replaced by the private, patched path packages.
    # Keep its exact pinned checkout available for Cargo's offline resolution;
    # it is not admitted as a numerical dependency by the graph check below.
    checkout = Path("git/checkouts/tirust-b799601ea2755a53/854b79d")
    database = Path("git/db/tirust-b799601ea2755a53")
    trees.update((checkout, database))
    origins[checkout/".git/config"] = database
    return {"trees": sorted(map(str, trees)), "files": sorted(map(str, files)),
            "private_git_origins": {str(k): str(v) for k, v in sorted(origins.items())}}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Build:
    def __init__(self, work, seconds, native):
        self.work = work
        self.deadline = time.monotonic() + seconds
        self.native = native
        self.events = []
        self.pending = None
        self.old_handlers = {}
        # Reconstruct the frozen, disclosed NEW diagnostic environment. Do not
        # inherit ephemeral caller PATH, compiler wrappers, flags or loader hooks.
        self.env = dict(native["environment"])
        self.lifecycle = runpy.run_path(str(HERE/"run.py"))
        self.owned_deadline = self.lifecycle["Deadline"](seconds)

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Build interrupted by signal " + str(self.pending))
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Whole private build deadline reached")

    def install_handlers(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self.old_handlers[sig] = signal.signal(sig, lambda s, frame: setattr(self, "pending", s))

    def restore_handlers(self):
        for sig, handler in self.old_handlers.items():
            signal.signal(sig, handler)

    def command(self, argv, cwd, name, env=None):
        self.checkpoint()
        log = self.work / "logs" / (name + ".txt")
        log.parent.mkdir(exist_ok=True)
        event = {"argv": list(map(str, argv)), "cwd": str(cwd), "log": str(log),
                 "returncode": None, "interrupted": False, "timed_out": False}
        self.events.append(event)
        child = self.lifecycle["OwnedChild"](self.owned_deadline, event)
        original_error = None
        try:
            with log.open("xb") as out:
                child.spawn(argv, cwd, env or self.env, out)
                child.process.stdin.close()
                while not child.exited():
                    self.checkpoint()
                    time.sleep(min(.1, self.owned_deadline.check()))
                event["returncode"] = child.finish()
                self.checkpoint()
        except BaseException as exc:
            original_error = exc
            event["error"] = repr(exc)
            event["interrupted"] = self.pending is not None
            event["timed_out"] = isinstance(exc, TimeoutError)
            child.cleanup_after(exc)
            raise
        finally:
            try:
                if log.exists():
                    event["log_sha256"] = digest(log)
                dump(self.work / "commands.json", self.events)
            except BaseException as metadata_error:
                if original_error is None:
                    raise
                original_error.add_note("Build command metadata failed: "+repr(metadata_error))
        if event["returncode"] != 0:
            raise RuntimeError("Build command failed: " + name)
        return log.read_text(errors="replace")

    def copy_file(self, source, destination):
        self.checkpoint()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if digest(source) != digest(destination):
                raise ValueError("Copy collision: " + str(destination))
        else:
            with Path(source).open("rb") as src, destination.open("xb") as dst:
                for block in iter(lambda: src.read(1024*1024), b""):
                    self.checkpoint()
                    dst.write(block)
            shutil.copymode(source, destination)

    def copy_tree(self, source, destination, exclude=()):
        source = Path(source)
        for directory, dirs, files in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in exclude)
            if any((Path(directory)/d).is_symlink() for d in dirs):
                raise ValueError("Directory symlink in copied Cargo cache")
            for name in sorted(files):
                p = Path(directory) / name
                if name in exclude or not p.is_file():
                    continue
                if p.is_symlink():
                    raise ValueError("File symlink in copied Cargo cache: " + str(p))
                self.copy_file(p, destination / p.relative_to(source))

    def tool(self, name):
        return self.native["tools"][name]["path"]


def native_build(build, snapshot, native):
    work = build.work
    bridge = work / "native/bridge"
    upstream = work / "native/tirust"
    for rel, expected in native["source_trees"]["tirust"]["files"].items():
        build.copy_file(Path(native["source_trees"]["tirust"]["root"])/rel, upstream/rel)
        if digest(upstream/rel) != expected:
            raise ValueError("Private upstream copy changed")
    for rel, expected in native["source_trees"]["pole_kernels"]["files"].items():
        build.copy_file(Path(native["source_trees"]["pole_kernels"]["root"])/rel,
                        upstream/"madloop-pole-kernels"/rel)
        if digest(upstream/"madloop-pole-kernels"/rel) != expected:
            raise ValueError("Private pole-kernel copy changed")
    for rel, record in native["bridge_files"].items():
        build.copy_file(record["path"], bridge/rel)
        if digest(bridge/rel) != record["sha256"]:
            raise ValueError("Private bridge copy changed")
    build.copy_file(native["installed_lock"]["path"], bridge/"Cargo.lock")
    build.command([build.tool("patch"), "-p1", "--batch", "--forward", "--fuzz=0", "-i",
                   str(bridge/"patches/0001-direct-poles-only.patch")], upstream, "native-patch")
    (bridge/".cargo").mkdir()
    (bridge/".cargo/config.toml").write_text('[patch."https://github.com/huashengshao/TIRust"]\n' +
        ''.join(name+' = { path = '+json.dumps(str(upstream/"crates"/name))+' }\n'
                for name in ("iregi-rust", "iregi-masters")))
    # Execute only the already hash-bound pure transformer, never its CLI.
    transformer = runpy.run_path(str(PROPOSAL/"native/transform.py"))
    original = (bridge/"src/lib.rs").read_bytes()
    changed = transformer["transform"](original)
    if transformer["restore"](changed) != original:
        raise ValueError("Native insertion erasure failed")
    (bridge/"src/lib.rs").write_bytes(changed)
    build.copy_file(PROPOSAL/"native/observer.rs", bridge/"src/observer.rs")
    build.copy_file(PROPOSAL/"native/observer_flags.c", bridge/"observer_flags.c")
    cargo_home = work/"cargo-home"
    # Copy the locked graph, not unrelated checkouts in the shared Cargo home.
    # All destinations are private; shared cache contents are read-only inputs.
    selection = locked_cache_paths(native)
    original_home = Path(native["environment"]["CARGO_HOME"])
    for rel in selection["trees"]:
        build.copy_tree(original_home/rel, cargo_home/rel)
    for rel in selection["files"]:
        build.copy_file(original_home/rel, cargo_home/rel)
    for rel, database in selection["private_git_origins"].items():
        config = cargo_home/rel
        old = "url = file://"+str(original_home/database)
        new = "url = file://"+str(cargo_home/database)
        text = config.read_text()
        if text.count(old) != 1:
            raise ValueError("Unexpected copied git checkout origin: "+rel)
        config.write_text(text.replace(old, new))
    private_packages = []
    for package in native["external_packages"]:
        rel = Path(package["root"]).relative_to(native["environment"]["CARGO_HOME"])
        private = {**package, "root": str(cargo_home/rel)}
        verify_tree(private)
        private_packages.append(private)
    env = dict(build.env)
    env.update(CARGO_HOME=str(cargo_home), CARGO_TARGET_DIR=str(work/"cargo-target"),
               CARGO_BUILD_BUILD_DIR=str(work/"cargo-build"), CARGO_BUILD_JOBS="1",
               CARGO_NET_OFFLINE="true", RUSTC=build.tool("rustc"), CC=build.tool("cc"),
               AR=build.tool("ar"), RUST_TEST_THREADS="1", RAYON_NUM_THREADS="1")
    for name, rel in (("XDG_CACHE_HOME", "xdg-cache"), ("XDG_CONFIG_HOME", "xdg-config"),
                      ("SYMBOLICA_LOCK_DIR", "symbolica-locks"), ("TMPDIR", "tmp"),
                      ("GMP_MPFR_SYS_CACHE", "gmp-cache")):
        private = work/rel
        private.mkdir()
        env[name] = str(private)
    metadata_text = build.command([build.tool("cargo"), "metadata", "--offline", "--locked",
                                   "--format-version=1"], bridge, "cargo-metadata", env)
    lines = [line for line in metadata_text.splitlines() if line.startswith('{"packages":')]
    if len(lines) != 1:
        raise ValueError("Missing unique locked Cargo dependency metadata")
    packages = json.loads(lines[0])["packages"]
    external = {(p["name"], p["version"], p["source"]): Path(p["manifest_path"]).parent for p in packages
                if p["source"] is not None}
    expected = {(p["name"], p["version"], p["source"]): Path(p["root"]) for p in private_packages}
    if external != expected or {p["name"] for p in packages if p["source"] is None} != {
            "madloop-tirust", "iregi-rust", "iregi-masters", "madloop-pole-kernels"}:
        raise ValueError("Resolved private dependency graph differs from installed locked graph")
    helper = bridge/"observer_flags.o"
    build.command([build.tool("cc"), "-O0", "-fPIC", "-c", "observer_flags.c", "-o", str(helper)],
                  bridge, "observer-c-flags", env)
    map_path = work/"native/observed-link.map"
    # The helper is a final-crate link input. Global RUSTFLAGS stay unchanged.
    cargo = [build.tool("cargo"), "rustc", "--release", "--offline", "--locked", "--lib",
             "--features", "poles-only", "--", "-C", "link-arg=-Wl,-soname,libmadloop_tirust.so",
             "-C", "link-arg=-Wl,--exclude-libs,ALL", "-C", "link-arg="+str(helper),
             "-C", "link-arg=-Wl,--undefined=mpfr_flags_save",
             "-C", "link-arg=-Wl,-Map,"+str(map_path)+",--cref"]
    build.command(cargo, bridge, "observed-cargo-build", env)
    for private in private_packages:
        verify_tree(private)
    if digest(bridge/"Cargo.lock") != native["installed_lock"]["sha256"]:
        raise ValueError("Resolved lock changed")
    dso = work/"cargo-target/release/libmadloop_tirust.so"
    dynamic = build.command([build.tool("readelf"), "-d", str(dso)], work, "observed-dynamic")
    if re.search(r'NEEDED.*(?:mpfr|gmp)', dynamic, re.I):
        raise ValueError("Observer must not load an independent MPFR/GMP library")
    symbols = build.command([build.tool("readelf"), "-Ws", str(dso)], work, "observed-symbols")
    for name in ("mpfr_flags_save", "mpfr_add", "__gmpfr_flags", "ml_tirust_private_observer_fp_read"):
        rows = [line.split() for line in symbols.splitlines() if line.split() and line.split()[-1] == name]
        if len(rows) != 1 or rows[0][6] == "UND" or rows[0][4] != "LOCAL":
            raise ValueError("MPFR/helper must bind one local definition: " + name)
    map_text = map_path.read_text()
    if "Cross Reference Table" not in map_text:
        raise ValueError("Missing actual linker cross-reference table")
    cref = map_text.split("Cross Reference Table", 1)[1]
    definitions = {}
    for name in ("mpfr_flags_save", "mpfr_add", "__gmpfr_flags"):
        matches = re.findall(r'^'+re.escape(name)+r'\s+(\S+)\n((?:[ \t]+\S.*\n)*)', cref, re.M)
        if len(matches) != 1:
            raise ValueError("Missing unique MPFR cross-reference definition: "+name)
        origin, references = matches[0]
        archive = re.fullmatch(r'(.+\.(?:a|rlib))\(([^()]+)\)', origin)
        if not archive:
            raise ValueError("MPFR definition is not from the bundled numerical archive: "+origin)
        definitions[name] = {"archive": archive[1], "member": archive[2],
                             "references": references.split()}
    archives = {entry["archive"] for entry in definitions.values()}
    if len(archives) != 1 or not any("observer_flags.o" in x for x in definitions["mpfr_flags_save"]["references"]):
        raise ValueError("Observer and numerical MPFR definitions do not share one linked archive")
    if len(definitions["__gmpfr_flags"]["references"]) < 2:
        raise ValueError("Missing numerical and flags-reader shared TLS references")
    return dso, {"private_source_tree": tree_map(bridge), "patched_upstream_tree": tree_map(upstream),
                 "cache_selection": selection,
                 "cargo_environment": env_projection(env), "link_map": pin(map_path),
                 "mpfr_archive": pin(next(iter(archives))), "mpfr_cross_references": definitions,
                 "same_local_mpfr_instance": True}


def runtime(build, executable, expected_dso, env, label):
    text = build.command([build.tool("readelf"), "-l", str(executable)], build.work, label+"-elf", env)
    matches = re.findall(r'Requesting program interpreter:\s*([^\]]+)\]', text)
    if len(matches) != 1:
        raise ValueError("Missing unique PT_INTERP")
    interpreter = Path(matches[0]).resolve()
    listing = build.command([str(interpreter), "--list", str(executable)], build.work,
                            label+"-loader", env)
    if "not found" in listing:
        raise ValueError("Missing runtime dependency")
    paths = {Path(x).resolve() for x in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', listing, re.M)}
    paths.add(interpreter)
    dsos = [p for p in paths if p.name == "libmadloop_tirust.so"]
    if dsos != [expected_dso.resolve()]:
        raise ValueError("Wrong or multiple TIRust DSOs selected by actual interpreter")
    return {str(p): digest(p) for p in sorted(paths)}, pin(interpreter)


def host_build(build, snapshot, observed_dso, plan):
    cases = {}
    installed = Path(snapshot["installed_native"]["lib/libmadloop_tirust.so"]["path"])
    for case in CASE_NAMES:
        saved = snapshot["cases"][case]
        original_root = Path(saved["generated_root"])
        sub_relative = Path(saved["focus_sources"]["check_sa.f"]["path"]).parent.relative_to(original_root)
        original_sub = original_root/sub_relative
        scope = saved["effective_compiler_flags"]["scopes"]["process"]
        original_lines = scope["expanded_make_output"].splitlines()
        logfile = ROOT/"docs/benchmarkfolder/campaigns/split-dp-terminal-1/inputs/candidate"/case/"compilation-process.log"
        compilation = read(saved["compilation"]["path"])
        if digest(logfile) != compilation["build_logs"]["process"]:
            raise ValueError("Historical process command log changed")
        links = [shlex.split(line) for line in logfile.read_text().splitlines() if " -o check " in line]
        if len(links) != 1:
            raise ValueError("Ambiguous historical link command")
        case_result = {"historical_link_log": pin(logfile), "arms": {}}
        for arm in ("original", "observed"):
            clone = build.work/"cases"/case/arm
            for collection in (saved["all_sources_and_includes"], saved["existing_numerical_objects"]):
                for rel, record in collection.items():
                    build.copy_file(check_pin(record), clone/rel)
            # MODEL/DHELAS and original exported library aliases are materialized,
            # not symlinked back to mutable source trees.
            for record in saved["link_inputs"]["archives"].values():
                p = Path(record["path"])
                if p.is_relative_to(original_root):
                    build.copy_file(check_pin(record), clone/p.relative_to(original_root))
            # Export lib aliases may resolve outside the generated root.
            for p in (original_root/"lib").iterdir():
                if p.is_file():
                    resolved = str(p.resolve())
                    known = saved["link_inputs"]["archives"]
                    if resolved in known:
                        build.copy_file(check_pin(known[resolved]), clone/"lib"/p.name)
            sub = clone/sub_relative
            changed_names = CHANGED if arm == "observed" else ("check_sa.f",)
            for name in changed_names:
                (sub/name).write_bytes((PROPOSAL/"prepared"/case/name).read_bytes())
            (sub/"host_observer.f90").write_bytes((PROPOSAL/"host_observer.f90").read_bytes())
            fc = scope["recipes"]["check_sa.f"]["compiler"][0]
            check_pin(fc)
            module = sub/"observer-modules"
            module.mkdir()
            build.command([fc["path"], "-w", "-fPIC", "-J"+str(module), "-I"+str(module),
                           "-c", "host_observer.f90", "-o", "host_observer.o"], sub,
                          case+"-"+arm+"-host-module")
            for name in changed_names:
                commands = [shlex.split(line) for line in original_lines if " -c "+name+" " in line]
                if len(commands) != 1:
                    raise ValueError("Ambiguous saved compile recipe: "+name)
                argv = [x.replace(str(original_root), str(clone)) for x in commands[0]]
                argv += ["-I"+str(module), "-J"+str(module)]
                build.command(argv, sub, case+"-"+arm+"-"+Path(name).stem)
            extras = {"host_observer.o": pin(sub/"host_observer.o")}
            if arm == "original":
                build.copy_file(PROPOSAL/"observer_noop.c", sub/"observer_noop.c")
                build.command([build.tool("cc"), "-O0", "-fPIC", "-c", "observer_noop.c", "-o",
                               "observer_noop.o"], sub, case+"-noop")
                extras["observer_noop.o"] = pin(sub/"observer_noop.o")
            argv = [x.replace(str(original_root), str(clone)) for x in links[0]]
            selected = observed_dso if arm == "observed" else installed
            if arm == "observed":
                if argv.count("-lmadloop_tirust") != 1:
                    raise ValueError("Unexpected historical TIRust link")
                argv[argv.index("-lmadloop_tirust")] = str(selected)
                argv = [x.replace("-Wl,-rpath,"+str(installed.parent),
                                  "-Wl,-rpath,"+str(selected.parent)) for x in argv]
            insert = argv.index("-L"+str(clone/"lib")+"/")
            argv[insert:insert] = list(extras)
            link_map = sub/"private-link.map"
            argv += ["-Wl,-Map,"+str(link_map)]
            build.command(argv, sub, case+"-"+arm+"-link")
            runtime_env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
                           "LD_LIBRARY_PATH": os.pathsep.join(map(str, [selected.parent,
                               ROOT/"HEPTools/ninja/lib", ROOT/"HEPTools/collier", ROOT/"HEPTools/ninjars/lib"]))}
            env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
            env.update(runtime_env)
            closure, interpreter = runtime(build, sub/"check", selected, env, case+"-"+arm)
            historical_runtime = {value["resolved_path"]: value["sha256"]
                                  for value in saved["runtime"]["libraries"].values()
                                  if "ld-linux" not in Path(value["resolved_path"]).name
                                  and Path(value["resolved_path"]).name != "libmadloop_tirust.so"}
            actual_runtime = {path: expected for path, expected in closure.items()
                              if "ld-linux" not in Path(path).name and Path(path).name != "libmadloop_tirust.so"}
            if actual_runtime != historical_runtime:
                raise ValueError("Non-TIRust runtime closure changed from the frozen snapshot")
            actual_objects = {rel: pin(clone/rel) for rel in saved["existing_numerical_objects"]}
            allowed = {str(sub_relative/Path(x).with_suffix('.o')) for x in changed_names}
            for rel, record in actual_objects.items():
                if rel not in allowed and record["sha256"] != saved["existing_numerical_objects"][rel]["sha256"]:
                    raise ValueError("Unexpected numerical object change: "+rel)
            loads = [line[5:].strip() for line in link_map.read_text().splitlines() if line.startswith("LOAD ")]
            link_inputs = {}
            allowed_loads = {str(Path(v["path"]).resolve()): v["sha256"]
                             for v in plan["toolchain_link_inputs"].values()}
            for group in (saved["link_inputs"]["archives"], saved["runtime"]["libraries"]):
                allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in group.values()})
            allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in actual_objects.values()})
            allowed_loads.update({str(Path(v["path"]).resolve()): v["sha256"] for v in extras.values()})
            allowed_loads[str(selected.resolve())] = digest(selected)
            for p in (clone/"lib").iterdir():
                if p.is_file():
                    expected = {v["sha256"] for v in saved["link_inputs"]["archives"].values()}
                    if digest(p) not in expected:
                        raise ValueError("Private library copy is not a frozen archive")
                    allowed_loads[str(p.resolve())] = digest(p)
            for value in loads:
                p = Path(value)
                p = p if p.is_absolute() else sub/p
                if not p.is_file():
                    raise ValueError("Unresolved real linker input: "+value)
                if allowed_loads.get(str(p.resolve())) != digest(p):
                    raise ValueError("Unbound actual linker input: "+str(p))
                link_inputs[str(p.resolve())] = pin(p)
            case_result["arms"][arm] = {
                "executable": pin(sub/"check"), "runtime": closure, "interpreter": interpreter,
                "launch_prefix": [interpreter["path"]], "runtime_environment": runtime_env,
                "objects": actual_objects, "extra_objects": extras, "link_inputs": link_inputs,
                "link_map": pin(link_map), "sources": {rel: pin(clone/rel)
                    for rel in saved["all_sources_and_includes"]},
                "extra_sources": {"host_observer.f90": pin(sub/"host_observer.f90")},
                "compiler_check": {"status": "complete", "compiler": fc},
                "observer_enabled_at_build": False}
            if arm == "original":
                case_result["arms"][arm]["extra_sources"]["observer_noop.c"] = pin(sub/"observer_noop.c")
        cases[case] = case_result
    return cases


def env_projection(env):
    exact = {"PATH", "CC", "CXX", "FC", "AR", "RUSTC", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
             "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "GFORTRAN_INCLUDE_PATH", "LIBRARY_PATH",
             "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "MAKEFLAGS", "OMP_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "LANG", "LC_ALL",
             "PYTHONDONTWRITEBYTECODE", "RUST_MIN_STACK", "RUST_TEST_THREADS", "SYMBOLICA_HIDE_BANNER",
             "SYMBOLICA_LOCK_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR", "CFLAGS", "CPPFLAGS",
             "CXXFLAGS", "FFLAGS", "GMP_MPFR_SYS_CACHE", "NUMEXPR_NUM_THREADS", "PKG_CONFIG_LIBDIR",
             "PKG_CONFIG_PATH", "PKG_CONFIG_SYSROOT_DIR"}
    return {k: v for k, v in sorted(env.items()) if k in exact or k.startswith(("NIX_", "LD_", "CARGO_"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--label", default="attempt1")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--expected-plan-sha")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from the repository root")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", args.label):
        raise ValueError("Unsafe label")
    if args.build and (args.expected_plan_sha is None or not re.fullmatch('[0-9a-f]{64}', args.expected_plan_sha)):
        raise ValueError("Explicit build requires --expected-plan-sha from the reviewed source receipt")
    snapshot, native, plan = load_inputs(args.expected_plan_sha)
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    # Root approval is an external task instruction, not a fabricated approval file.
    # This explicit entry point alone does not claim that approval was obtained.
    root_work = (args.work_root or ROOT/".tirust-build/dp-rescue-cause-observer"/args.label).absolute()
    if not root_work.is_relative_to(ROOT/".tirust-build") or root_work.resolve() != root_work:
        raise ValueError("Private build must be an unaliased ignored work directory")
    work = root_work/"build"
    work.mkdir(parents=True, exist_ok=False)
    build = Build(work, MAX_SECONDS, native)
    record = {"schema": "dp-rescue-observer-build-v1", "status": "started",
              "source_index_sha256": SOURCE_INDEX, "build_plan_sha256": digest(HERE/"build-plan.json"),
              "builder_sha256": digest(__file__), "native_inputs_sha256": digest(HERE/"native-inputs.json"),
              "cpu": 11, "deadline_seconds": MAX_SECONDS, "cases": {}, "artifacts": {},
              "historical_environment_reproduced": False, "old_dso_byte_reproduction_claim": False,
              "numerical_execution_performed": False, "cause_attribution_qualified": False}
    inputs = gather_pins(snapshot)
    inputs.update(native_pins(native))
    inputs.update(gather_pins(plan["toolchain_link_inputs"]))
    inputs.update(gather_pins(plan["compiler_support_files"]))
    for source in (HERE/"native-inputs.json", HERE/"build-plan.json", HERE/"run.py", Path(__file__)):
        inputs[str(source)] = pin(source)
    build.install_handlers()
    original_error = None
    try:
        os.sched_setaffinity(0, {11})
        if any(key in build.env for key in native["absent_environment_keys"]):
            raise ValueError("Frozen build environment includes an allegedly absent override")
        if any(os.environ.get(k) for k in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "RUSTC_WRAPPER",
                                          "RUSTC_WORKSPACE_WRAPPER")):
            raise ValueError("Unsupported compiler/runtime interception")
        record["protected_before"] = {p: pin(p) for p in inputs}
        observed_dso, native_result = native_build(build, snapshot, native)
        record["native"] = native_result
        record["observed_dso"] = pin(observed_dso)
        record["original_dso"] = snapshot["installed_native"]["lib/libmadloop_tirust.so"]
        if record["original_dso"]["sha256"] != DSO_SHA:
            raise ValueError("Original installed DSO identity changed")
        record["cases"] = host_build(build, snapshot, observed_dso, plan)
        build.checkpoint()
        record["status"] = "complete"
    except BaseException as exc:
        original_error = exc
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        try:
            record["protected_after"] = {p: pin(p) for p in inputs}
            if record.get("protected_before") != record["protected_after"]:
                record["status"] = "failed"
                record["protected_inputs_unchanged"] = False
            else:
                record["protected_inputs_unchanged"] = True
            # Includes source inputs, module outputs, exact argv/logs, maps and binaries.
            record["artifacts"] = {str(p): pin(p) for p in sorted(work.rglob("*"))
                                   if p.is_file() and p.name != "build.json"}
            record["commands"] = build.events
            dump(work/"build.json", record)
        except BaseException as metadata_error:
            record["status"] = "failed"
            record["postprocessing_error"] = repr(metadata_error)
            try:
                dump(work/"build.json", record)
            except BaseException as second_error:
                metadata_error.add_note("Failure manifest also failed: "+repr(second_error))
            if original_error is None:
                raise
            original_error.add_note("Build finalization failed: "+repr(metadata_error))
        finally:
            build.restore_handlers()
    if record["status"] != "complete":
        raise RuntimeError("Private build did not qualify")
    print(json.dumps({"manifest": str(work/"build.json"), "sha256": digest(work/"build.json")}))


if __name__ == "__main__":
    main()
