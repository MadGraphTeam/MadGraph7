"""Read-only binding of separately qualified subprocess/event cache evidence.

Physical amplitudes, tensor-key diagnostics and installed ABI checks retain
their distinct scopes. This module never opens recorded machine paths or runs
a compiler, generated executable, or numerical library.
"""
import ast
import hashlib
import json
from pathlib import PurePosixPath


PREFIX = "experiments/subprocess-cache-lifecycle/"
INSTALLER = "qualification/tirust-install/qp-transport-cache/check-installation.json"
INDEX = PREFIX + "evidence-index.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pending():
    return {"status": "pending", "evidence": None,
            "scope": "Cross-subprocess and event-boundary cache lifecycle requires separate frozen native qualification; installed ABI/public-call parity does not establish it."}


def retained_verifier(source):
    """Load the indexed pure verifier without executing its CLI or top level.

    The verifier receives only an indexed report reader. Its fixed-hash analyzer
    and smoke parser are replayed by that same reader; machine paths in native
    build metadata are never resolved.
    """
    tree = ast.parse(source.decode(), filename=PREFIX + "verify_evidence.py")
    allowed_imports = {"argparse", "ast", "collections", "gzip", "hashlib", "json",
                       "pathlib", "struct", "types"}
    selected = []
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(name in allowed_imports for name in names)
                    and (not isinstance(node, ast.ImportFrom) or node.level == 0),
                    "Unexpected lifecycle verifier import")
            selected.append(node)
        elif isinstance(node, ast.Assign):
            require(all(isinstance(target, ast.Name) for target in node.targets), "Invalid verifier constant")
            ast.literal_eval(node.value)
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "verify":
            require(not node.decorator_list, "Lifecycle verifier must be an undecorated definition")
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
            require(node.returns is None and all(arg.annotation is None for arg in arguments)
                    and not node.type_params, "Unexpected lifecycle verifier annotation")
            for default in node.args.defaults + [value for value in node.args.kw_defaults if value is not None]:
                ast.literal_eval(default)
            functions.append(node)
            selected.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        elif isinstance(node, ast.If):
            require(ast.dump(node.test) == ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body),
                    "Unexpected lifecycle verifier top-level condition")
        else:
            raise ValueError("Unexpected lifecycle verifier top-level code")
    require(len(functions) == 1, "Missing unique lifecycle verifier")
    namespace = {"__name__": "retained_lifecycle_verifier", "__file__": PREFIX + "verify_evidence.py"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), namespace["__file__"], "exec"), namespace)
    return namespace["verify"]


def collect_lifecycle(evidence, manifest):
    if not evidence.exists(INDEX):
        return pending()
    index_raw = evidence.add(INDEX)
    index = json.loads(index_raw)
    require(index.get("schema_version") == 1 and isinstance(index.get("files"), dict)
            and bool(index["files"]), "Unknown or empty lifecycle evidence index")

    def read_bytes(name):
        require(isinstance(name, str) and name and "\\" not in name,
                "Invalid lifecycle report path")
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
                "Lifecycle parser attempted access outside its indexed evidence")
        if name == "evidence-index.json":
            return index_raw
        entry = index["files"].get(name)
        require(isinstance(entry, dict) and entry.get("kind") == "file",
                "Lifecycle parser requested an unbound or nonregular file: " + name)
        raw = evidence.add(PREFIX + name, entry["sha256"])
        require(len(raw) == entry["size"], "Lifecycle evidence size differs: " + name)
        return raw

    def read_json(name):
        def invalid(value):
            raise ValueError("Nonfinite lifecycle evidence JSON: " + value)
        return json.loads(read_bytes(name), parse_constant=invalid)

    verifier = retained_verifier(read_bytes("verify_evidence.py"))
    proof = verifier(read_bytes=read_bytes)
    require(proof.get("passed") is True and proof.get("index_sha256") == digest(index_raw),
            "Lifecycle portable verification did not pass this evidence index")
    summary = read_json("attempt5/qualification-summary.json")
    require(proof.get("summary") == summary and summary.get("schema_version") == 1
            and summary.get("kind") == "tirust-generated-subprocess-cache-lifecycle"
            and summary.get("passed") is True, "Lifecycle qualification summary differs from verified evidence")
    identity = validate_candidate(read_json("materialized-source.json"), read_json("source-freeze.json"),
                                  evidence.json(INSTALLER), manifest)
    require(summary["implementation_revision"] == identity["implementation_revision"],
            "Lifecycle qualification names a different implementation")
    require(proof.get("physical_workers") == 4 and proof.get("measured_physical_calls") == 76
            and proof.get("tensor_key_calls") == 96 and proof.get("card_smoke_passed") is True
            and proof.get("historical_symlinks_are_metadata_only") is True,
            "Lifecycle verifier returned a different qualification scope")
    return {"status": "passed", "evidence": INDEX, "index_sha256": digest(index_raw),
            "summary_evidence": PREFIX + "attempt5/qualification-summary.json",
            "identity": identity, "qualification": summary,
            "verification": {key: value for key, value in proof.items() if key != "summary"},
            "scope": "Four fresh workers each evaluate 19 checked native-DP physical calls and 24 separate DP/QP tensor-key requests across two subprocess prefixes. Coefficients, status, accuracy, inputs and IEEE state agree exactly with caching enabled/disabled and observation enabled/disabled. Each observed worker records 38 automatic physical-boundary clears and two context generations. QP coverage is confined to tensor-key diagnostics; this N≤3, rank≤2 experiment makes no physical-QP-rescue, rank-six, EW, poles-only or timing claim."}


def validate_candidate(materialized, source_freeze, installed, manifest):
    """Compare retained identities; absolute metadata paths are never opened."""
    candidate = manifest["candidate"]
    production = dict(candidate["exporter_identity"]["files"])
    require(production and "madgraph/iolibs/helas_call_writers.py" in production,
            "Lifecycle report requires the registered exporter source closure")
    for name, identity in candidate["numerical_sources"].items():
        checksum = identity["sha256"]
        require(name not in production or production[name] == checksum,
                "Registered lifecycle source identities disagree: " + name)
        production[name] = checksum
    for name, checksum in production.items():
        observed = materialized["files"].get(name, {})
        # The registered exporter inventory hashes resolved source contents and
        # uses null for its two historical dangling links. These are inert JSON
        # descriptions here; no recorded link or absolute machine path is read.
        require(observed.get("kind") in ("file", "link") and observed.get("sha256") == checksum
                and (checksum is not None or observed.get("kind") == "link")
                and (observed.get("kind") != "link" or isinstance(observed.get("target"), str)),
                "Lifecycle proof differs from registered production source: " + name)
    require(materialized["revision"] == source_freeze["implementation_revision"],
            "Lifecycle materialized source and original freeze differ")

    artifacts = source_freeze["installation"]
    roots = [path[:-len("lib/libmadloop_tirust.so")] for path in artifacts
             if path.endswith("/lib/libmadloop_tirust.so")]
    require(len(roots) == 1, "Lifecycle installation must identify one native package")
    prefix = roots[0]
    required = ("include/mtirust.f90", "include/tirust.h", "lib/libmadloop_tirust.so",
                "lib/libtirust.a", "lib/tirust-config.json", "lib/tirust-link-before.txt",
                "lib/tirust-native-libs.txt")
    native = candidate["native"]["files"]
    for name in required:
        require(name in native and artifacts.get(prefix + name, {}).get("sha256") == native[name],
                "Lifecycle proof differs from registered native artifact: " + name)
    for name, key in (("Cargo.lock", "lock_sha256"), ("Cargo.toml", "manifest_sha256")):
        # Installation resolves the isolated dependencies, so its lockfile is
        # bound to the installed build proof, not the repository's source lock.
        require(native.get(name) == installed["build_provenance"][key],
                "Lifecycle proof differs from registered native build input: " + name)
    require(set(native) == set(required) | {"Cargo.lock", "Cargo.toml"},
            "Lifecycle proof does not cover the complete registered native inventory")
    for name in ("libmadloop_tirust.so", "libtirust.a", "mtirust.mod"):
        observed = artifacts.get(prefix + "lib/" + name, {})
        expected = installed["native_artifacts"][name]
        require(observed.get("sha256") == expected["sha256"]
                and observed.get("size") == expected["bytes"],
                "Lifecycle proof differs from installed ABI qualification: " + name)
    return {"registered_production_entries": len(production),
            "registered_production_content_hashes": sum(value is not None for value in production.values()),
            "implementation_revision": materialized["revision"],
            "library_sha256": native["lib/libmadloop_tirust.so"],
            "fortran_archive_sha256": native["lib/libtirust.a"],
            "fortran_module_sha256": installed["native_artifacts"]["mtirust.mod"]["sha256"]}
