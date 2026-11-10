"""Static directory-order analysis; inspected ELF/DSO files are never invoked."""
from pathlib import Path
import hashlib
import itertools
import json
import stat


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    prior = here.parent / "run2-closure-audit-proposal-1"
    source = here.parent / "source-proposal2"
    expected_prior_index = "1dd2b40762df401685288e3579705a22c6a22d7020970ea90e7afe800dd2b3b6"
    if sha(prior / "index.json") != expected_prior_index:
        raise ValueError("Prior static audit seal mismatch")
    audit = json.loads((prior / "audit.json").read_text())
    binding = json.loads((source / "bindings.json").read_text())
    directories = binding["native_library_path"].split(":")
    sonames = ["libquadmath.so.0", "libstdc++.so.6", "libgcc_s.so.1"]
    expected = {x["soname"]: x for x in audit["expected_runtime"]}
    matrix, candidates = {}, {}
    for soname in sonames:
        rows = []
        for directory in directories:
            path = Path(directory) / soname
            row = {"directory": directory, "candidate": str(path), "present": path.exists()}
            if row["present"]:
                row.update(resolved_path=str(path.resolve(strict=True)), sha256=sha(path))
                row["matches_expected_bytes"] = row["sha256"] == expected[soname]["sha256"]
            rows.append(row)
        matrix[soname] = rows
        candidates[soname] = {r["directory"]: r for r in rows if r["present"]}
    successful = []
    tested = 0
    for order in itertools.permutations(directories):
        tested += 1
        choice = {name: next((candidates[name][d] for d in order if d in candidates[name]), None)
                  for name in sonames}
        if all(r is not None and r["matches_expected_bytes"] for r in choice.values()):
            successful.append(order)
    constraints = []
    for name, rows in matrix.items():
        valid = [r["directory"] for r in rows if r.get("matches_expected_bytes")]
        wrong = [r["directory"] for r in rows if r["present"] and not r["matches_expected_bytes"]]
        if len(valid) == 1:
            constraints.extend({"soname": name, "before": valid[0], "after": d} for d in wrong)
    cycles = [{"left": a, "right": b} for i, a in enumerate(constraints)
              for b in constraints[i + 1:] if a["before"] == b["after"] and a["after"] == b["before"]]
    pins = {r["soname"]: {"canonical_path": r["resolved_path"], "sha256": r["sha256"]}
            for r in audit["expected_runtime"]}
    if len(pins) != 15 or len({p["canonical_path"] for p in pins.values()}) != 15:
        raise ValueError("Expected fifteen unique original providers")
    all_records = audit["expected_runtime"] + [audit["diagnostic"]]
    no_rpath = all(r["rpath"] is None for r in all_records)
    basenames = all(Path(name).name == name for r in all_records for name in r["needed"])
    all_needed = sorted({n for r in all_records for n in r["needed"]})
    executable_mode = Path(audit["diagnostic"]["resolved_path"]).stat().st_mode
    result = {
        "schema": "qcd-n3-donor-closure-resolution-proposal-1", "static_only": True,
        "native_or_compiler_or_loader_calls": 0, "prior_static_audit_index_sha256": expected_prior_index,
        "binding_sha256": sha(source / "bindings.json"), "directory_order": directories,
        "candidate_matrix": matrix, "permutations_tested": tested,
        "successful_directory_orders": successful, "necessary_order_constraints": constraints,
        "contradictory_order_pairs": cycles,
        "proposed_shim_pins": pins,
        "mechanism_checks": {"original_provider_count": len(pins), "all_expected_providers_and_fresh_executable_have_no_DT_RPATH": no_rpath,
                             "all_DT_NEEDED_names_are_basenames": basenames,
                             "all_DT_NEEDED_names_have_shim_aliases": set(all_needed) <= set(pins),
                             "fresh_executable_has_no_setuid_or_setgid_bit": not bool(executable_mode & (stat.S_ISUID | stat.S_ISGID)),
                             "fresh_executable_PT_INTERP_is_original_pinned_loader": audit["diagnostic"]["pt_interp"] in {p["canonical_path"] for p in pins.values()}},
        "mechanism": [
            "Build an isolated private directory containing exactly fifteen SONAME symlinks, each with an absolute literal target equal to the corresponding original canonical provider.",
            "At final target execve only, set LD_LIBRARY_PATH to that one directory. Keep all host tools and Python on the existing clean host environment.",
            "DT_RUNPATH does not supersede LD_LIBRARY_PATH; there is no DT_RPATH in the original providers or fresh executable. The one shim therefore supplies direct and transitive SONAME lookups before embedded compiler paths.",
            "Kernel PT_INTERP remains the original absolute glibc2.40 loader; the loader SONAME alias resolves to the same pinned file. Do not change the build2 executable or original provider bytes.",
            "Reject extra files or glibc-hwcaps subdirectories, non-symlinks, relative/indirect/wrong symlink targets, unresolved providers or altered hashes. Recheck before and after trace/native phases.",
            "Retain the source2 expected runtime as exact resolved canonical paths and hashes. The future trace and worker maps must match those fifteen providers plus the executable without new exclusions. A successful static preflight is not a substitute for these actual runtime checks.",
            "Keep failed run2 and source2 unchanged; new controller and attempt must use distinct additive namespaces and record a source3 binding/manifest for the shim policy.",
        ],
        "limitations": ["No shim has been created for the actual diagnostic and no actual trace/worker was run by this audit.",
                        "dlopen or other runtime additions remain subject to the strict actual worker-map set check.",
                        "No numerical, performance or QP qualification follows from this static resolution proposal."],
    }
    if successful or not cycles or not all(result["mechanism_checks"].values()):
        raise ValueError("Unexpected static resolution facts")
    path = here / "resolution-audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_sha256": sha(path), "permutations_tested": tested,
                      "successful_orders": len(successful), "contradictions": len(cycles),
                      "mechanism_checks": result["mechanism_checks"]}, indent=2))


if __name__ == "__main__":
    main()
