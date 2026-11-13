#!/usr/bin/env python3
"""Append qualified initialization/profile evidence while preserving all45 pages.

Requires the actual closed pilot/profile packets and pure derivation.
Canonical replacement additionally requires the explicit publication flag; the
selection admits the concrete append proposal without performing publication.
No native physics is run by this document owner.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import types

SCHEMA = "madloop-tirust-initialization-profile-composite-report-v1"
BASE_PDF_SHA = "fa83602d57a43f25bc469e8d7185ba54b3236031eb861b12193a8788afd38678"
BASE_LEDGER_SHA = "7caeda73a75936563218c236499085d014ab01202436b3a0dabcc599e0766158"
BASE_OWNER_SHA = "990b0506ae96c00ab9ac76211334d49dcd968399c2ebd2a2145f789e4268c672"
SIGNED_TIRUST_REVISION = "10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1"
SIGNED_DSO_SHA = "85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0"
BASE_PAGES = 45
RENDER_INPUTS = ('initialization-and-profile.typ', 'initialization-profile-data.json', 'initialization-profile-selection.json', 'initialization-source-binding.json', 'derive_initialization_profile.py', 'prototype_derivation.py', 'comparison-criterion.py', 'comparison-criterion-provenance.json', 'render_phase_bars.py', 'phase-bars.svg', 'phase-timings.csv', 'stock-profile.csv', 'candidate-routes.csv', 'clean-initialization.csv', 'reference-component-misses.csv', 'first-point-timings.csv', 'first-point-routes.csv', 'first-point-checks.csv', 'stock-profile-current.csv', 'style.typ', 'style-provenance.json')
TYPST = "/nix/store/2szzba1xi60y8i3qcyxwvgq2i6w9iy5s-typst-0.15.1/bin/typst"
FONT_PATHS = ["/nix/store/fsknhsx8k14l62da4d07pilsnxsvkdc1-libertinus-7.051/share/fonts",
              "/nix/store/0lhf2h2fnjqvpzmk92s5axr2ybvs981s-dejavu-fonts-2.37/share/fonts"]


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def identity(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    def bad(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def inside(root, name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid relative input")
    part = PurePosixPath(name)
    require(not part.is_absolute() and ".." not in part.parts and str(part) == name, "Unsafe relative input")
    path = root
    for component in part.parts:
        path /= component
        require(not path.is_symlink(), "Symlink report input")
    return path


def relative_argument(root, value):
    path = Path(os.path.abspath(value))
    require(path.is_relative_to(root), "Report dependencies must be retained inside benchmark root")
    name = path.relative_to(root).as_posix()
    inside(root, name)
    return name


def load_archive(root, archive, index_sha):
    pinned = {"benchmark.pdf": BASE_PDF_SHA, "report-build.json": BASE_LEDGER_SHA,
              "build_zggg_timing_report.py": BASE_OWNER_SHA}
    require(identity(archive / "index.json")["sha256"] == index_sha, "Archive index changed")
    require({p.name for p in archive.iterdir()} == set(pinned) | {"index.json"}, "Archive inventory changed")
    index = read(archive / "index.json")
    expected = {}
    for name, sha in pinned.items():
        path = inside(archive, name)
        expected[name] = identity(path)
        require(expected[name]["sha256"] == sha, "Archived input differs: " + name)
    require(index["files"] == expected and index["page_count"] == BASE_PAGES, "Archive index disagrees")
    previous = types.ModuleType("pinned45_owner")
    previous.__file__ = str(archive / "build_zggg_timing_report.py")
    exec(compile((archive / "build_zggg_timing_report.py").read_bytes(), previous.__file__, "exec"), previous.__dict__)
    previous.verify(root, archive / "benchmark.pdf", archive / "report-build.json")
    prior = read(archive / "report-build.json")["arguments"]
    owner, _ = previous.load_archive(root, inside(root, prior["archive"]), prior["archive_index_sha256"])
    return owner, {"index": identity(archive / "index.json"), "files": expected,
                   "document_helper_owner": identity(owner.__file__)}


def render_record(report, pages):
    receipt = read(report / "render-receipt.json")
    require(receipt["schema"] == "madloop-tirust-initialization-profile-tail-render-v1" and receipt["closed"] is True,
            "Render receipt is not closed")
    expected = {name: identity(report / name) for name in RENDER_INPUTS}
    require(receipt["inputs"] == expected and receipt["PDF"] == identity(report / "initialization-profile-supplement.pdf")
            and receipt["page_count"] == pages and receipt["new_native_physics_evaluations"] == 0,
            "Render receipt inputs/output differ")
    commands = receipt["commands"]
    require(commands and all(c["returncode"] == 0 for c in commands), "Render commands failed or absent")
    versions = [c for c in commands if len(c["argv"]) == 2 and c["argv"][1] == "--version"
                and c["argv"][0] == TYPST]
    require(len(versions) == 1 and versions[0]["stdout"].startswith("typst 0.15.1 "), "Unqualified Typst")
    compiles = [c for c in commands if len(c["argv"]) >= 4 and c["argv"][1] == "compile"
                and c["argv"][0] == TYPST]
    require(len(compiles) == 1, "Expected one bound Typst compilation")
    argv = compiles[0]["argv"]
    require(Path(argv[2]).name == "initialization-and-profile.typ" and Path(argv[3]).name == "initialization-profile-supplement.pdf",
            "Render command names different source/output")
    require("--creation-timestamp" in argv and argv.index("--creation-timestamp") + 1 < len(argv)
            and argv[argv.index("--creation-timestamp") + 1] == "1794441600", "Render timestamp differs")
    require(receipt["tool_paths"]["typst"] == TYPST and receipt["font_paths"] == FONT_PATHS,
            "Unqualified render tool/font paths")
    return {"receipt": identity(report / "render-receipt.json"), "inputs": expected,
            "PDF": identity(report / "initialization-profile-supplement.pdf"), "page_count": pages}


def input_packet(folder, expected, index_name):
    require(index_name == "index.json", "Unexpected scientific packet index name")
    require(identity(folder / index_name)["sha256"] == expected, "Scientific packet changed")
    value = read(folder / index_name)
    keys = [key for key in ("files", "payloads") if key in value]
    require(len(keys) == 1, "Ambiguous scientific inventory")
    files = value[keys[0]]
    require({p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            == set(files) | {index_name}, "Scientific inventory changed")
    for name, item in files.items():
        require(identity(inside(folder, name)) == item, "Scientific payload changed: " + name)


def selected(root, arguments):
    archive = inside(root, arguments["archive"])
    report = inside(root, arguments["report"])
    owner, old = load_archive(root, archive, arguments["archive_index_sha256"])
    index = owner.packet(report, arguments["report_index_sha256"])
    pages = index["page_count"]
    require(pages == 2, "Expected the two-page final-source supplement")
    choice = read(report / "initialization-profile-selection.json")
    require(choice["schema"] == "madloop-initialization-profile-report-selection-v1"
            and choice["closed"] is True and choice["candidate_qualified"] is True
            and choice["production_artifact_qualified"] is True and choice["append_proposal_admitted"] is True and choice["canonical_publication_admitted"] is False
            and choice["production_source_promotion"] is False
            and choice["performance_acceptance"] is False and choice["new_native_physics_evaluations"] == 0,
            "A report cannot promote code or invent candidate qualification/performance")
    require(choice["packets"], "Missing exact scientific packet selectors")
    for item in choice["packets"]:
        require(set(item) == {"path", "index_name", "index_sha256"}, "Unexpected packet selector")
        input_packet(inside(root, item["path"]), item["index_sha256"], item["index_name"])
    module = types.ModuleType("pinned_initialization_profile_derivation")
    module.__file__ = str(report / "derive_initialization_profile.py")
    exec(compile((report / "derive_initialization_profile.py").read_bytes(), module.__file__, "exec"), module.__dict__)
    data, expected_choice = module.derive(root)
    require(read(report / "initialization-profile-data.json") == data and choice == expected_choice,
            "Rendered facts differ from exact contained evidence replay")
    for name, payload in module.csv_payloads(data).items():
        require((report / name).read_bytes() == payload, "Report CSV differs from replay: " + name)
    require(data["schema"] == "madloop-initialization-profile-report-data-v1"
            and data["closed"] is True and data["controls"]["o2_promoted"] is False
            and data["controls"]["o2_benefit_qualified"] is False
            and data["production_artifact_qualified"] is True and data["preview_only"] is False
            and data["performance_acceptance"] is False and data["new_native_physics_evaluations"] == 0,
            "Unexpected controls/report scope")
    candidate = data["candidate"]
    require(candidate["actual_source_bound"] is True and candidate["functional_qualified"] is True,
            "Candidate requires actual source and accuracy/functional qualification")
    require(candidate["measurement_role"] in ("functional-only", "performance-qualified", "descriptive-clean-initialization"), "Unknown candidate role")
    if candidate["measurement_role"] == "functional-only":
        require(candidate["timing_qualified"] is False and candidate["phases"] is None,
                "Functional-only elapsed times cannot become performance evidence")
    elif candidate["measurement_role"] == "descriptive-clean-initialization":
        require(candidate["descriptive_initialization_timing_admitted"] is True and candidate["phases"] is None,
                "Quiet one-pair evidence cannot acquire an instrumented phase breakdown")
    else:
        require(candidate["timing_qualified"] is True, "Missing candidate timing admission")
    rendered = render_record(report, pages)
    require(rendered["inputs"]["style.typ"]["sha256"] == choice["style_sha256"], "Style differs")
    return owner, {"archive": old, "report_index": identity(report / "index.json"),
                   "render": rendered, "scientific_packets": choice["packets"]}


def prove(owner, tools, archive, appendix, merged, scratch, pages):
    old = owner.inspect_document(tools, archive / "benchmark.pdf", scratch, "base45", BASE_PAGES)
    tail = owner.inspect_document(tools, appendix, scratch, "initialization-profile", pages)
    combined = owner.inspect_document(tools, merged, scratch, "combined", BASE_PAGES + pages)
    require(combined[:BASE_PAGES] == old and combined[BASE_PAGES:] == tail, "Page preservation failed")
    return {"base_pages": BASE_PAGES, "tail_pages": pages, "total_pages": BASE_PAGES + pages,
            "prefix_preserved": True, "tail_preserved": True, "pages": combined,
            "PDF": identity(merged), "dpi": 144, "tool_versions": tools.versions,
            "scope": "Paired RGB pixels, per-page text/geometry/fonts; not PDF object-graph or new annotation/link proof."}


def verify(root, output, ledger, verify_pages=False):
    record = read(ledger)
    require(record["schema"] == SCHEMA and record["builder"] == identity(__file__), "Wrong or changed owner")
    arguments = record["arguments"]
    require(set(arguments) == {"archive", "report", "archive_index_sha256", "report_index_sha256"}, "Unexpected arguments")
    owner, inputs = selected(root, arguments)
    require(inputs == record["selected_inputs"] and identity(output) == record["PDF"], "Composite input/output changed")
    proof = record["page_proof"]
    pages = inputs["render"]["page_count"]
    require(digest(proof) == record["page_proof_sha256"] and proof["PDF"] == record["PDF"], "Changed page proof")
    require(proof["prefix_preserved"] and proof["tail_preserved"] and proof["base_pages"] == BASE_PAGES
            and proof["tail_pages"] == pages and proof["total_pages"] == BASE_PAGES + pages
            and len(proof["pages"]) == BASE_PAGES + pages, "Incomplete page proof")
    require(record["production_source_promotion"] is False and record["performance_acceptance"] is False
            and record["new_native_physics_evaluations"] == 0, "Unexpected release claim")
    if verify_pages:
        with tempfile.TemporaryDirectory(prefix="initialization-profile-pages-") as name:
            fresh = prove(owner, owner.DocumentTools(), inside(root, arguments["archive"]),
                          inside(root, arguments["report"]) / "initialization-profile-supplement.pdf",
                          output, Path(name), pages)
            require(fresh["pages"] == proof["pages"], "Fresh page proof differs")
    return {"verified": True, "pages": BASE_PAGES + pages, "new_native_physics_evaluations": 0}


def output_state(output, ledger, canonical):
    require(not any(p.is_symlink() for p in (output, ledger, *output.parents)), "Symlink publication output")
    if canonical:
        require(output.exists() and ledger.exists(), "Missing original canonical PDF/ledger")
        before = {"PDF": identity(output), "ledger": identity(ledger)}
        require(before["PDF"]["sha256"] == BASE_PDF_SHA and before["ledger"]["sha256"] == BASE_LEDGER_SHA,
                "Canonical replacement requires the exact old45-page PDF and ledger")
        return before
    require(not output.exists() and not ledger.exists(), "Fresh output already exists")
    return {"PDF": None, "ledger": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-index-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-index-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-canonical", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-pages", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = Path(os.path.abspath(args.output or root / "benchmark.pdf"))
    canonical = output == root / "benchmark.pdf"
    ledger = root / "report-build.json" if canonical else output.with_suffix(".build.json")
    require(not args.verify_pages or args.verify_only, "Page verification requires --verify-only")
    if args.verify_only:
        require(not args.publish_canonical, "Verification never publishes")
        print(encoded(verify(root, output, ledger, args.verify_pages)).decode(), end="")
        return
    require(canonical == args.publish_canonical, "Canonical replacement requires explicit --publish-canonical")
    require(all(value is not None for value in (args.archive, args.archive_index_sha256,
            args.report, args.report_index_sha256)), "Missing explicit archive/report pins")
    arguments = {"archive": relative_argument(root, args.archive), "archive_index_sha256": args.archive_index_sha256,
                 "report": relative_argument(root, args.report), "report_index_sha256": args.report_index_sha256}
    owner, inputs = selected(root, arguments)
    output.parent.mkdir(parents=True, exist_ok=True)
    lockdir = root.parents[1] / ".tirust-build"
    lockdir.mkdir(parents=True, exist_ok=True)
    with (lockdir / "stress-report-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = output_state(output, ledger, canonical)
        with tempfile.TemporaryDirectory(prefix=".initialization-profile-report-", dir=output.parent) as name:
            scratch, tools = Path(name), owner.DocumentTools()
            archive = inside(root, arguments["archive"])
            appendix = inside(root, arguments["report"]) / "initialization-profile-supplement.pdf"
            merged = scratch / "combined.pdf"
            tools.run([tools.paths["pdfunite"], str(archive / "benchmark.pdf"), str(appendix), str(merged)])
            proof = prove(owner, tools, archive, appendix, merged, scratch, inputs["render"]["page_count"])
            require(selected(root, arguments)[1] == inputs, "Input changed during append")
            require(output_state(output, ledger, canonical) == before, "Output changed during append")
            record = {"schema": SCHEMA, "arguments": arguments, "selected_inputs": inputs,
                      "builder": identity(__file__), "PDF": identity(merged), "page_proof": proof,
                      "page_proof_sha256": digest(proof), "commands": tools.commands,
                      "performance_acceptance": False, "production_source_promotion": False,
                      "new_native_physics_evaluations": 0,
                      "publication": "Individually atomic renames under exclusive lock, ledger last with exception rollback; not crash-atomic across two paths."}
            owner.publish_pair(merged, encoded(record), output, ledger)
    print(encoded({"PDF": str(output), "pages": proof["total_pages"], "sha256": record["PDF"]["sha256"]}).decode(), end="")


if __name__ == "__main__":
    main()
