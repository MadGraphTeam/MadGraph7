#!/usr/bin/env python3
"""Append the analytic Fortran repair experiment while preserving all42 old pages."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import types

SCHEMA = "madloop-tirust-analytic-repair-composite-report-v1"
BASE_PDF_SHA = "4f087a337a8cf736d8d21509486727f7c6f55b3467dd1535d4190f96be73a18f"
BASE_LEDGER_SHA = "9a070d34713d90fe474e8887bdea0c2c461a60d4eb452904f1c9682a9a2806b1"
BASE_OWNER_SHA = "0ed3e3465facd71de8e1f8a63109fcdd73b37229dd9e2ac5b1247ad73274b594"
PRODUCTION_REVISION = "0fcd2cd873d61d134837e1d9e87afb13a47824fd"
BASE_PAGES = 42
RENDER_INPUTS = ("analytic-repair.typ", "analytic-repair-data.json", "analytic-repair-selection.json",
                 "derive_timing.py", "style.typ", "style-provenance.json")
CAPSULE = {"path": "experiments/fortran-analytic-repair-qualification-1", "index_name": "index.json",
           "index_sha256": "f8bc6ce06201665abc0da7d79bcb55697a215d99f2258e9ba623453ebc33e6e9"}
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
              "build_qualification_report.py": BASE_OWNER_SHA}
    require(identity(archive / "index.json")["sha256"] == index_sha, "Archive index changed")
    require({p.name for p in archive.iterdir()} == set(pinned) | {"index.json"}, "Archive inventory changed")
    index = read(archive / "index.json")
    expected = {}
    for name, sha in pinned.items():
        path = inside(archive, name)
        expected[name] = identity(path)
        require(expected[name]["sha256"] == sha, "Archived input differs: " + name)
    require(index["files"] == expected and index["page_count"] == BASE_PAGES, "Archive index disagrees")
    previous = types.ModuleType("pinned42_owner")
    previous.__file__ = str(archive / "build_qualification_report.py")
    exec(compile((archive / "build_qualification_report.py").read_bytes(), previous.__file__, "exec"), previous.__dict__)
    previous.verify(root, archive / "benchmark.pdf", archive / "report-build.json")
    prior = read(archive / "report-build.json")["arguments"]
    owner, _ = previous.load_archive(root, inside(root, prior["archive"]), prior["archive_index_sha256"])
    return owner, {"index": identity(archive / "index.json"), "files": expected,
                   "document_helper_owner": identity(owner.__file__)}


def render_record(report, pages):
    receipt = read(report / "render-receipt.json")
    require(receipt["schema"] == "madloop-tirust-analytic-repair-tail-render-v1" and receipt["closed"] is True,
            "Render receipt is not closed")
    expected = {name: identity(report / name) for name in RENDER_INPUTS}
    require(receipt["inputs"] == expected and receipt["PDF"] == identity(report / "analytic-repair-supplement.pdf")
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
    require(Path(argv[2]).name == "analytic-repair.typ" and Path(argv[3]).name == "analytic-repair-supplement.pdf",
            "Render command names different source/output")
    require("--creation-timestamp" in argv and argv.index("--creation-timestamp") + 1 < len(argv)
            and argv[argv.index("--creation-timestamp") + 1] == "1794441600", "Render timestamp differs")
    require(receipt["tool_paths"]["typst"] == TYPST and receipt["font_paths"] == FONT_PATHS,
            "Unqualified render tool/font paths")
    return {"receipt": identity(report / "render-receipt.json"), "inputs": expected,
            "PDF": identity(report / "analytic-repair-supplement.pdf"), "page_count": pages}


def selected(root, arguments):
    archive = inside(root, arguments["archive"])
    report = inside(root, arguments["report"])
    owner, old = load_archive(root, archive, arguments["archive_index_sha256"])
    index = owner.packet(report, arguments["report_index_sha256"])
    pages = index["page_count"]
    require(pages == 1, "Expected exactly one analytic repair page")
    choice = read(report / "analytic-repair-selection.json")
    require(choice["schema"] == "madloop-tirust-analytic-repair-report-selection-v1", "Wrong analytic repair selection")
    require(choice["production_revision"] == PRODUCTION_REVISION and choice["production_source_promotion"] is False
            and choice["source_promoted"] is False and choice["performance_acceptance"] is False
            and choice["new_native_physics_evaluations"] == 0 and len(choice["packets"]) == 1,
            "This appendix cannot promote code, physics qualification or performance")
    require(all(choice["packets"][0].get(k) == v for k, v in CAPSULE.items()), "Wrong analytic experiment capsule")
    for item in choice["packets"]:
        owner.packet(inside(root, item["path"]), item["index_sha256"], item["index_name"])
    rendered = render_record(report, pages)
    require(rendered["inputs"]["style.typ"]["sha256"] == choice["style_sha256"], "Style differs")
    return owner, {"archive": old, "report_index": identity(report / "index.json"),
                   "render": rendered, "scientific_packets": choice["packets"]}


def prove(owner, tools, archive, appendix, merged, scratch, pages):
    old = owner.inspect_document(tools, archive / "benchmark.pdf", scratch, "base42", BASE_PAGES)
    tail = owner.inspect_document(tools, appendix, scratch, "analytic-repair", pages)
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
        with tempfile.TemporaryDirectory(prefix="analytic-repair-pages-") as name:
            fresh = prove(owner, owner.DocumentTools(), inside(root, arguments["archive"]),
                          inside(root, arguments["report"]) / "analytic-repair-supplement.pdf",
                          output, Path(name), pages)
            require(fresh["pages"] == proof["pages"], "Fresh page proof differs")
    return {"verified": True, "pages": BASE_PAGES + pages, "new_native_physics_evaluations": 0}


def output_state(output, ledger, canonical):
    require(not any(p.is_symlink() for p in (output, ledger, *output.parents)), "Symlink publication output")
    if canonical:
        require(output.exists() and ledger.exists(), "Missing original canonical PDF/ledger")
        before = {"PDF": identity(output), "ledger": identity(ledger)}
        require(before["PDF"]["sha256"] == BASE_PDF_SHA and before["ledger"]["sha256"] == BASE_LEDGER_SHA,
                "Canonical replacement requires the exact old42-page PDF and ledger")
        return before
    require(not output.exists() and not ledger.exists(), "Fresh output already exists")
    return {"PDF": None, "ledger": None}


def guard_ew_successor(root, output, ledger):
    """The later EW owner owns its canonical composite; old archives stay valid."""
    if output == root / "benchmark.pdf" and ledger.exists() and read(ledger).get("schema") == "madloop-tirust-ew-qualification-composite-report-v1":
        raise ValueError("Canonical report contains the EW appendix. Use build_ew_qualification_report.py to build or verify it; --output CUSTOM.pdf remains available for this historical append owner.")


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
    guard_ew_successor(root, output, ledger)
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
        with tempfile.TemporaryDirectory(prefix=".analytic-repair-report-", dir=output.parent) as name:
            scratch, tools = Path(name), owner.DocumentTools()
            archive = inside(root, arguments["archive"])
            appendix = inside(root, arguments["report"]) / "analytic-repair-supplement.pdf"
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
