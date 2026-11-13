#!/usr/bin/env python3
"""Append one actual stock-O3 profile page while preserving all47 old pages.

Authenticates the old47 PDF/ledger/owner transitively and replays only the new
five-run dataset; never calls the predecessor scientific chain.
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

SCHEMA = "madloop-stock-o3-profile-composite-report-v1"
BASE_PDF_SHA = "a6cdf944f8ec8d936574371a225d18365cbdc6eddd5acf94c340dbc071a7aa46"
BASE_LEDGER_SHA = "8c20f265ea092bf4a1cc36673817a86ea41b10cf79c45cdb651428c552a12b03"
BASE_OWNER_SHA = "df7884bfa512c2e74924c0eed79e5ac53d09e393d512e25a78e6515388525ec8"
SIGNED_TIRUST_REVISION = "10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1"
SIGNED_DSO_SHA = "85ee0cdf860e5ad4c7a5c26163802673de22404690a10c7c53fb73ebe1ff5bb0"
BASE_PAGES = 47
RENDER_INPUTS = ('stock-o3-profile.typ', 'stock-o3-data.json', 'stock-o3-selection.json', 'stock-o3-binding.json', 'derive_stock_o3_profile.py', 'stock-o3-timings.csv', 'style.typ', 'style-provenance.json')
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
    """Authenticate old47 transitively; never call its scientific owner."""
    pinned = {"benchmark.pdf": BASE_PDF_SHA, "report-build.json": BASE_LEDGER_SHA,
              "build_initialization_profile_report.py": BASE_OWNER_SHA}
    require(identity(archive / "index.json")["sha256"] == index_sha, "Archive index changed")
    require({p.name for p in archive.iterdir()} == set(pinned) | {"index.json"}, "Archive inventory changed")
    index = read(archive / "index.json")
    expected = {name: identity(inside(archive, name)) for name in pinned}
    require(all(expected[name]["sha256"] == sha for name, sha in pinned.items()), "Archived47 changed")
    require(index["files"] == expected and index["page_count"] == BASE_PAGES, "Archive binding differs")
    ledger = read(archive / "report-build.json")
    proof = ledger["page_proof"]
    require(ledger["PDF"] == expected["benchmark.pdf"] and ledger["builder"] == expected["build_initialization_profile_report.py"], "Old47 PDF/owner linkage differs")
    require(digest(proof) == ledger["page_proof_sha256"] == "c74d69e19b2ea76b8e6acb223f3b2e8e2dbcb6455b4af6698115ef908f57683b", "Old47 proof seal differs")
    require(proof["total_pages"] == 47 and len(proof["pages"]) == 47 and proof["prefix_preserved"] and proof["tail_preserved"], "Old47 preservation proof incomplete")
    helper_name = "history/report-before-five-backend-timing-1/build_stress_report.py"
    helper_path = inside(root, helper_name)
    helper_identity = identity(helper_path)
    require(helper_identity["sha256"] == "f9df20e66a8457eb492941da67426ad2af1174d0ed4593065ff119697219803b" and index["document_helper"] == {"path": helper_name, **helper_identity}, "Document helper changed")
    helper = types.ModuleType("pinned_document_helper")
    helper.__file__ = str(helper_path)
    exec(compile(helper_path.read_bytes(), str(helper_path), "exec"), helper.__dict__)
    return helper, {"index": identity(archive / "index.json"), "files": expected,
                    "document_helper": {"path": helper_name, **helper_identity},
                    "predecessor_scientific_chain_replayed": False,
                    "old47_page_proof_sha256": ledger["page_proof_sha256"]}


def render_record(report, pages):
    receipt = read(report / "render-receipt.json")
    require(receipt["schema"] == "madloop-tirust-stock-o3-profile-tail-render-v1" and receipt["closed"] is True,
            "Render receipt is not closed")
    expected = {name: identity(report / name) for name in RENDER_INPUTS}
    require(receipt["inputs"] == expected and receipt["PDF"] == identity(report / "stock-o3-profile-supplement.pdf")
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
    require(Path(argv[2]).name == "stock-o3-profile.typ" and Path(argv[3]).name == "stock-o3-profile-supplement.pdf",
            "Render command names different source/output")
    require("--creation-timestamp" in argv and argv.index("--creation-timestamp") + 1 < len(argv)
            and argv[argv.index("--creation-timestamp") + 1] == "1794528000", "Render timestamp differs")
    require(receipt["tool_paths"]["typst"] == TYPST and receipt["font_paths"] == FONT_PATHS,
            "Unqualified render tool/font paths")
    return {"receipt": identity(report / "render-receipt.json"), "inputs": expected,
            "PDF": identity(report / "stock-o3-profile-supplement.pdf"), "page_count": pages}


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
        require((identity(inside(folder, name))["sha256"] == item) if isinstance(item, str) else (identity(inside(folder, name)) == item), "Scientific payload changed: " + name)


def selected(root, arguments):
    archive = inside(root, arguments["archive"])
    report = inside(root, arguments["report"])
    helper, old = load_archive(root, archive, arguments["archive_index_sha256"])
    input_packet(report, arguments["report_index_sha256"], "index.json")
    require(read(report / "index.json")["page_count"] == 1, "Expected one stock-O3 page")
    module = types.ModuleType("closed_stock_o3_derivation")
    module.__file__ = str(report / "derive_stock_o3_profile.py")
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    data, selection = module.derive(root)
    require(data == read(report / "stock-o3-data.json") and selection == read(report / "stock-o3-selection.json"), "New actual profile replay differs")
    require(data["schema"] == "madloop-stock-o3-zggg-report-data-v1" and data["closed"] is True
            and data["source_mg_revision"] == "cda704e28d634e893fabc3ac358d9a27789f93aa"
            and data["all_object_o3_proven"] is True and data["performance_acceptance"] is False,
            "Wrong source/O3/performance scope")
    require([r["backend"] for r in data["rows"]] == [1, 6, 7, 8, 9], "All five reducers must remain visible")
    require(selection["append_proposal_admitted"] is True and selection["new_native_physics_evaluations"] == 0,
            "Append proposal is not based on closed actual evidence")
    for packet in selection["packets"]:
        input_packet(inside(root, packet["path"]), packet["index_sha256"], "index.json")
    for name, payload in module.csv_payloads(data).items():
        require((report / name).read_bytes() == payload, "New profile CSV differs: " + name)
    rendered = render_record(report, 1)
    require(rendered["inputs"]["style.typ"]["sha256"] == "68e21877f61f92de75525e8d088b7de3fc00d8223b3aef393babc6015b59255e", "Existing report style changed")
    return helper, {"archive": old, "report_index": identity(report / "index.json"),
                    "render": rendered, "scientific_packets": selection["packets"]}


def prove(owner, tools, archive, appendix, merged, scratch, pages):
    old = owner.inspect_document(tools, archive / "benchmark.pdf", scratch, "base47", BASE_PAGES)
    tail = owner.inspect_document(tools, appendix, scratch, "stock-o3-profile", pages)
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
        with tempfile.TemporaryDirectory(prefix="stock-o3-profile-pages-") as name:
            fresh = prove(owner, owner.DocumentTools(), inside(root, arguments["archive"]),
                          inside(root, arguments["report"]) / "stock-o3-profile-supplement.pdf",
                          output, Path(name), pages)
            require(fresh["pages"] == proof["pages"], "Fresh page proof differs")
    return {"verified": True, "pages": BASE_PAGES + pages, "new_native_physics_evaluations": 0}


def output_state(output, ledger, canonical):
    require(not any(p.is_symlink() for p in (output, ledger, *output.parents)), "Symlink publication output")
    if canonical:
        require(output.exists() and ledger.exists(), "Missing original canonical PDF/ledger")
        before = {"PDF": identity(output), "ledger": identity(ledger)}
        require(before["PDF"]["sha256"] == BASE_PDF_SHA and before["ledger"]["sha256"] == BASE_LEDGER_SHA,
                "Canonical replacement requires the exact old47-page PDF and ledger")
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
        with tempfile.TemporaryDirectory(prefix=".stock-o3-profile-report-", dir=output.parent) as name:
            scratch, tools = Path(name), owner.DocumentTools()
            archive = inside(root, arguments["archive"])
            appendix = inside(root, arguments["report"]) / "stock-o3-profile-supplement.pdf"
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
