#!/usr/bin/env python3
"""Current-source report pages for a campaign, appended to the benchmark PDF.

  collect --campaign C [--dd-gate DIR]   gather compact summaries into <campaign>/summaries/
                                         and write <campaign>/report/report-data.json
  render  --campaign C [--publish]       typeset <campaign>/report/current-source.typ and append it
                                         to the base PDF (docs/benchmarkfolder/history/...); with
                                         --publish the result replaces docs/benchmarkfolder/benchmark.pdf
  verify  --campaign C                   re-render and check every recorded hash

Only committed summaries feed the pages; raw lane evidence stays on the producing
host. The base PDF is kept byte for byte as the first pages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs/benchmarkfolder"
sys.path.insert(0, str(ROOT / "tools"))
import tirust_numerical_summary as numerical  # noqa: E402

BASE_NAME = "history/report-before-current-source-1/benchmark.pdf"
CREATION_TIMESTAMP = "1794528000"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def campaign_dir(campaign):
    return BENCH / "campaigns" / campaign


def collect_dd_gate(source, target):
    cases = {}
    for path in sorted(Path(source).glob("*/dd-gate.json")):
        gate = json.loads(path.read_text())
        capture = json.loads((path.parent / "capture.json").read_text())
        h26 = json.loads((path.parent / "h26.json").read_text())
        refs = gate["references"]
        exercised = gate["dd_calls"] > 0
        cases[path.parent.name] = {
            "dp_calls": gate["production_replay"]["calls"],
            "production_bit_identical": gate["production_replay"]["non_identical"] == 0
            and gate["production_replay"]["status_mismatch"] == 0,
            "segments": h26["segments"], "routes_primary": h26["routes_primary"], "routes_check": h26["routes_check"],
            "dd_calls": gate["dd_calls"],
            "window_calls": refs["h40"]["dd_calls_compared"],
            "window_max": refs["h40"]["error_quantiles"].get("max"),
            "all_dd_max": refs["h105"]["error_quantiles"].get("max"),
            "above_flag": max(refs["h40"]["above_flag"], refs["h105"]["above_flag"]),
            "above_fail": max(refs["h40"]["above_fail"], refs["h105"]["above_fail"]),
            "verdict": ("qualified" if gate["qualified"] else "NOT qualified") if exercised else "not exercised",
            "trace_sha256": capture["trace_sha256"], "flagged": refs["h105"]["flagged"][:10],
            "failing": refs["h105"]["failing"][:10]}
    summary = {"metric": "max |difference| / max |value| of the same tensor degree over all Laurent slots",
               "flag": 1e-12, "fail": 1e-9, "cases": cases,
               "dd_calls_total": sum(c["dd_calls"] for c in cases.values()),
               "window_calls_total": sum(c["window_calls"] for c in cases.values()),
               "max_error": max([c["all_dd_max"] or 0 for c in cases.values()] + [0]),
               "qualified": all(c["verdict"] != "NOT qualified" for c in cases.values())}
    dump(target / "dd-gate-summary.json", summary)
    lines = ["| Case | DP calls | DD calls | new-window calls | max error (window) | max error (all DD) | verdict |",
             "|---|---|---|---|---|---|---|"]
    for name, c in cases.items():
        fmt = lambda x: "—" if x is None else f"{x:.1e}"
        lines.append(f"| {name} | {c['dp_calls']} | {c['dd_calls']} | {c['window_calls']} | {fmt(c['window_max'])} | "
                     f"{fmt(c['all_dd_max'])} | {c['verdict']} |")
    (target / "dd-gate-summary.md").write_text("\n".join(lines) + "\n")
    return summary


LABELS = {"qcd_n1_uux_z": "QCD u ū → Z", "qcd_n2_gg_ttx": "QCD g g → t t̄", "qcd_n3_uux_zgg": "QCD u ū → Z g g",
          "qcd_n4_uux_zzgg": "QCD u ū → Z Z g g", "ew_n1_udx_wp": "EW u d̄ → W⁺", "ew_n2_epem_mupmum": "EW e⁺e⁻ → μ⁺μ⁻",
          "ew_n3_epem_ttxa": "EW e⁺e⁻ → t t̄ γ", "ew_n4_epem_mupmumaa": "EW e⁺e⁻ → μ⁺μ⁻ γ γ",
          "mixed_uux_ga": "QCD+QED u ū → g γ", "li_gg_h": "loop-induced g g → H", "li_gg_hh": "loop-induced g g → H H",
          "cms_ew_n2_epem_mupmum": "CMS EW e⁺e⁻ → μ⁺μ⁻"}


def sci(x):
    if x is None:
        return "—"
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def display(data):
    out = {}
    cases, previous = data["numerical"]["cases"], data["numerical"]["predecessor"]
    rows, passed, total, target = [], 0, 0, 0
    for case, entry in cases.items():
        lanes = entry["lanes"]
        done = [l for k, l in lanes.items() if l["status"] == "complete" and not l.get("reference")]
        total += 6
        passed += sum(1 for l in lanes.values() if l["status"] == "complete" and (l.get("reference") or l.get("passed")))
        target += sum(1 for l in lanes.values() if l["status"] == "complete" and (l.get("reference") or l.get("target_misses") == 0))
        misses = "/".join(str(lanes.get(f"candidate-{b}", {}).get("target_misses", "–")) for b in (1, 6, 7, 8, 9))
        nine, old = lanes.get("candidate-9", {}), previous.get(case, {}).get("lanes", {}).get("candidate-9", {})
        rows.append([LABELS[case], f"{sum(1 for l in lanes.values() if l['status'] == 'complete' and (l.get('reference') or l.get('passed')))}/{len(lanes)}",
                     misses, sci(nine.get("max_symmetric_finite")),
                     f"{nine.get('dp', '–')}/{nine.get('qp', '–')}",
                     f"{old.get('dp', '–')}/{old.get('qp', '–')}" if old.get("status") == "complete" else "–"])
    out["numerical_rows"] = rows
    out["numerical_totals"] = {"lanes": total, "passed": passed, "target": target}
    if "dd_gate" in data:
        out["dd_rows"] = [[LABELS[c], str(v["dp_calls"]), str(v["dd_calls"]), str(v["window_calls"]), sci(v["window_max"]),
                           sci(v["all_dd_max"]), v["verdict"]] for c, v in data["dd_gate"]["cases"].items()]
        out["dd_totals"] = {"dd": data["dd_gate"]["dd_calls_total"], "window": data["dd_gate"]["window_calls_total"],
                            "max": sci(data["dd_gate"]["max_error"]), "qualified": data["dd_gate"]["qualified"]}
    if "provenance" in data:
        yes = {True: "yes", False: "NO", None: "–"}
        out["provenance_rows"] = [[LABELS[c], *(str(v["counts"][k]) for k in ("dp_native", "dp_dd", "dp_mpfr", "qp", "failed")),
                                   f"{v.get('numerical_lane_status_agreement', '–')}/{v['points']}",
                                   yes[v.get("cache-off_identical")], yes[v.get("small-budget_identical")]]
                                  for c, v in data["provenance"]["cases"].items() if v["status"] == "complete"]
    if "stress" in data:
        names = {"1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS"}
        groups = {}
        for e in data["stress"]["points"]:
            lanes = e["lanes"]
            tirust, chain, qp = lanes.get("candidate-9-auto", {}), lanes.get("chain-9671-auto", {}), lanes.get("candidate-9-qp", {})
            reference = e["reference"]
            if reference.startswith("checked: "):
                reference = "checked " + " = ".join(names[x] for x in reference[len("checked: "):].split(" = "))
            qp_text = ("agrees" if qp.get("vs_reference", {}).get("passed") else "differs" if "vs_reference" in qp else "–")
            codes = lambda lane: ",".join(map(str, lane.get("codes", []))) or "–"
            row = (tirust.get("classification", "pending"), codes(tirust),
                   chain.get("classification", "pending"), codes(chain), qp_text, reference)
            # Ordinary anchor points of a case collapse into one counted row.
            key = (e["case"], e.get("kind") or "", e["id"] if e.get("kind") != "ordinary" else "ordinary")
            groups.setdefault(key, []).append(row)
        rows = []
        for (case, kind, ident), members in groups.items():
            count = len(members)
            def summary(index_label, index_codes=None):
                values = {}
                for m in members:
                    label = m[index_label] + (f" ({m[index_codes]})" if index_codes is not None else "")
                    values[label] = values.get(label, 0) + 1
                return "; ".join(k if count == 1 else f"{k} ×{v}" for k, v in values.items())
            point = kind if count == 1 else f"{count} {kind}"
            rows.append([LABELS[case], point, summary(5), summary(0, 1), summary(2, 3), summary(4)])
        out["stress_rows"] = rows
        out["stress_blocking"] = data["stress"]["blocking_wrong_accepted"]
    if "profile" in data:
        prof = data["profile"]
        ninjars = next(r["full_ms_median"] for r in prof["rows"] if r["backend"] == 8)
        out["profile_rows"] = [[r["name"], f"{r['full_ms_median']:.1f}", " / ".join(f"{x:.1f}" for x in r["full_ms"]),
                                f"{r['initialization_s_median']:.1f}", f"{r['full_ms_median'] / ninjars:.3f}"]
                               for r in prof["rows"]]
        out["profile_ratio"] = f"{prof['tirust_over_ninjars_median']:.3f}"
        out["profile_dso"] = prof["tirust_dso_sha256"][:12]
    if "stability" in data:
        out["stability_rows"] = [[r["variant"], LABELS[r["case"]], r["reducer"], f"{r['available']}/{r['points']}",
                                  *(("{:.2f}".format(lo) if lo == hi else "[{:.2f}, {:.2f}]".format(lo, hi))
                                    for lo, hi in (r["tail"][k] for k in ("0.001", "1e-06", "1e-09")))]
                                 for r in data["stability"]["table"]]
    return out


def collect(campaign, dd_gate=None, predecessor="entry-latch-cache-guard-1"):
    base = campaign_dir(campaign)
    target = base / "summaries"
    target.mkdir(parents=True, exist_ok=True)
    data = {"campaign": campaign, "manifest_sha256": sha256(base / "manifest.json")}
    manifest = json.loads((base / "manifest.json").read_text())
    native = manifest["candidate"]["native"]
    data["identity"] = {"exporter_revision": manifest["candidate"]["exporter_identity"]["revision"],
                        "exporter_files_sha256": manifest["candidate"]["exporter_identity"]["files_sha256"],
                        "tirust_revision": json.loads((Path(native["prefix"]) / "lib/tirust-config.json").read_text())
                        ["local_sources"]["tirust"]["revision"] if Path(native["prefix"]).is_dir() else None,
                        "native_files": native["files"]}
    summary = numerical.collect(campaign)
    previous = numerical.collect(predecessor)
    dump(target / "numerical-summary.json", {"campaign": campaign, "predecessor": predecessor, "cases": summary})
    (target / "numerical-summary.csv").write_text(numerical.rows(summary))
    (target / "numerical-summary.md").write_text(numerical.table(summary, previous))
    data["numerical"] = {"cases": summary, "predecessor": previous}
    if dd_gate:
        data["dd_gate"] = collect_dd_gate(dd_gate, target)
    for name, source in (("stress", base / "experiments/stress/stress-summary.json"),
                         ("provenance", base / "provenance/provenance-summary.json")):
        if source.is_file():
            data[name] = json.loads(source.read_text())
            shutil.copy2(source, target / source.name)
            if source.with_suffix(".md").is_file():
                shutil.copy2(source.with_suffix(".md"), target / source.with_suffix(".md").name)
    profile = target / "profile-zggg-o3.json"
    if profile.is_file():
        data["profile"] = json.loads(profile.read_text())
    table = base / "stability/stability-table.json"
    if table.is_file():
        data["stability"] = {"table": json.loads(table.read_text()),
                             "figures": [p.name for p in sorted((base / "stability").glob("stability-*.pdf"))]}
    data["display"] = display(data)
    dump(base / "report/report-data.json", data)
    return data


def render(campaign, output, ledger=None, supplement=None):
    base = campaign_dir(campaign)
    report = base / "report"
    canonical_supplement = report / "current-source-supplement.pdf"
    supplement = canonical_supplement if supplement is None else Path(supplement)
    typst = shutil.which("typst")
    commands = []
    def run(argv):
        completed = subprocess.run(argv, text=True, capture_output=True)
        commands.append({"argv": argv, "returncode": completed.returncode, "stdout": completed.stdout.strip()})
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return completed.stdout
    version = run([typst, "--version"]).strip()
    run([typst, "compile", "--root", str(BENCH), "--creation-timestamp", CREATION_TIMESTAMP,
         str(report / "current-source.typ"), str(supplement)])
    pages = int(run(["pdfinfo", str(supplement)]).split("Pages:")[1].split()[0])
    base_pdf = BENCH / BASE_NAME
    base_pages = int(run(["pdfinfo", str(base_pdf)]).split("Pages:")[1].split()[0])
    run(["pdfunite", str(base_pdf), str(supplement), str(output)])
    inputs = {p.relative_to(BENCH).as_posix(): sha256(p) for p in sorted(report.iterdir())
              if p.is_file() and p.name not in (canonical_supplement.name, "report-build.json", "benchmark-preview.pdf")}
    inputs.update({p.relative_to(BENCH).as_posix(): sha256(p) for p in sorted((base / "stability").glob("stability-*.pdf"))})
    inputs["style.typ"] = sha256(BENCH / "style.typ")
    record = {"schema": "madloop-tirust-current-source-report-v1", "campaign": campaign, "typst": version,
              "base": {"path": BASE_NAME, "sha256": sha256(base_pdf), "pages": base_pages},
              "supplement": {"path": canonical_supplement.relative_to(BENCH).as_posix(), "sha256": sha256(supplement),
                             "pages": pages},
              "PDF": {"sha256": sha256(output), "pages": base_pages + pages}, "inputs": inputs,
              "builder_sha256": sha256(__file__), "commands": commands}
    if ledger:
        dump(ledger, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "render", "verify"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--dd-gate", type=Path, help="directory with per-case dd-gate.json (collect)")
    parser.add_argument("--publish", action="store_true", help="replace docs/benchmarkfolder/benchmark.pdf")
    args = parser.parse_args(argv)
    ledger = campaign_dir(args.campaign) / "report/report-build.json"
    if args.command == "collect":
        collect(args.campaign, args.dd_gate)
    elif args.command == "render":
        output = BENCH / "benchmark.pdf" if args.publish else campaign_dir(args.campaign) / "report/benchmark-preview.pdf"
        record = render(args.campaign, output, ledger if args.publish else None)
        print(json.dumps({"PDF": str(output), "pages": record["PDF"]["pages"], "sha256": record["PDF"]["sha256"]}))
    else:
        recorded = json.loads(ledger.read_text())
        with tempfile.TemporaryDirectory() as scratch:
            fresh = render(args.campaign, Path(scratch) / "benchmark.pdf",
                           supplement=Path(scratch) / "supplement.pdf")
        checks = {"supplement": fresh["supplement"]["sha256"] == recorded["supplement"]["sha256"],
                  "inputs": fresh["inputs"] == recorded["inputs"],
                  "base": fresh["base"] == recorded["base"],
                  "canonical": sha256(BENCH / "benchmark.pdf") == recorded["PDF"]["sha256"]}
        print(json.dumps(checks))
        return 0 if all(checks.values()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
