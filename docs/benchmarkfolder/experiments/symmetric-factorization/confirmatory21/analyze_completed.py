#!/usr/bin/env python3
"""Recheck all fixed confirmation samples after completion, without exclusions."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run_id):
    report = HERE / run_id
    monitor = HERE / (run_id + "-monitor")
    result = json.loads((report / "result.json").read_text())
    context = json.loads((monitor / "result.json").read_text())
    topology = json.loads((monitor / "topology.json").read_text())
    assert result["complete"] and context["returncode"] == 0
    assert not result["acceptance_eligible"]
    assert result["policy"]["required_repetitions"] == 21
    assert result["policy"]["retention_ratio_threshold"] == 1.0
    assert [x["repetition"] for x in result["comparisons"]] == list(range(21))
    for variant in ("original", "factor"):
        measurements = result["measurements"][variant]
        assert [x["repetition"] for x in measurements] == list(range(21))
        for measurement in measurements:
            rep = measurement["repetition"]
            values = report / f"{variant}-{rep}-values.dat"
            assert sha(values) == measurement["values_sha256"]
            parsed = []
            for line in values.read_text().splitlines():
                f = line.split()
                assert len(f) == 8
                parsed.append(dict(point=int(f[0]), order=int(f[1]), status=int(f[2]),
                    coefficients=list(map(float, f[3:7])), accuracy=float(f[7])))
            assert parsed == measurement["rows"]
            dims = measurement["answer_dimensions"]
            assert [(x["point"], x["order"]) for x in parsed] == [
                (point, order) for point in range(1, 101)
                for order in range(dims[1] + 1)]
            timed = {}
            for line in (report / f"{variant}-{rep}-timed_status.dat").read_text().splitlines():
                status, count = map(int, line.split())
                assert str(status) not in timed
                bench.validate_status(status, 6)
                timed[str(status)] = count
            assert timed == measurement["timed_status_counts"]
            assert sum(timed.values()) == 100
            seconds = list(map(float, (report / f"{variant}-{rep}-timing.dat")
                               .read_text().split()))
            assert len(seconds) == 2
            assert seconds[1] / 100 == measurement["seconds_per_point"]
    baseline = [x["seconds_per_point"] for x in result["measurements"]["original"]]
    candidate = [x["seconds_per_point"] for x in result["measurements"]["factor"]]
    statistics = bench.runtime_statistics(baseline, candidate, 1.0)
    assert statistics == result["paired_statistics"]
    comparisons = result["comparisons"]
    assert all(x["passed"] for x in comparisons)
    return dict(
        complete=True, acceptance_eligible=False, primary_campaign_pairs=21,
        result_sha256=sha(report / "result.json"),
        monitor_result_sha256=sha(monitor / "result.json"),
        point_stream_sha256=sha(report / "points.dat"),
        earlier_seven_result_sha256=result["prior_seven_result_sha256"],
        analysis_script_sha256=sha(Path(__file__)),
        physical_checks=dict(
            points_per_variant_per_repetition=100,
            compared_coefficients=sum(x["coefficient_count"] for x in comparisons),
            printed_binary64_identical_coefficients=sum(
                x["printed_binary64_identical_coefficients"] for x in comparisons),
            maximum_tolerance_fraction=max(
                x["maximum_tolerance_fraction"] for x in comparisons),
            accuracy_or_status_changes=[
                dict(repetition=x["repetition"], **change)
                for x in comparisons for change in x["accuracy_or_status_changes"]],
            added_qp_rescues=[x["added_qp_rescues"] for x in comparisons],
            timed_status_counts={
                variant: [x["timed_status_counts"] for x in result["measurements"][variant]]
                for variant in ("original", "factor")}),
        paired_statistics=statistics,
        host_context=dict(topology=topology, monitoring=context),
        interpretation="Independent fixed21-pair primary analysis, all samples "
            "retained. Earlier7 remains separate and inconclusive; no combined "
            "estimate or adaptive additional run is used. This is an isolated "
            "changed-association prototype, not a production or final-benchmark "
            "acceptance claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run_id)
    target = HERE / args.run_id / "analysis.json"
    if args.write:
        with target.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    else:
        assert json.loads(target.read_text()) == result
    print(json.dumps(dict(status=result["paired_statistics"]["status"],
        ratio=result["paired_statistics"]["runtime_ratio"],
        confidence_interval_95=result["paired_statistics"]["ratio_confidence_interval_95"],
        physical_checks=result["physical_checks"]), indent=2))
