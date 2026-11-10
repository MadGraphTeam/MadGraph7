"""Optional implementation notes derived only from explicitly selected packages."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

ROLES = {"adaptive_debug", "adaptive_release", "born_backend89", "symmetry_components",
         "symmetry_ew100", "cms_diagnostic", "common_input_diagnostic", "mass_diagnostic",
         "installer_promotion", "symmetry_promotion", "common_input_finite_audit"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid implementation path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe implementation path")
    return name


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Package:
    def __init__(self, evidence, pin):
        require(isinstance(pin, dict) and set(pin) == {"index", "sha256"},
                "Implementation proof requires an exact index pin")
        self.evidence = evidence
        self.name = safe(pin["index"])
        self.checksum = pin["sha256"]
        require(digest(self.checksum), "Implementation proof requires a SHA-256 pin")
        self.index = evidence.json(self.name, self.checksum)
        self.prefix = str(PurePosixPath(self.name).parent)
        self.files = self.index.get("files")
        require(isinstance(self.files, dict) and self.files, "Empty implementation proof")
        for name, identity in self.files.items():
            expected = identity if isinstance(identity, str) else identity["sha256"]
            require(digest(expected), "Implementation payload requires a SHA-256 identity")
            raw = evidence.add(self.prefix + "/" + safe(name), expected)
            if isinstance(identity, dict) and "bytes" in identity:
                require(type(identity["bytes"]) is int and len(raw) == identity["bytes"],
                        "Implementation payload byte count differs")

    def raw(self, name):
        require(safe(name) in self.files, "Unindexed implementation claim input: " + name)
        identity = self.files[name]
        expected = identity if isinstance(identity, str) else identity["sha256"]
        return self.evidence.add(self.prefix + "/" + name, expected)

    def json(self, name):
        raw = self.raw(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        def invalid(value):
            raise ValueError("Nonfinite implementation claim: " + value)
        return json.loads(raw, parse_constant=invalid)


def symmetry_components(package):
    require(package.index.get("schema") == "ew-invariant-fixture-component-package-v2",
            "Unknown symmetry component package")
    value = package.json("summary.json")
    require(value.get("schema") == "ew-invariant-fixture-retained-reparse-v2"
            and value["fixture_status"] == "complete" and value["fixture_passed"] is True
            and value["semantic_reparse_matches_execution"] is True
            and value["physical_amplitude"] is False and value["performance_eligible"] is False
            and value["full_adapter_qualified"] is False and value["reducers"] is False,
            "Symmetry component scope differs")
    bulk = value["bulk"]
    require(len(bulk) == 4
            and {(row["config"], row["precision"]) for row in bulk}
                == {(config, precision) for config in ("historical-flags", "supplemental-O3")
                    for precision in ("dp", "qp")}
            and all(type(row["pairs"]) is int and row["pairs"] > 0
            and row["frames"] == 2 * row["pairs"]
            and row["failed_pairs"] == 0 and row["problems"] == {}
            and row["caller_input_words_unchanged"] is True
            and row["halting_and_rounding_unchanged"] is True
            and row["ieee_state_parity_passed"] is True and row["matrix_policy_passed"] is True
            for row in bulk), "Symmetry component comparison failed")
    traps, mutations = value["traps"], value["mutations"]
    require(type(traps["pairs"]) is int and traps["pairs"] > 0
            and traps["mismatched"] == 0 and traps["pairs"] == traps["supported"]
            and traps["post_call_evidence_for_trapped_calls"] is False
            and len(mutations) == 2
            and {row["mutation"] for row in mutations} == {"no-finite", "no-equality"}
            and all(row["detected_matrix_failure"] is True and row["failed_pairs"] > 0 for row in mutations),
            "Symmetry traps or negative controls failed")
    return {"component_pairs": sum(row["pairs"] for row in bulk), "trap_pairs": traps["pairs"],
            "negative_controls": len(mutations), "physical_amplitude_qualified": False}


def symmetry_ew100(package):
    require(package.index.get("schema") == "ew-v2-portable-runtime-v1", "Unknown symmetry runtime package")
    value = package.json("summary.json")
    comparisons = value["comparisons"]
    require(value.get("schema") == "ew-v2-full100-retained-summary-v1"
            and value["worker_exit"] == 0 and value["worker_reaped"] is True
            and value["actual_loaded_runtime_bound"] is True and value["input_words_exact"] is True
            and value["protected_inputs_unchanged"] is True
            and value["previously_accepted_DP_answer_and_accuracy_words_exact"] is True
            and value["global_campaign_qualified"] is False and value["performance_eligible"] is False,
            "Symmetry full100 scope differs")
    require(value["physical_points"] == 100 and value["public_calls_including_bootstrap"] == 101
            and all(type(value[key]) is int and value[key] >= 0 for key in (
                "new_accepted_DP_points", "new_accepted_QP_points", "old_accepted_DP_points", "old_accepted_QP_points"))
            and value["new_accepted_DP_points"] + value["new_accepted_QP_points"] == 100
            and value["old_accepted_DP_points"] + value["old_accepted_QP_points"] == 100
            and value["all_six_reference_comparisons_passed"] is True
            and set(comparisons) == {"baseline-1", "candidate-1", "candidate-6", "candidate-7", "candidate-8", "candidate-9"}
            and all(row["passed"] is True and row["ordinary_target_passed"] is True
                    and row["tolerance"] == 3e-6 and row["ordinary_target_tolerance"] == 1e-8
                    for row in comparisons.values()), "Symmetry physical coverage differs")
    require(value["W03_lines"] == value["ieee_state_differences"] == 0,
            "Symmetry warning or IEEE gate is open")
    return {"physical_points": value["physical_points"], "references": len(comparisons),
            "new_DP": value["new_accepted_DP_points"], "new_QP": value["new_accepted_QP_points"],
            "old_DP": value["old_accepted_DP_points"], "old_QP": value["old_accepted_QP_points"],
            "W03_lines": value["W03_lines"], "ieee_state_differences": value["ieee_state_differences"],
            "global_campaign_qualified": False, "performance_qualified": False}


def diagnostic(package, role):
    value = package.json("summary.json")
    if role == "cms_diagnostic":
        require(value.get("schema") == "madloop-tirust-cms-width-runtime-summary-v1"
                and value["runtime_complete"] is True and value["acceptance"] is False
                and value["physical_amplitude_qualification"] is False and value["performance_qualification"] is False
                and value["combined_status"] == "open_acceptance_items", "CMS diagnostic was promoted")
        return {key: value[key] for key in ("new_rows", "cms_tirust_dp_refusals", "cms_tirust_qp_successes",
                "cms_collier_stable_rows", "cms_ieee_flag_rows", "combined_open_items")}
    if role == "common_input_diagnostic":
        require(value.get("schema") == "madloop-tirust-common-input-retained-summary-v1"
                and value["protocol_complete"] is True and value["effective_inputs_exact"] is True
                and value["physical_captures"] == 0 and value["status"] == "open_acceptance_items",
                "Common-input diagnostic was promoted")
        require(len(value["finite_comparison_mismatches"]) == value["open_item_counts"]["mismatched"],
                "Common-input mismatch count differs")
        return {"tensor_rows": value["tensor_rows"], "effective_inputs_exact": True,
                "finite_mismatches": len(value["finite_comparison_mismatches"]),
                "open_item_counts": value["open_item_counts"], "physical_amplitude_qualified": False}
    require(package.index.get("transport_input_comparison_eligible") is True
            and package.index.get("threshold_tirust_aggregate_complete") is False
            and package.index.get("checked_qp_qualified") is False
            and package.index.get("physical_accuracy_qualified") is False
            and package.index.get("performance_eligible") is False
            and value["envelope_verified"] is True and value["transport_interpretation_eligible"] is True,
            "Mass transport diagnostic was promoted")
    require(all(row["nonmass_input_bits_match"] is True and row["failures"] == []
                for row in value["comparisons"].values()), "Mass transport input comparison failed")
    return {"tirust_arms": {key: value["cases"][key] for key in ("original-9", "candidate-9")},
            "checked_qp_qualified": False, "physical_amplitude_qualified": False}


def adaptive_profile(package, role, primitive):
    profile = role.removeprefix("adaptive_")
    require(package.index.get("scope") == f"retained {profile} standalone tensor runtime; "
            "portable Python replay only; no amplitude/performance qualification",
            "Adaptive build-profile label differs from its retained package")
    if profile == "release":
        bindings = package.json("frozen/build1/bindings.json")
        require(bindings["build_complete"] is True and bindings["physics_qualified"] is False
                and bindings["production_promoted"] is False
                and bindings["selected_release_sha256"] == primitive["adaptive_library_sha256"],
                "Adaptive release selection differs from its observed library")
    primitive["build_profile"] = profile


def installer_promotion(package):
    require(package.index.get("schema") == "tirust-strict-metadata-promotion-index-v1",
            "Unknown installer promotion package")
    value = package.json("qualification.json")
    require(value.get("schema") == "tirust-strict-metadata-production-qualification-v1"
            and value["promoted"] is True and type(value["pytest_returncode"]) is int
            and value["pytest_returncode"] == 0 and type(value["passed"]) is int and value["passed"] > 0
            and type(value["subtests_passed"]) is int and value["subtests_passed"] > 0
            and value["actual_qualified_prefix_available"] is True
            and value["actual_qualified_prefix_supports_QP"] is True
            and value["native_build_or_evaluation_performed"] is False,
            "Installer production qualification is incomplete or has different scope")
    require(value["source"] == "madgraph/various/tirust_install.py"
            and value["tests"] == "tests/unit_tests/various/test_tirust_install.py"
            and value["source_sha256"] == hashlib.sha256(package.raw("tirust_install.py")).hexdigest()
            and value["tests_sha256"] == hashlib.sha256(package.raw("test_tirust_install.py")).hexdigest(),
            "Installer receipt source/test binding differs")
    return {key: value[key] for key in ("passed", "subtests_passed", "source_sha256", "tests_sha256",
            "actual_qualified_prefix_available", "actual_qualified_prefix_supports_QP", "observation", "scope")}


def collect(evidence, selected):
    require(isinstance(selected, dict) and selected and set(selected) <= ROLES,
            "Unknown or empty implementation proof selection")
    packages = {role: Package(evidence, pin) for role, pin in selected.items()}
    source = Path(__file__).with_name("campaign_report_implementation_runtime.py")
    spec = importlib.util.spec_from_file_location("campaign_report_implementation_runtime", source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    primitives, notes = {}, []
    def note(role, title, text):
        notes.append({"title": title, "text": text, "evidence": packages[role].name,
                      "proof_role": role, "primitives": primitives[role]})
    for role, package in packages.items():
        if role.startswith("adaptive_"):
            primitives[role] = runtime.adaptive(package)
            adaptive_profile(package, role, primitives[role])
        elif role == "born_backend89":
            primitives[role] = runtime.born_backend89(package)
        elif role == "symmetry_components":
            primitives[role] = symmetry_components(package)
        elif role == "symmetry_ew100":
            primitives[role] = symmetry_ew100(package)
        elif role == "installer_promotion":
            primitives[role] = installer_promotion(package)
        elif role == "symmetry_promotion":
            primitives[role] = runtime.symmetry_promotion(package)
        elif role == "common_input_finite_audit":
            primitives[role] = runtime.finite_audit(package)
        else:
            primitives[role] = diagnostic(package, role)
    adaptive_roles = [role for role in ("adaptive_debug", "adaptive_release") if role in primitives]
    if len(adaptive_roles) == 2:
        require(packages["adaptive_debug"].checksum != packages["adaptive_release"].checksum
                and primitives["adaptive_debug"]["adaptive_library_sha256"]
                != primitives["adaptive_release"]["adaptive_library_sha256"],
                "Debug and release must bind distinct observed libraries")
        require(packages["adaptive_release"].json("frozen/build1/bindings.json")["original_debug_build_sha256"]
                == packages["adaptive_debug"].json("execution.json")["build_sha256"],
                "Adaptive release does not name the selected debug build")
    if adaptive_roles:
        descriptions = []
        for role in adaptive_roles:
            p = primitives[role]
            descriptions.append(f"{role.removeprefix('adaptive_')}: {p['genuine_binary128_input_components']} genuine binary128 components, "
                                f"{p['adaptive_tensor_calls']} adaptive and {p['legacy_tensor_calls']} legacy tensor calls")
        note(adaptive_roles[0], "Experimental adaptive-precision bridge",
             "Separately observed Fortran runtimes now cover " + "; ".join(descriptions) +
             ". Actual 256/512-bit dispatch, cache and check telemetry are retained, including the deliberate error call. "
             "These are standalone tensor/transport proofs. Accepted amplitude precision remains unobserved; "
             "full-amplitude accuracy, Born behavior and performance are not qualified. Production promotion remains pending a separate receipt.")
        notes[-1]["evidence"] = "; ".join(packages[role].name for role in adaptive_roles)
        notes[-1]["primitives"] = {role: primitives[role] for role in adaptive_roles}
    if "born_backend89" in primitives:
        p = primitives["born_backend89"]
        note("born_backend89", "Born/effective-point consistency",
             f"Actual backends 8 and 9 complete {p['ordinary_runs']} ordinary and {p['fixed_qp_runs']} fixed-QP real-Z controls. "
             f"The {p['output_logs']} retained logs contain {p['W03_lines']} W03 warnings; "
             f"{p['unchecked_status120_rows']} fixed-QP observations carry unchecked status 120. "
             "The requested Born/coordinate views and recorded coefficient comparisons pass within this control scope. "
             "Unchecked calls do not establish independent stability. CMS/stress coverage, a full campaign and performance remain unqualified.")
    if "symmetry_components" in primitives:
        p = primitives["symmetry_components"]
        note("symmetry_components", "Symmetric invariants: component proof",
             f"The v2 extracted components pass {p['component_pairs']} DP/QP pairs and {p['trap_pairs']} trap pairs; "
             f"{p['negative_controls']} deliberate mutations are detected. Caller words, IEEE state, halting and rounding safeguards are retained. "
             "This component proof alone does not qualify a physical amplitude.")
    if "symmetry_ew100" in primitives:
        p = primitives["symmetry_ew100"]
        note("symmetry_ew100", "Symmetric invariants: separate EW stream",
             f"The v2 candidate accepts {p['new_DP']} DP / {p['new_QP']} QP points on one frozen {p['physical_points']}-point EW stream, "
             f"versus {p['old_DP']} DP / {p['old_QP']} QP for its old-source comparator. All {p['references']} retained references pass 3e-6 and 1e-8; "
             f"W03 warnings and IEEE state differences are both zero. These observations belong to the separate v2 candidate. "
             "They do not replace any of the older campaign's 66 numerical lanes or its provisional timing identity, and supply no runtime acceptance.")
    if "cms_diagnostic" in primitives:
        p = primitives["cms_diagnostic"]
        note("cms_diagnostic", "Positive-width CMS diagnostic remains open",
             f"The {p['new_rows']} new positive-width rows retain {p['cms_tirust_dp_refusals']} DP refusals, "
             f"{p['cms_tirust_qp_successes']} returned QP tensors and {p['cms_collier_stable_rows']} returned COLLIER tensors. "
             f"All {p['cms_ieee_flag_rows']} rows retain IEEE flags; the combined analysis preserves "
             f"{sum(p['combined_open_items'].values())} open items, including ancestral controls. Protocol completion is not CMS acceptance.")
    if "common_input_diagnostic" in primitives:
        p = primitives["common_input_diagnostic"]
        note("common_input_diagnostic", "Common-input tensor diagnostic remains open",
             f"All {p['tensor_rows']} synthetic rows have exact effective inputs. The retained comparison still reports "
             f"{p['finite_mismatches']} comparison discrepancies, {p['open_item_counts']['refused']} refusals and "
             f"{p['open_item_counts']['host_ieee']} host IEEE observations. These are retained comparator observations, "
             "not an attribution of fault to a reducer. No physical or performance qualification follows.")
    if "mass_diagnostic" in primitives:
        p = primitives["mass_diagnostic"]
        refusal = {key: value["refusals"] for key, value in p["tirust_arms"].items()}
        note("mass_diagnostic", "Fixed-QP mass transport remains open",
             "Nonmass input comparisons pass in the retained transport diagnostic. TIRust refusal counts by its two saved points are "
             f"{refusal['original-9']} for the original arm and {refusal['candidate-9']} for the candidate arm. "
             "QP exception observations remain retained. Threshold aggregates are incomplete; checked-QP, physical accuracy and performance remain unqualified.")
    if "installer_promotion" in primitives:
        p = primitives["installer_promotion"]
        note("installer_promotion", "Installer metadata hardening",
             f"The strict JSON version-type correction is applied to production installer source {p['source_sha256'][:16]}. "
             f"The bound receipt records {p['passed']} passing tests and {p['subtests_passed']} passing subtests, "
             "with the existing qualified prefix still available and QP-capable. "
             f"Receipt provenance: {p['observation']}. "
             "This closes the selected metadata-hardening item; it does not supply new native or numerical qualification.")
        notes[-1]["replaces"] = "Installer metadata hardening remains open"
    if "symmetry_promotion" in primitives:
        p = primitives["symmetry_promotion"]
        note("symmetry_promotion", "Symmetric invariants: Fortran template promoted",
             f"The applied Fortran template is {p['applied_template_sha256'][:16]}. Its bound source-comparison receipt records "
             f"{p['canonical_statement_count']} matching canonical statements in the qualified EW adapter context. "
             f"All {p['passed']} tests pass: {p['compiled_fixture_test_methods']} compiled fixture methods and "
             f"{p['pure_python_test_methods']} Python permutation method. No real reducer or new physical replay is invoked by these tests. "
             "New exports can use this template; the older campaign's 66 lanes and provisional timings remain tied to their original source. "
             "Adaptive-bridge production promotion and final performance acceptance remain separate.")
    if "common_input_finite_audit" in primitives:
        p = primitives["common_input_finite_audit"]
        note("common_input_finite_audit", "Higher-rank finite residual clarification",
             f"The additive audit separates the worst finite symmetric residual, approximately {float(p['maximum_finite_symmetric_residual']):.10g}, "
             f"from the aggregate maximum {p['maximum_aggregate_symmetric_residual']} in the {p['maximum_aggregate_slot']}-pole slot. "
             "The aggregate value includes near-zero UV comparison noise; it is not the finite residual. "
             "The original diagnostic remains preserved. Finite discrepancies and the comparator's requested-accuracy observations remain open; "
             "this offline audit establishes neither a reducer fault nor physical accuracy or performance.")
    return {"schema": "madloop-tirust-selected-implementation-supplement-v1",
            "packages": {role: {"index": p.name, "sha256": p.checksum} for role, p in packages.items()},
            "primitives": primitives, "notes": notes, "production_promotion": "pending separate receipt",
            "strict_installer_promotion": "applied; recorded tests passed" if "installer_promotion" in primitives else "pending separate receipt",
            "symmetry_fortran_promotion": "applied; recording tests passed" if "symmetry_promotion" in primitives else "pending separate receipt",
            "main_campaign_modified": False,
            "performance_acceptance_eligible": False}


def apply(refreshed, supplement):
    """Only the diagnostic prose is replaced; no campaign or acceptance mutation."""
    replacements = {row.get("replaces", row["title"]): row for row in supplement["notes"]}
    old_titles = {row["title"] for row in refreshed["private_diagnostics"]}
    refreshed["private_diagnostics"] = [replacements.get(row["title"], row) for row in refreshed["private_diagnostics"]]
    refreshed["private_diagnostics"].extend(row for row in supplement["notes"]
                                          if row.get("replaces", row["title"]) not in old_titles)
    refreshed["implementation_supplement"] = supplement
