"""Pure-Python adversarial checks; no compiler, worker or numerical execution."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = ROOT / ".benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z"
PINS = ROOT / "docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json"
LOGS = ROOT / "docs/benchmarkfolder/inputs/candidate/qcd_n1_uux_z"
SPEC = importlib.util.spec_from_file_location("consumer_audit", HERE / "audit_consumers.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture(scope="module")
def frozen():
    pins = AUDIT.load_pins(PINS)
    source = AUDIT.source_inventory(BASE, pins)
    units, _ = AUDIT.actual_compile_units(pins, AUDIT.compile_units(pins), LOGS)
    return pins, source, units


def materialize(root, sources):
    for name, data in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


@pytest.fixture
def copied(tmp_path, frozen):
    return materialize(tmp_path / "baseline", frozen[1])


def transform_independently(caller, polynomial, helper):
    # Load only the original builder's pure transform functions; importing the
    # builder would load unrelated build/protocol machinery.
    parsed = ast.parse((HERE.parent / "prepare_amplitude.py").read_text())
    pure = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
                           and node.name in {"once", "fixed_form", "transform"}], type_ignores=[])
    namespace = {"re": re}
    exec(compile(pure, "independent_builder_functions", "exec"), namespace)
    return namespace["transform"](caller.decode(), polynomial.decode(), helper.decode())


@pytest.fixture
def variant(tmp_path, frozen):
    root = materialize(tmp_path / "variant", frozen[1])
    helper = HERE.parent / "terminal.f90"
    caller, polynomial = transform_independently(frozen[1][AUDIT.CALLER],
                                                 frozen[1][AUDIT.POLYNOMIAL], helper.read_bytes())
    (root / AUDIT.CALLER).write_text(caller)
    (root / AUDIT.POLYNOMIAL).write_text(polynomial)
    return root, helper


def check_variant(variant, **kwargs):
    root, helper = variant
    return AUDIT.audit(BASE, PINS, root, helper, AUDIT.sha(helper.read_bytes()), **kwargs)


def check_semantics(source, frozen):
    expanded, _ = AUDIT.expanded_units(source, frozen[2])
    return AUDIT.validate_consumers(expanded)


def test_frozen_actual_compile_and_include_closure():
    result = AUDIT.audit(BASE, PINS)
    assert result["source_count"] == 111
    assert result["compile_recipe_unit_count"] == 52
    assert result["actual_compiled_unit_count"] == 49
    assert result["textual_closure_count"] == 79
    assert result["resolved_closure_count"] == 76
    assert result["include_file_count"] == 30
    assert result["distinct_include_edge_count"] == 97
    assert {Path(name).name for name in result["uncompiled_recipe_units"]} == {
        "testprog.f", "check_sa_born_splitOrders.f", "f2py_wrapper.f"}
    assert result["consumer_proof"]["common_aliases"][AUDIT.PROCESS + "/mp_compute_loop_coefs.f"] == "DP_WL"
    assert len(result["consumer_proof"]["executable_uses"]) == 18
    assert result["native_execution"] is False


@pytest.mark.parametrize("helper_name", ["terminal.f90", "packed-scan/terminal.f90"])
def test_exact_independent_transforms_pass(tmp_path, frozen, helper_name):
    root = materialize(tmp_path / "variant", frozen[1])
    helper = HERE.parent / helper_name
    caller, polynomial = transform_independently(frozen[1][AUDIT.CALLER],
                                                 frozen[1][AUDIT.POLYNOMIAL], helper.read_bytes())
    (root / AUDIT.CALLER).write_text(caller)
    (root / AUDIT.POLYNOMIAL).write_text(polynomial)
    result = check_variant((root, helper))
    assert result["transformed"] is True
    assert set(result["changed_sources"]) == {AUDIT.CALLER, AUDIT.POLYNOMIAL}
    assert result["original_merger_and_create_bytes_unchanged"] is True


@pytest.mark.parametrize("name,old,new,error", [
    ("mp_compute_loop_coefs.f", "COMMON/ML5_0_WL/DP_WL,DP_PL", "COMMON/ML5_0_WL/OTHER,DP_PL", "alias/layout"),
    ("mp_compute_loop_coefs.f", "COMMON/ML5_0_WL/DP_WL,DP_PL", "COMMON/ML5_0_WL/DP_PL,DP_WL", "alias/layout"),
    ("mp_compute_loop_coefs.f", "DP_WL(I,J,K,-1)=(0.0D0,0.0D0)", "DP_WL(I,J,K,3)=(0.0D0,0.0D0)", "Non-seed"),
    ("loop_matrix.f", "WL(I,J,K,-1)=(0.0D0,0.0D0)", "CALL HIDDEN(WL)", "Non-seed"),
    ("loop_matrix.f", "WL(I,J,K,-1)=(0.0D0,0.0D0)", "EQUIVALENCE(WL,ESCAPE)", "Non-seed"),
    ("coef_construction_1.f", "CALL ML5_0_CREATE_LOOP_COEFS", "CALL HIDDEN", "consumer graph"),
    ("coef_construction_1.f", "WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3)", "WL(1,0,1,3),4,COEFS,4,4,WL(1,0,1,3)", "consumer graph"),
    ("polynomial.f", "OUT(J)=OUT(J)+WL(I,J,I)*CONST", "OUT(J)=OUT(J)+WL(I,J,1)*CONST", "merger implementation"),
])
def test_semantic_graph_rejects_mutation_without_relying_on_hashes(frozen, name, old, new, error):
    source = dict(frozen[1])
    name = AUDIT.PROCESS + "/" + name
    assert old.encode() in source[name]
    source[name] = source[name].replace(old.encode(), new.encode(), 1)
    with pytest.raises(AUDIT.AuditError, match=error):
        check_semantics(source, frozen)


def test_semantic_graph_rejects_later_consumer(frozen):
    source = dict(frozen[1])
    source[AUDIT.CALLER] += b"      CALL ML5_0_UPDATE_WL_1_1(WL(1,0,1,3),4,COEFS,4,4,WL(1,0,1,4))\n"
    with pytest.raises(AUDIT.AuditError, match="consumer graph"):
        check_semantics(source, frozen)


def test_reachable_include_hidden_consumer_is_rejected(frozen):
    source = dict(frozen[1])
    source[AUDIT.PROCESS + "/coupl.inc"] += b"      X=WL(1,0,2,3)\n"
    with pytest.raises(AUDIT.AuditError, match="Included DP COMMON consumer"):
        check_semantics(source, frozen)


@pytest.mark.parametrize("target", ["missing.inc", "../../../../outside.inc", "coupl.inc"])
def test_include_closure_rejects_unpinned_escaping_or_cyclic_input(frozen, target):
    source = dict(frozen[1])
    source[AUDIT.PROCESS + "/coupl.inc"] += ("      INCLUDE '" + target + "'\n").encode()
    with pytest.raises(AUDIT.AuditError, match="INCLUDE"):
        AUDIT.expanded_units(source, frozen[2])


def test_source_mutation_cannot_be_resealed_by_changing_manifest(copied, tmp_path):
    manifest = json.loads(PINS.read_text())
    (copied / AUDIT.CALLER).write_bytes(b"      CALL HIDDEN(WL)\n")
    manifest["compilation"]["sources"][AUDIT.CALLER] = AUDIT.sha((copied / AUDIT.CALLER).read_bytes())
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps(manifest))
    with pytest.raises(AUDIT.AuditError, match="candidate-inputs identity"):
        AUDIT.audit(copied, pins, build_logs_dir=LOGS)


@pytest.mark.parametrize("name", ["hidden.f", "hidden.f90", "shadow.inc", "GNUmakefile"])
def test_unknown_custom_source_or_include_rejected(copied, name):
    (copied / AUDIT.PROCESS / name).write_text("      COMMON/ML5_0_WL/WL,PL\n")
    with pytest.raises(AUDIT.AuditError, match="Unknown source/include/build"):
        AUDIT.audit(copied, PINS)


@pytest.mark.parametrize("kind", ["unknown-alias", "outside", "baseline-backlink"])
def test_unrecognized_symlinks_rejected(copied, tmp_path, kind):
    name = AUDIT.PROCESS + "/coupl.inc"
    path = copied / name
    data = path.read_bytes()
    path.unlink()
    if kind == "outside":
        target = tmp_path / "outside.inc"
        target.write_bytes(data)
    elif kind == "baseline-backlink":
        target = BASE / name
    else:
        target = copied / "Source/MODEL/coupl.inc"
    path.symlink_to(target)
    with pytest.raises(AUDIT.AuditError, match="escapes|Unrecognized source symlink"):
        AUDIT.audit(copied, PINS)


def test_known_internal_source_alias_is_accepted(copied):
    name = AUDIT.PROCESS + "/coupl.inc"
    path = copied / name
    path.unlink()
    path.symlink_to(AUDIT.LINKS[name])
    assert AUDIT.audit(copied, PINS)["passed"]


def test_changed_historical_build_log_rejected(tmp_path, frozen):
    for name in ("DHELAS", "MODEL", "process"):
        path = tmp_path / ("compilation-" + name + ".log")
        path.write_bytes((LOGS / path.name).read_bytes())
    (tmp_path / "compilation-process.log").write_text("compile nothing\n")
    with pytest.raises(AUDIT.AuditError, match="build log changed"):
        AUDIT.audit(BASE, PINS, build_logs_dir=tmp_path)


def test_dropped_actual_compile_root_rejected(frozen):
    units = AUDIT.compile_units(frozen[0])
    del units[AUDIT.PROCESS + "/mp_compute_loop_coefs.f"]
    with pytest.raises(AUDIT.AuditError, match="Actual compile command differs"):
        AUDIT.actual_compile_units(frozen[0], units, LOGS)


def test_changed_compiler_include_order_rejected(frozen):
    units = deepcopy(AUDIT.compile_units(frozen[0]))
    argv = units[AUDIT.CALLER]["argv"]
    first, second = argv.index("-I../../Source/MODEL"), argv.index("-I../../Source/DHELAS")
    argv[first], argv[second] = argv[second], argv[first]
    with pytest.raises(AUDIT.AuditError, match="Actual compile command differs"):
        AUDIT.actual_compile_units(frozen[0], units, LOGS)


@pytest.mark.parametrize("mutation", ["caller-argument", "capture-removed", "merge", "duplicate-helper", "qp-change", "label-column"])
def test_transformed_delta_rejects_any_unapproved_edit(variant, mutation):
    root, _ = variant
    path = root / (AUDIT.CALLER if mutation in ("caller-argument", "capture-removed") else AUDIT.POLYNOMIAL)
    data = path.read_bytes()
    if mutation == "caller-argument":
        data = data.replace(b"WL(1,0,1,3),DP_TERMINAL_ENTRY_INEXACT", b"WL(1,0,1,2),DP_TERMINAL_ENTRY_INEXACT", 1)
    elif mutation == "capture-removed":
        data = data.replace(b"      CALL IEEE_GET_FLAG(IEEE_INEXACT,DP_TERMINAL_ENTRY_INEXACT)\n", b"", 1)
    elif mutation == "merge":
        data = data.replace(b"WL(I,J,I)*CONST", b"WL(I,J,1)*CONST", 1)
    elif mutation == "duplicate-helper":
        data += b"      SUBROUTINE ML5_0_DP_TERMINAL_1_1()\n      END\n"
    elif mutation == "qp-change":
        data = data.replace(b"SUBROUTINE MP_ML5_0_MERGE_WL", b"SUBROUTINE MP_ML5_0_ALTERED_MERGE_WL", 1)
    else:
        data = data.replace(b"  900 continue", b"      900 continue", 1)
    assert data != path.read_bytes()
    path.write_bytes(data)
    with pytest.raises(AUDIT.AuditError, match="Unexpected transform delta"):
        check_variant(variant)


def test_unapproved_helper_rejected_even_with_matching_supplied_hash(variant, tmp_path):
    helper = tmp_path / "helper.f90"
    helper.write_bytes(variant[1].read_bytes().replace(b"if (.not.ENTRY_INEXACT) goto 900", b"! removed guard"))
    with pytest.raises(AUDIT.AuditError, match="Unreviewed terminal helper"):
        check_variant((variant[0], helper))


def test_trace_delta_requires_explicit_diagnostic_selection(variant):
    path = variant[0] / AUDIT.CALLER
    anchor = b"      CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1,1,1,2,H)\n"
    path.write_bytes(path.read_bytes().replace(anchor,
        b"      WRITE(*,*) 'DP_TERMINAL_ADMISSION',\n"
        b"     $ DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST\n" + anchor))
    with pytest.raises(AUDIT.AuditError, match="Unexpected transform delta"):
        check_variant(variant)
    result = check_variant(variant, trace_admission=True)
    assert result["trace_is_diagnostic_only"] is True
    assert result["performance_eligible"] is False
