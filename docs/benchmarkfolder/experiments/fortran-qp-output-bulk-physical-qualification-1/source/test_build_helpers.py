"""Pure argv/make/nm/map fixtures; never run make, compilers or loaders."""
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("physical_build_helpers", HERE / "build_helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
COMPILER = "/reviewed/gfortran"
MAKEFILE = """$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
\t$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)
TIRUST_interface.o: TIRUST_interface.f
\t$(FC) $(FFLAGS) -c $< -o $@
"""


@pytest.fixture(autouse=True)
def compiler_identity(monkeypatch):
    monkeypatch.setattr(helpers.shutil, "which", lambda text, path=None:
                        COMPILER if text in {COMPILER, "gfortran"} else None)


def commands(wire=False):
    module = COMPILER + " -O3 -fPIC -c mtirust.f90 -o mtirust.o"
    worker = COMPILER + " -w -fPIC -ffixed-line-length-132 -c check_sa.f -o check_sa.o -I ../../Source/MODEL"
    link = (COMPILER + " -w -fPIC " + helpers.LINK_MAP_OPTION +
            (" " + helpers.WRAP_OPTION if wire else "") +
            " -o check check_sa.o mtirust.o" + (" wire_capture.o" if wire else "") +
            " TIRUST_interface.o TIR_interface.o -ldhelas -lmodel -ltirust -lmadloop_tirust")
    return [module, worker, link]


def select(lines, wire=False):
    return helpers.select_commands("\n".join(lines), COMPILER, {"PATH": "/reviewed"}, wire=wire)


@pytest.mark.parametrize("wire", [False, True])
def test_private_makefile_preserves_numeric_recipes(wire):
    result = helpers.private_makefile(MAKEFILE, wire=wire)
    assert "TIRUST_interface.o: TIRUST_interface.f\n\t$(FC) $(FFLAGS) -c $< -o $@" in result
    assert "$(PROG): check_sa.o mtirust.o" in result
    assert "mtirust.o: mtirust.f90\n\t$(FC) -O3 -fPIC -c $< -o $@" in result
    assert result.count(helpers.LINK_MAP_OPTION) == 1
    assert result.count(helpers.WRAP_OPTION) == int(wire)
    assert "TIRUST_interface.o: mtirust.o" not in result
    assert "TIR_interface.o: mtirust.o" not in result
    with pytest.raises(ValueError):
        helpers.private_makefile(result, wire=wire)


@pytest.mark.parametrize("text", ["", MAKEFILE + MAKEFILE,
    MAKEFILE.replace("$(PROG): check_sa.o $(PROCESS)", "check: check_sa.o"),
    MAKEFILE.replace("-o $(PROG)", "-o other")])
def test_missing_or_ambiguous_make_anchor_rejected(text):
    with pytest.raises(ValueError):
        helpers.private_makefile(text)


@pytest.mark.parametrize("wire", [False, True])
def test_select_exactly_two_sources_one_link_and_ignore_numerics(wire):
    lines = commands(wire)
    lines.insert(1, COMPILER + " -w -c TIRUST_interface.f -o TIRUST_interface.o")
    lines.insert(1, "cd ../../Source/DHELAS; make -n all")
    selected, link = select(lines, wire=wire)
    assert [name for name, _ in selected] == ["mtirust.f90", "check_sa.f"]
    assert len(selected) == 2 and link == commands(wire)[2].split()


@pytest.mark.parametrize("bad", [
    "; echo bad", "&&echo bad", "| cat", "> output", "`true`", "$(true)",
    " @response", " -Wl,@response", " # comment",
])
def test_shell_compound_response_and_comment_rejected(bad):
    lines = commands()
    lines[1] += bad
    with pytest.raises(ValueError):
        select(lines)


@pytest.mark.parametrize("change", [
    lambda a: [a[0], a[0], *a[1:]],
    lambda a: a[1:],
    lambda a: [*a, a[-1]],
    lambda a: [a[0], a[1].replace("check_sa.f", "path/check_sa.f"), a[2]],
    lambda a: [a[0], a[1] + " extra.f", a[2]],
    lambda a: [a[0], a[1] + " extra.c", a[2]],
    lambda a: [a[0], a[1].replace("check_sa.o", "other.o"), a[2]],
    lambda a: [a[0], a[1].replace("-c ", "-c -c "), a[2]],
    lambda a: [a[0], a[1] + " -o again.o", a[2]],
    lambda a: [a[0], a[1] + " -oagain.o", a[2]],
    lambda a: [a[0].replace("-O3", "-O2"), *a[1:]],
    lambda a: [a[0].replace("-fPIC ", ""), *a[1:]],
    lambda a: [a[0].replace(COMPILER, "/wrong/gfortran"), *a[1:]],
    lambda a: [a[0], a[1], a[2] + " extra.f90"],
    lambda a: [a[0], a[1], a[2].replace("mtirust.o ", "") + " mtirust.o"],
    lambda a: [a[0], a[1], a[2].replace("mtirust.o", "mtirust.o mtirust.o")],
    lambda a: [a[0], a[1], a[2] + " ./mtirust.o"],
    lambda a: [a[0], a[1], a[2] + " /old/check_sa.o"],
    lambda a: [a[0], a[1], a[2].replace(helpers.LINK_MAP_OPTION, "")],
    lambda a: [a[0], a[1], a[2] + " -Wl,-Map,other.map"],
    lambda a: [a[0], a[1], a[2].replace("mtirust.o TIRUST_interface.o", "TIRUST_interface.o mtirust.o")],
])
def test_ambiguous_source_compiler_flags_output_and_link_rejected(change):
    with pytest.raises(ValueError):
        select(change(commands()))


def test_observer_link_is_explicit_and_exclusive():
    with pytest.raises(ValueError):
        select(commands(True), wire=False)
    with pytest.raises(ValueError):
        select(commands(False), wire=True)
    with pytest.raises(ValueError):
        select([*commands()[:-1], commands()[-1] + " -Wl,--wrap=other"])


MODULE_NM = """0000000000000000 T __mtirust_MOD_tensor_mp
0000000000000000 b __mtirust_MOD_context
0000000000000000 t __mtirust_MOD_pack
"""
EXE_NM = """0000000000400100 T __mtirust_MOD_tensor_mp
0000000000600200 b __mtirust_MOD_context
0000000000400300 t __mtirust_MOD_pack
0000000000400400 T MAIN__
"""
MAP = "LOAD check_sa.o\nLOAD mtirust.o\nLOAD TIRUST_interface.o\n"


@pytest.mark.parametrize("wire", [False, True])
def test_symbols_private_owner_and_explicit_wire_wrapper(wire):
    exe = EXE_NM + ("0000000000400500 T __wrap_ml_tirust_evaluate_mp\n" if wire else "")
    mapping = MAP + ("LOAD wire_capture.o\n" if wire else "")
    result = helpers.verify_symbols(MODULE_NM, exe, mapping, wire=wire)
    assert len(result["definitions"]) == 3
    assert result["wire_wrapper_definitions"] == int(wire)
    assert result["original_archive_member_loaded"] is False


@pytest.mark.parametrize("module,exe,mapping,wire", [
    ("", EXE_NM, MAP, False),
    (MODULE_NM + MODULE_NM, EXE_NM, MAP, False),
    (MODULE_NM, EXE_NM + EXE_NM, MAP, False),
    (MODULE_NM, EXE_NM.replace("__mtirust_MOD_pack", "other"), MAP, False),
    (MODULE_NM, EXE_NM, MAP + "LOAD mtirust.o\n", False),
    (MODULE_NM, EXE_NM, MAP.replace("LOAD mtirust.o\n", ""), False),
    (MODULE_NM, EXE_NM, MAP + "/lib/libtirust.a(mtirust.o)\n", False),
    (MODULE_NM, EXE_NM, MAP + "/lib/libtirust.a(gfortran_mtirust.f90.o)\n", False),
    (MODULE_NM, EXE_NM + "0000000000400500 T __wrap_ml_tirust_evaluate_mp\n", MAP, False),
    (MODULE_NM, EXE_NM, MAP, True),
    (MODULE_NM, EXE_NM, MAP + "LOAD wire_capture.o\n", False),
    (MODULE_NM, EXE_NM + "0000000000400600 T __real_ml_tirust_evaluate_mp\n", MAP, False),
    (MODULE_NM + "0000000000000000 U __mtirust_MOD_fake\n", EXE_NM, MAP, False),
])
def test_symbol_or_archive_ownership_failure_rejected(module, exe, mapping, wire):
    with pytest.raises(ValueError):
        helpers.verify_symbols(module, exe, mapping, wire=wire)
