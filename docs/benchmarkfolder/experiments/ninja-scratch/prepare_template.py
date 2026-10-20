#!/usr/bin/env python3
"""Prepare a reviewable template diff without modifying production sources."""
import difflib
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = ROOT / ".benchmark-work/experiments/ninja-scratch"
HERE.mkdir(parents=True, exist_ok=True)
TARGET = "madgraph/iolibs/template_files/loop_optimized/CT_interface.inc"
source = (ROOT / TARGET).read_text()


def optimized_body(body, mp=False):
    assert body.count("      use mninja\n") == 1
    body = body.replace("      use mninja\n", "      use mninja\n"
        "      USE %(proc_prefix)sPOLYNOMIAL_CONSTANTS, ONLY: NCOEF_R,COEFTORANK_MAP\n")
    if mp:
        body = body.replace(
            "\t  %(complex_mp_format)s, ALLOCATABLE :: MP_TENSORCOEFS(:)\n"
            "\t  complex(KI_QNIN), ALLOCATABLE :: MP_NINJA_TENSORCOEFS(:)",
            "      %(complex_mp_format)s MP_COEFFICIENT\n"
            "      complex(KI_QNIN) MP_NINJA_TENSORCOEFS(0:LOOPMAXCOEFS-1)")
    else:
        body = body.replace("\t  %(complex_dp_format)s, ALLOCATABLE :: TENSORCOEFS(:)",
                            "      %(complex_dp_format)s TENSORCOEFS(0:LOOPMAXCOEFS-1)")
    start = body.index("C     Number of coefficients for the current rank")
    end = body.index("C     Compute the kinematic matrix", start)
    variable, coefficients = ("MP_COEFFICIENT", "MP_LOOPCOEFS") if mp else ("TENSORCOEFS(I)", "LOOPCOEFS")
    preparation = f"""C     Copy the active tensor, applying q_loop -> -q_loop.
      CURR_MAXCOEF = NCOEF_R(RANK)
      DO I=0,CURR_MAXCOEF-1
## if(AmplitudeReduction) {{
        {variable} = {coefficients}(I,ID)
## }} else {{
        {variable} = {coefficients}(I,SQSOINDEX,ID)
## }}
        IF (MOD(COEFTORANK_MAP(I),2).EQ.1) {variable}=-{variable}
"""
    if mp:
        preparation += "        MP_NINJA_TENSORCOEFS(I)=CMPLX(MP_COEFFICIENT,KIND=KI_QNIN)\n"
    preparation += "      ENDDO\n\n"
    body = body[:start] + preparation + body[end:]
    if mp:
        body, count = re.subn(
            r"      DO I=0,CURR_MAXCOEF-1\n\s*MP_NINJA_TENSORCOEFS\(I\)=CMPLX\(MP_TENSORCOEFS\(I\),KIND=KI_QNIN\)\n      ENDDO\n",
            "", body)
        assert count == 1
    body, count = re.subn(r"C     Make sure to deallocate the tensor of coefficients\n.*?(?=## if\(AmplitudeReduction\))",
                         "", body, flags=re.S)
    assert count == 1
    variable = "MP_NINJA_TENSORCOEFS" if mp else "TENSORCOEFS"
    body = body.replace(f"call ninja_tensor_evaluate({variable},",
                        f"call ninja_tensor_evaluate({variable}(0:CURR_MAXCOEF-1),")
    assert "ALLOCATABLE" not in body
    assert "INVERT_MOMENTA_IN_POLYNOMIAL" not in body
    return body


dp = source.index("      SUBROUTINE %(proc_prefix)sNINJA_LOOP")
mp = source.index("      SUBROUTINE %(proc_prefix)sMP_NINJA_LOOP")
end = source.index("\t  SUBROUTINE %(proc_prefix)sINITNINJA")
result = (source[:dp] + optimized_body(source[dp:mp]) +
          optimized_body(source[mp:end], mp=True) + source[end:])
(HERE / "CT_interface.optimized.inc").write_text(result)
(HERE / "production-template.patch").write_text("".join(difflib.unified_diff(
    source.splitlines(True), result.splitlines(True),
    fromfile="a/" + TARGET, tofile="b/" + TARGET)))
print("Prepared", HERE / "production-template.patch")
