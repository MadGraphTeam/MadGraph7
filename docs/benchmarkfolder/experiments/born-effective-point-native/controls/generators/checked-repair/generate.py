#!/usr/bin/env python3
"""Render private checked-repair siblings; never edit a production source."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PREFIX = "%(proc_prefix)s%(mp_prefix)s"
LEGACY_NAME = PREFIX + "IMPROVE_PS_POINT_PRECISION"
CHECKED_NAME = LEGACY_NAME + "_CHECKED"
HEADER = "      SUBROUTINE " + LEGACY_NAME + "(KEEP_OFFSHELL_MASS, P)\n"
CHECKED_HEADER = ("      SUBROUTINE " + CHECKED_NAME +
    "(KEEP_OFFSHELL_MASS, P, ERR_PRIMARY, ERR_ALTERNATE, OUTCOME, ADOPTED)\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Source anchor must occur exactly once: " + repr(old))
    return text.replace(old, new, 1)


def annotation(name, text):
    return ("C CHECKED-REPORT-BEGIN " + name + "\n" + text +
            "C CHECKED-REPORT-END " + name + "\n")


def remove_annotations(text):
    return re.sub(r"(?m)^C CHECKED-REPORT-BEGIN ([A-Z_]+)\n.*?"
                  r"^C CHECKED-REPORT-END \1\n", "", text,
                  flags=re.DOTALL)


def extract_wrapper(source):
    if source.count(HEADER) != 1:
        raise ValueError("Expected one legacy MP repair entry")
    start = source.index(HEADER)
    end = re.search(r"(?m)^[ \t]*END[ \t]*\n", source[start:])
    if end is None:
        raise ValueError("Cannot locate legacy MP repair terminator")
    return source[start:start + end.end()]


def checked_wrapper(source):
    """Clone the wrapper, adding only integer/logical reporting and a mode guard.

    All numerical statements and helper calls are byte-preserved. The helpers
    themselves remain the single original copies in the source unit.
    """
    legacy = extract_wrapper(source)
    checked = replace_once(legacy, HEADER, CHECKED_HEADER)
    checked = replace_once(checked,
        "\t  LOGICAL KEEP_OFFSHELL_MASS(NEXTERNAL)\n",
        "\t  LOGICAL KEEP_OFFSHELL_MASS(NEXTERNAL)\n" + annotation("DECLARATIONS",
        "      INTEGER ERR_PRIMARY, ERR_ALTERNATE, OUTCOME\n"
        "      LOGICAL ADOPTED\n"))
    checked = replace_once(checked, "      ERRCODETMP=0\n",
        annotation("INITIAL_STATE",
        "      ADOPTED=.FALSE.\n"
        "      OUTCOME=0\n") + "      ERRCODETMP=0\n")
    checked = replace_once(checked, "C Check the sanity of the original PS point\n",
        annotation("SUPPORTED_METHOD",
        "C     The legacy >2 branch invokes no numerical repair method.\n"
        "C     The checked entry must not certify that no-op as a repair.\n"
        "      IF (ImprovePSPoint.GT.2) THEN\n"
        "        OUTCOME=-3\n"
        "        GOTO 102\n"
        "      ENDIF\n") + "C Check the sanity of the original PS point\n")
    checked = replace_once(checked, "\t    ERRCODE = 1\n",
        "\t    ERRCODE = 1\n" + annotation("INPUT_REJECTED",
        "      OUTCOME=-1\n"))
    checked = replace_once(checked, "\t  IF (ERRCODE.NE.0) THEN\n",
        annotation("PRIMARY_COMPLETED", "      OUTCOME=1\n") +
        "\t  IF (ERRCODE.NE.0) THEN\n")
    checked = replace_once(checked,
        "\t\tIF (ERRCODETMP.NE.0) GOTO 100\n",
        annotation("ALTERNATE_COMPLETED",
        "      OUTCOME=-2\n"
        "      IF (ERRCODETMP.EQ.0) OUTCOME=2\n") +
        "\t\tIF (ERRCODETMP.NE.0) GOTO 100\n")
    checked = replace_once(checked, "102   CONTINUE\n",
        annotation("ADOPTED", "      ADOPTED=.TRUE.\n") +
        "102   CONTINUE\n" + annotation("RETURN_ERRORS",
        "      ERR_PRIMARY=ERRCODE\n"
        "      ERR_ALTERNATE=ERRCODETMP\n"))
    recovered = remove_annotations(checked).replace(CHECKED_HEADER, HEADER, 1)
    if recovered != legacy:
        raise AssertionError("Checked wrapper altered legacy statements")
    return legacy, checked


HOST = """C CHECKED PREPARATION: algorithm outcome only, not an admission certificate.
C The caller owns finite/IEEE/conversion/Born checks before publishing C128.
      SUBROUTINE %(proc_prefix)sPREPARE_MP_PS_CHECKED(P_USER,C128,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P_USER(0:3,NEXTERNAL)
      %(real_mp_format)s C128(0:3,NEXTERNAL)
      INTEGER ERR_PRIMARY,ERR_ALTERNATE,OUTCOME
      LOGICAL ADOPTED
      %(real_mp_format)s P_LOCAL(0:3,NEXTERNAL)
      LOGICAL KEEP_OFFSHELL_MASS(NEXTERNAL)
      INTEGER I,J

      %(keep_offshell_mass)s
      DO I=1,NEXTERNAL
        DO J=0,3
          P_LOCAL(J,I)=P_USER(J,I)
        ENDDO
      ENDDO
      CALL %(proc_prefix)sMP_IMPROVE_PS_POINT_PRECISION_CHECKED(KEEP_OFFSHELL_MASS,P_LOCAL,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)
      IF (.NOT.ADOPTED) RETURN
      DO I=1,NEXTERNAL
        DO J=0,3
          C128(J,I)=P_LOCAL(J,I)
        ENDDO
      ENDDO
      END

C OPTIONAL ALGORITHM-ONLY SETTER. Main effective-point host uses PREPARE above.
C This routine does not set readiness/completion flags or validate Born/IEEE.
      SUBROUTINE %(proc_prefix)sSET_MP_PS_CHECKED(P_USER,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      %(real_dp_format)s P_USER(0:3,NEXTERNAL)
      %(real_mp_format)s C128(0:3,NEXTERNAL)
      %(real_mp_format)s MP_PS(0:3,NEXTERNAL),MP_P(0:3,NEXTERNAL)
      COMMON/%(proc_prefix)sMP_PSPOINT/MP_PS,MP_P
      INTEGER ERR_PRIMARY,ERR_ALTERNATE,OUTCOME
      LOGICAL ADOPTED
      INTEGER I,J
      CALL %(proc_prefix)sPREPARE_MP_PS_CHECKED(P_USER,C128,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)
      IF (.NOT.ADOPTED) RETURN
      DO I=1,NEXTERNAL
        DO J=0,3
          MP_PS(J,I)=C128(J,I)
          MP_P(J,I)=C128(J,I)
        ENDDO
      ENDDO
      END
"""


def source_inputs():
    metadata = json.loads((HERE / "inputs/source.json").read_text())
    result = {}
    for record in metadata["sources"]:
        data = (HERE / record["copy"]).read_bytes()
        if digest(data) != record["sha256"]:
            raise ValueError("Frozen input identity changed: " + record["copy"])
        result[Path(record["path"]).name] = data.decode()
    return metadata, result


def fixture_parameters(name, masses, offshell=()):
    n = len(masses)
    definitions = []
    keep = []
    for i, mass in enumerate(masses, 1):
        keep.append("KEEP_OFFSHELL_MASS(%d)=%s" %
                    (i, ".TRUE." if i in offshell else ".FALSE."))
        if i in offshell:
            definitions.append("MASSES(%d)=SQRT(ABS(P(0,%d)**2-P(1,%d)**2-P(2,%d)**2-P(3,%d)**2))" % ((i,)*5))
        else:
            definitions.append("MASSES(%d)=MP__%s" % (i, mass))
    return dict(proc_prefix="CHECK_" + name.upper() + "_", mp_prefix="MP_",
                nexternal=n, ninitial=2, real_format="REAL*16",
                real_mp_format="REAL*16", real_dp_format="REAL*8",
                exp_letter="e", mp_specifier="_16", include_vector="",
                coupl_inc_name="mp_coupl.inc",
                masses_def="\n      ".join(definitions),
                keep_offshell_mass="\n      ".join(keep))


def build_outputs():
    metadata, source = source_inputs()
    legacy, checked = checked_wrapper(source["improve_ps.inc"])
    improve = source["improve_ps.inc"] + "\n" + checked
    host = source["loop_matrix_standalone.inc"] + "\n" + HOST
    result = {"generated/improve_ps.inc": improve,
              "generated/loop_matrix_standalone.inc": host,
              "generated/checked_host.inc": HOST}
    patch = []
    for name, candidate in [("improve_ps.inc", improve),
                            ("loop_matrix_standalone.inc", host)]:
        rel = next(r["path"] for r in metadata["sources"]
                   if Path(r["path"]).name == name)
        patch.extend(difflib.unified_diff(source[name].splitlines(keepends=True),
                     candidate.splitlines(keepends=True),
                     fromfile="a/" + rel, tofile="b/" + rel))
    result["patches/checked-repair.patch"] = "".join(patch)
    fixtures = {
        "real_n3": fixture_parameters("real_n3", ["ZERO", "ZERO", "MDL_MZ"]),
        "real_n5": fixture_parameters("real_n5", ["ZERO", "ZERO", "MDL_MT", "MDL_MT", "ZERO"]),
        "cms_external_n4": fixture_parameters("cms_n4", ["ZERO", "ZERO", "MDL_MT", "MDL_MT"]),
        "legacy_offshell_n4": fixture_parameters("offshell_n4", ["ZERO", "ZERO", "MDL_MT", "MDL_MT"], (3,)),
    }
    for name, parameters in fixtures.items():
        result["rendered/" + name + "/improve_ps.f"] = improve % parameters
        result["rendered/" + name + "/checked_host.f"] = HOST % parameters
    result["api.json"] = json.dumps({
        "schema": 1, "native_qualified": False,
        "repair": "<prefix>MP_IMPROVE_PS_POINT_PRECISION_CHECKED(KEEP_OFFSHELL_MASS,P,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)",
        "primary_host": "<prefix>PREPARE_MP_PS_CHECKED(P_USER,C128,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)",
        "optional_setter": "<prefix>SET_MP_PS_CHECKED(P_USER,ERR_PRIMARY,ERR_ALTERNATE,OUTCOME,ADOPTED)",
        "types": {"P_USER": "REAL*8(0:3,NEXTERNAL)", "P and C128": "REAL*16(0:3,NEXTERNAL)",
                  "KEEP_OFFSHELL_MASS": "LOGICAL(NEXTERNAL)", "errors and OUTCOME": "INTEGER", "ADOPTED": "LOGICAL"},
        "outcomes": {"1": "primary method succeeded and copied its result", "2": "alternate method succeeded and copied its result",
                     "-1": "input rejected; error pair (1,0)", "-2": "both methods failed; raw errors retained",
                     "-3": "ImprovePSPoint>2 unsupported; no method called; error pair (0,0)"},
        "method_selection": "ImprovePSPoint=1: PSMC then ORIG. ImprovePSPoint=2 or <=0: ORIG then PSMC. Caller records the selected ImprovePSPoint value; OUTCOME identifies attempted/successful branches.",
        "ownership": "PREPARE stages locally and leaves C128 untouched on failure; it owns no COMMON state or readiness certificate. SET copies MP_PS/MP_P only on algorithm ADOPTED. Host must perform finite/IEEE/conversion/Born admission separately and must not infer C_READY from ADOPTED alone.",
        "warnings": "Checked sibling has its own saved WARNED/TOLD_SUPPRESS counter; legacy entry and its warning state are unchanged. Numerical helper warnings use the checked counter by reference. Mixed-entry warning histories, including warning arithmetic, need not coincide.",
        "ieee": "No flag/trap/rounding changes are introduced. Existing arithmetic and diagnostics can raise exceptions; enabled traps can prevent any return. Host owns exception-aware admission and preserves entry state.",
        "offshell": "KEEP_OFFSHELL_MASS declaration/pass is unchanged; runtime flag is not consulted by the legacy body. Generated masses_def controls off-shell behavior. The off-shell fixture proves generation preservation only, not new support.",
        "legacy": "Both original complete template files are byte-identical prefixes of the generated copies; no existing call site is rewired.",
    }, indent=2) + "\n"
    result["proof.json"] = json.dumps({
        "schema": 1, "legacy_mp_wrapper_sha256": digest(legacy.encode()),
        "checked_mp_wrapper_sha256": digest(checked.encode()),
        "annotation_stripped_wrapper_equals_legacy": True,
        "original_helpers_have_single_unmodified_copy": True,
        "fixture_names": sorted(fixtures),
        "fixture_limit": "Fixed-form source rendered from existing template substitutions; no model/reducer imports or compilation. Long template statements require the usual Fortran writer formatting or fixed-line-length-none for future compilation.",
    }, indent=2) + "\n"
    return {name: text.encode() for name, text in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(HERE) or destination == HERE:
        parser.error("Output must be inside the private checked-repair namespace")
    outputs = build_outputs()
    manifest = {"schema": 1, "generator_sha256": digest(Path(__file__).read_bytes()),
                "inputs_sha256": digest((HERE / "inputs/source.json").read_bytes()),
                "files": {name: digest(data) for name, data in sorted(outputs.items())}}
    outputs["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, data in outputs.items():
        path = destination / name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit("Generated artifact differs: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(("Verified" if args.check else "Generated") +
          " %d deterministic source artifacts; no native work." % len(outputs))


if __name__ == "__main__":
    main()
