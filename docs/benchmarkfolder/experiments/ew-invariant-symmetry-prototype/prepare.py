#!/usr/bin/env python3
"""Derive a private Fortran proposal; default verifies, never compiles or runs."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/prepared/ew_n3_epem_ttxa/TIRUST_interface.f.original"
SOURCE_SHA = "e60570ec599263807165f2f289add2076a4c5b71b9e09fe0de83306e9ab2f7ff"
INSERTION = """C     Private proposal: the ABI consumes the upper triangle (I<J).
C     Keep all original reconstruction arithmetic and the diagonal.
C     Do not hide any nonfinite lower-triangle input from validation.
      IF(ALL(IEEE_IS_FINITE(INVARIANTS))) THEN
        DO J=2,NLOOPLINE
          DO I=1,J-1
            IF(INVARIANTS(J,I).NE.INVARIANTS(I,J)) THEN
              INVARIANTS(J,I)=INVARIANTS(I,J)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
"""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def derive():
    original = SOURCE.read_bytes()
    if sha(original) != SOURCE_SHA:
        raise ValueError("Frozen generated adapter changed")
    text = original.decode("ascii")
    for kind, scale in (("KI_TIR", "MU_R**2"), ("KI_QTIR", "REAL(MU_R,KIND=KI_QTIR)**2")):
        anchor = ("          INVARIANTS(I,J)=REAL(S_MAT(I,J)+MASSES(I)+MASSES(J)\n"
                  "     $     ,KIND=" + kind + ")\n"
                  "        ENDDO\n      ENDDO\n      SCALE=" + scale + "\n")
        if text.count(anchor) != 1:
            raise ValueError("Expected exactly one original " + kind + " reconstruction")
        text = text.replace(anchor, anchor.replace("      SCALE=", INSERTION + "      SCALE="))
    if text.count(INSERTION) != 2 or text.replace(INSERTION, "").encode("ascii") != original:
        raise ValueError("Proposal is not exactly two insertion-only mirror blocks")
    if any(len(line) > 132 for line in text.splitlines()):
        raise ValueError("Fixed-form line exceeds original compiler allowance")
    candidate = text.encode("ascii")
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), text.splitlines(True),
            fromfile="frozen/TIRUST_interface.f", tofile="private/TIRUST_interface.f")).encode("ascii")
    return original, candidate, patch


def run(prepare):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from MadGraph repository")
    original, candidate, patch = derive()
    records = {"TIRUST_interface.f": candidate, "derivation.patch": patch}
    result = {"schema": "ew-invariant-canonical-triangle-source-proposal-v1",
              "source": str(SOURCE.relative_to(ROOT)), "source_sha256": SOURCE_SHA,
              "candidate_sha256": sha(candidate), "patch_sha256": sha(patch),
              "insertion_blocks": 2, "strip_insertions_byte_identical": True,
              "upper_triangle_and_diagonal_arithmetic_unchanged": True,
              "all_original_reconstruction_arithmetic_retained": True,
              "mirror_only_if_whole_matrix_finite": True,
              "already_equal_pairs_and_signed_zeros_retained": True,
              "shared_kinematic_builder_modified": False, "native_bridge_modified": False,
              "compiled": False, "native_executed": False, "numerical_parity_qualified": False,
              "performance_eligible": False,
              "limitations": ["Source-derived proposal only; no captured physical failing invariant matrix.",
                              "IEEE flag/trap behavior of the added finite predicate needs compiled qualification.",
                              "Formerly rejected asymmetric inputs require independent MadLoop stability and reference accuracy checks."]}
    receipt = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    records["preparation.json"] = receipt
    for name, data in records.items():
        target = HERE / name
        if prepare:
            with target.open("xb") as out:
                out.write(data)
        elif target.read_bytes() != data:
            raise ValueError("Private proposal differs from deterministic source derivation: " + name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Write proposal sources once; no compiler/native call")
    run(parser.parse_args().prepare)
