#!/usr/bin/env python3
"""Isolated private terminal updaters; never replace the public full updater."""
from pathlib import Path
import hashlib
import re
import tempfile

from prototype import logical_lines

BOUNDED = """ELEMENTAL LOGICAL FUNCTION WL_BOUNDED(X)
  REAL*16, INTENT(IN) :: X
  INTEGER*8, PARAMETER :: BOUND_ONE(2)=TRANSFER(1.0_16,(/0_8,0_8/))
  INTEGER, PARAMETER :: BOUND_HI=MERGE(1,2,BOUND_ONE(1).NE.0_8)
  WL_BOUNDED=.FALSE.
  IF(STORAGE_SIZE(X).NE.128.OR.DIGITS(X).NE.113.OR.RADIX(X).NE.2.OR.MAXEXPONENT(X).NE.16384) RETURN
  IF(BOUND_ONE(BOUND_HI).NE.ISHFT(16383_8,48).OR.BOUND_ONE(3-BOUND_HI).NE.0_8) RETURN
  ASSOCIATE(WORDS=>TRANSFER(X,BOUND_ONE))
    IF(IBCLR(WORDS(BOUND_HI),63).EQ.0_8.AND.WORDS(3-BOUND_HI).EQ.0_8) THEN
      WL_BOUNDED=.TRUE.
    ELSE
      WL_BOUNDED=IBITS(WORDS(BOUND_HI),48,15).GE.15383_8.AND.IBITS(WORDS(BOUND_HI),48,15).LE.17383_8
    ENDIF
  END ASSOCIATE
END FUNCTION
"""


def current_classifier(polynomial):
    """Apply the committed storage fix equally to both isolated variants."""
    def replace(match):
        source = match.group()
        if "ASSOCIATE(WORDS=>TRANSFER(X,ONE_WORDS))" in source:
            return source
        old = "      INTEGER*8 WORDS(2)\n"
        assignment = ("      WORDS=TRANSFER(X,WORDS)\n"
                      "      WL_IS_FINITE=IBITS(WORDS(HI),48,15).NE.32767_8\n")
        if source.count(old) != 1 or source.count(assignment) != 1:
            raise RuntimeError("Unknown finite classifier before storage compatibility fix")
        return source.replace(old, "").replace(assignment,
            "      ASSOCIATE(WORDS=>TRANSFER(X,ONE_WORDS))\n"
            "      WL_IS_FINITE=IBITS(WORDS(HI),48,15).NE.32767_8\n"
            "      END ASSOCIATE\n")
    return re.sub(r"(?m)^      ELEMENTAL LOGICAL FUNCTION WL_IS_FINITE\(X\)"
                  r".*?^      END FUNCTION", replace, polynomial, flags=re.S | re.M)


def helper(source):
    """Input is the logical source of one existing sparse QP routine."""
    from madgraph.various.q_polynomial import get_number_of_coefs_for_rank
    name, a, b = re.search(
        r"SUBROUTINE (MP_\w+UPDATE_WL_(\d+)_(\d+))\(", source).groups()
    na, nb = (get_number_of_coefs_for_rank(int(r))-1 for r in (a, b))
    private = name.replace("UPDATE_WL_", "TERMINAL_WL_")
    body, contained = source.split("CONTAINS", 1)
    body = body.replace("SUBROUTINE "+name+"(", "SUBROUTINE "+private+"(", 1)
    body = body.replace(private+"(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)",
                        private+"(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT,WL_ENTRY_INEXACT)",1)
    body = body.replace("USE, INTRINSIC :: IEEE_ARITHMETIC",
                        "USE, INTRINSIC :: IEEE_ARITHMETIC\n"
                        "USE, INTRINSIC :: IEEE_EXCEPTIONS", 1)
    body = body.replace("LOGICAL WL_NEAREST",
                        "LOGICAL WL_NEAREST\nLOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT", 1)
    guard = f"""WL_ALREADY=WL_ENTRY_INEXACT
WL_HALT=.TRUE.
IF(IEEE_SUPPORT_HALTING(IEEE_INEXACT)) CALL IEEE_GET_HALTING_MODE(IEEE_INEXACT,WL_HALT)
IF(.NOT.WL_NEAREST.OR..NOT.WL_ALREADY.OR.WL_HALT.OR.LCUT_SIZE.GT.OUT_SIZE.OR.MAXLWFSIZE.GT.16.OR.{na+1}.GT.65536.OR.{nb+1}.GT.65536.OR..NOT.ALL(WL_BOUNDED(REAL(A(1:IN_SIZE,0:{na},1:LCUT_SIZE),KIND=16))).OR..NOT.ALL(WL_BOUNDED(AIMAG(A(1:IN_SIZE,0:{na},1:LCUT_SIZE)))).OR..NOT.ALL(WL_BOUNDED(REAL(B(1:OUT_SIZE,0:{nb},1:IN_SIZE),KIND=16))).OR..NOT.ALL(WL_BOUNDED(AIMAG(B(1:OUT_SIZE,0:{nb},1:IN_SIZE))))) THEN
  CALL {name}(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)
  RETURN
ENDIF
"""
    body, count = re.subn(r"IF\(\.NOT\.WL_NEAREST\.OR\..*?\n\s*ENDIF\n",
                         guard, body, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("Original finite-path guard not found")
    if "UPDATER_COEF" in body:
        before, arithmetic = body.split("OUT(:,:,:)=", 1)
        arithmetic, count = re.subn(r"DO I=1,LCUT_SIZE", "DO I=J,MIN(J,LCUT_SIZE)",
                                   arithmetic, count=1)
        if count != 1:
            raise RuntimeError("Compact diagonal loop not found")
        body = before+"OUT(:,:,:)="+arithmetic
    else:
        # Keep clearing identical. Only the arithmetic loop's J range changes.
        body, count = re.subn(r"(DO K=1,IN_SIZE\n\s*)DO J=1,OUT_SIZE",
                             r"\1DO J=I,I", body, count=1)
        if count != 1:
            raise RuntimeError("Expanded diagonal loop not found")
    contained = contained.replace("END FUNCTION\n", "END FUNCTION\n"+BOUNDED, 1)
    return body+"CONTAINS"+contained


def fixed(source):
    from madgraph.iolibs.file_writers import FortranWriter
    with tempfile.TemporaryDirectory(prefix="qp-terminal-") as temp:
        output = Path(temp)/"helper.f"
        with FortranWriter(str(output)) as writer:
            writer.writelines(source)
        return output.read_text()


def replace_fixed_calls(source, replacements):
    """Keep original fixed-form labels and all unrelated statement bytes."""
    lines = source.splitlines(keepends=True)
    output, matched = [], set()
    index = 0
    while index < len(lines):
        end = index+1
        while (end < len(lines) and len(lines[end]) > 6 and
               lines[end][0] not in "cC*!" and lines[end][5] not in " 0"):
            end += 1
        original = "".join(lines[index:end])
        statement = logical_lines(original).strip()
        if statement in replacements:
            if lines[index][:5].strip():
                raise RuntimeError("A redirected terminal call has a statement label")
            if statement in matched:
                raise RuntimeError("A terminal call occurs more than once")
            matched.add(statement)
            output.append(fixed(replacements[statement]))
        else:
            output.append(original)
        index = end
    if matched != set(replacements):
        raise RuntimeError("A proven terminal call was not found in fixed-form source")
    return "".join(output)


def transform(directory):
    """Prove the whole generated WL consumer graph before changing any call."""
    files = {}
    # Analyze every generated source that can mention this COMMON-backed WL.
    sources = sorted(directory.glob("*.f"))
    all_text = {p.name: logical_lines(p.read_text()) for p in sources}
    batch_names = sorted((name for name in all_text
                          if re.fullmatch(r"mp_coef_construction_\d+\.f", name)),
                         key=lambda name: int(re.search(r"(\d+)\.f$", name)[1]))
    if not batch_names:
        raise RuntimeError("No generated multiple-precision coefficient batches")
    calls, roots, producers = [], [], {}
    update = re.compile(
        r"CALL (MP_\w+UPDATE_WL_(\d+)_(\d+))\(WL\(1,0,1,(-?\d+)\),"
        r"(\d+),COEFS,(\d+),(\d+),WL\(1,0,1,(-?\d+)\)\)")
    create = re.compile(r"CALL MP_\w+CREATE_LOOP_COEFS\(WL\(1,0,1,(-?\d+)\),")
    for name in batch_names:
        text = all_text[name]
        for match in update.finditer(text):
            routine, a, b, origin, cut, ins, outs, destination = match.groups()
            item = dict(routine=routine, left=int(a), right=int(b), origin=int(origin),
                        cut=int(cut), incoming=int(ins), outgoing=int(outs),
                        destination=int(destination), file=name, statement=match.group())
            if item["destination"] in producers or item["origin"] == item["destination"]:
                raise RuntimeError("Nonunique or aliased generated wavefunction output")
            if item["origin"] not in producers and item["origin"] not in (-1, 0):
                raise RuntimeError("Updater reads an unproven predecessor")
            producers[item["destination"]] = item
            calls.append(item)
        for match in create.finditer(text):
            roots.append(int(match[1]))
        # Every executable use of WL must be a recognized updater or CREATE call.
        for line in text.splitlines():
            if (re.search(r"\bWL\(", line) and
                    not re.match(r"^[cC*!](?:\s|$)", line) and
                    not line.startswith(("COMPLEX", "COMMON"))):
                if not (update.fullmatch(line) or create.match(line)):
                    raise RuntimeError(f"Unclassified generated wavefunction consumer: {name}: {line}")
    terminal = set(roots) - {item["origin"] for item in calls}
    if not terminal or not terminal.issubset(producers):
        raise RuntimeError("Missing/unknown terminal producer")
    # Other routines legitimately initialize/read COMMON arrays. They must never
    # reference a concrete positive terminal node, nor pass the whole WL elsewhere.
    for name, text in all_text.items():
        if name in batch_names:
            continue
        # DP batches have an independent COMMON block with similar node numbers.
        if not re.search(r"COMMON/\w+_MP_WL/WL,PL", text):
            continue
        if re.search(r"\bCALL\b[^\n]*\bWL\b", text, re.I):
            # polynomial helpers take their local WL argument; coefficient caller
            # merely calls the numbered batches without WL as an argument.
            if name != "polynomial.f":
                raise RuntimeError(f"Unclassified cross-routine WL call: {name}")
        for line in text.splitlines():
            if re.match(r"^[cC*!](?:\s|$)", line) or line.startswith(("COMPLEX", "COMMON")):
                continue
            for use in re.finditer(r"\bWL\(([^)\n]*)\)", line):
                subscripts = use[1].split(",")
                if len(subscripts) != 4 or subscripts[-1].strip() not in ("-1", "0"):
                    raise RuntimeError(f"Unclassified COMMON wavefunction access: {name}: {line}")
    polynomial = (directory/"polynomial.f").read_text()
    merge = re.search(r"(?m)^      SUBROUTINE MP_\w+MERGE_WL\(.*?^      END\s*$",
                      polynomial, flags=re.S|re.M)
    create_body = re.search(r"(?m)^      SUBROUTINE MP_\w+CREATE_LOOP_COEFS\(.*?^      END\s*$",
                            polynomial, flags=re.S|re.M)
    if (not merge or "OUT(J)=OUT(J)+WL(I,J,I)*CONST" not in logical_lines(merge.group()) or
            len(re.findall(r"\bWL\(", logical_lines(merge.group()))) != 2 or
            not create_body or len(re.findall(r"\bLOOP_WF\b", create_body.group())) != 3):
        raise RuntimeError("The merge consumer is not the expected diagonal-only trace")
    helpers, selected = {}, []
    for item in calls:
        if item["destination"] not in terminal:
            continue
        if item["cut"] > item["outgoing"]:
            raise RuntimeError("Terminal merge diagonal exceeds output dimension")
        routine = item["routine"]
        if routine not in helpers:
            pattern = re.compile(r"(?m)^      SUBROUTINE "+routine+r"\(.*?^      END\s*$",
                                 re.S|re.M)
            match = pattern.search(polynomial)
            if not match or "WL_NEAREST" not in match.group():
                raise RuntimeError("Expected frozen sparse QP routine missing")
            helpers[routine] = fixed(helper(logical_lines(match.group())))
        selected.append(item)
    for name in batch_names:
        text = (directory/name).read_text()
        replacements = {}
        if any(item["file"] == name for item in selected):
            if text.count("      IMPLICIT NONE\n") != 1:
                raise RuntimeError("Unexpected coefficient batch declaration")
            text = text.replace("      IMPLICIT NONE\n",
                                "      USE, INTRINSIC :: IEEE_EXCEPTIONS\n"
                                "      IMPLICIT NONE\n"
                                "      LOGICAL WL_ENTRY_INEXACT\n",1)
        for item in selected:
            if item["file"] == name:
                replacement = item["statement"].replace("UPDATE_WL_", "TERMINAL_WL_", 1)
                replacement = replacement[:-1]+",WL_ENTRY_INEXACT)"
                replacements[item["statement"]] = (
                    "CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n"+replacement)
        files[name] = replace_fixed_calls(text, replacements)
    files["polynomial.f"] = polynomial+"\n"+"\n".join(helpers.values())
    evidence = {"terminal_calls": selected, "terminal_roots": sorted(terminal),
                "all_update_calls": calls, "merge_roots": roots,
                "private_helpers": sorted(helpers),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                "contract": "Only diagonal values are consumed; off-diagonal terminal "
                            "scratch values are intentionally not materialized. Public "
                            "updaters and all nonterminal calls remain unchanged."}
    return files, evidence
