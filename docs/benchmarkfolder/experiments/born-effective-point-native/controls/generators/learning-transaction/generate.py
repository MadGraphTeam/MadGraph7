#!/usr/bin/env python3
"""Isolated source generator for checked original-Born learning transactions.

Import transform(source, split) from the supported private exporter. This is
opt-in: no production exporter or ordinary tree template is changed here.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REVISION = 2
PREFIX = "%(proc_prefix)s"
NAMES = {False: "matrix_standalone_v4.inc",
         True: "matrix_standalone_splitOrders_v4.inc"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Expected unique source anchor: " + repr(old))
    return source.replace(old, new, 1)


def block(name, text):
    return "C BORN-TRANSACTION-BEGIN " + name + "\n" + text + "C BORN-TRANSACTION-END " + name + "\n"


def strip_additions(source):
    return re.sub(r"(?m)^C BORN-TRANSACTION-BEGIN ([A-Z_]+)\n.*?"
                  r"^C BORN-TRANSACTION-END \1\n", "", source, flags=re.S)


def extract_aggregate(source, split):
    name = "SMATRIX_SPLITORDERS" if split else "SMATRIX"
    header = "      SUBROUTINE " + PREFIX + name + "(P, FLAV_IDX, ANS)\n"
    if source.count(header) != 1:
        raise ValueError("Expected one original Born aggregate")
    first = source.index(header)
    end = re.search(r"(?m)^      END\s*\n", source[first:])
    if end is None:
        raise ValueError("Missing aggregate END")
    return source[first:first+end.end()]


def guard(name, expression):
    return block(name, "      IF (BT_CHECKED_CALL) THEN\n"
                 "        IF (.NOT.(" + expression + ")) GOTO 9020\n"
                 "      ENDIF\n")


def wrapper(split):
    suffix = "_SPLITORDERS" if split else ""
    dimensions = "(0:NSQAMPSO)" if split else ""
    extra = ("      INTEGER NSQAMPSO\n"
             "      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)\n") if split else ""
    result = ("      SUBROUTINE " + PREFIX + "SMATRIXHEL" + suffix +
              "_CHECKED(P,HEL,FLAV_IDX,ANS,OK)\n"
              "      IMPLICIT NONE\n"
              "      INTEGER NEXTERNAL\n"
              "      PARAMETER (NEXTERNAL=%(nexternal)d)\n" + extra +
              "      REAL*8 P(0:3,NEXTERNAL),ANS" + dimensions + "\n"
              "      INTEGER HEL,FLAV_IDX,USERHEL,BT_USERHEL_SAVED\n"
              "      LOGICAL OK\n"
              "      COMMON/" + PREFIX + "HELUSERCHOICE/USERHEL\n"
              "      BT_USERHEL_SAVED=USERHEL\n"
              "      USERHEL=HEL\n"
              "      CALL " + PREFIX + "SMATRIX" + suffix +
              "_CHECKED(P,FLAV_IDX,ANS,OK)\n"
              "      IF (OK) THEN\n"
              "        USERHEL=-1\n"
              "      ELSE\n"
              "        USERHEL=BT_USERHEL_SAVED\n"
              "      ENDIF\n"
              "      END\n")
    if split:
        result += ("\n      SUBROUTINE " + PREFIX +
                   "SMATRIXHEL_CHECKED(P,HEL,FLAV_IDX,ANS,OK)\n"
                   "      IMPLICIT NONE\n"
                   "      INTEGER NEXTERNAL,NSQAMPSO\n"
                   "      PARAMETER (NEXTERNAL=%(nexternal)d)\n"
                   "      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)\n"
                   "      REAL*8 P(0:3,NEXTERNAL),ANS,ANS_SPLIT(0:NSQAMPSO)\n"
                   "      INTEGER HEL,FLAV_IDX\n"
                   "      LOGICAL OK\n"
                   "      CALL " + PREFIX + "SMATRIXHEL_SPLITORDERS_CHECKED(\n"
                   "     $ P,HEL,FLAV_IDX,ANS_SPLIT,OK)\n"
                   "      ANS=ANS_SPLIT(0)\n"
                   "      END\n")
    else:
        result += ("\n      SUBROUTINE " + PREFIX +
                   "SMATRIXHEL_SPLITORDERS_CHECKED(P,HEL,FLAV_IDX,ANS,OK)\n"
                   "      IMPLICIT NONE\n"
                   "      INTEGER NEXTERNAL\n"
                   "      PARAMETER (NEXTERNAL=%(nexternal)d)\n"
                   "      REAL*8 P(0:3,NEXTERNAL),ANS(0:1),ANS_SCALAR\n"
                   "      INTEGER HEL,FLAV_IDX\n"
                   "      LOGICAL OK\n"
                   "      CALL " + PREFIX + "SMATRIXHEL_CHECKED(\n"
                   "     $ P,HEL,FLAV_IDX,ANS_SCALAR,OK)\n"
                   "      ANS(0)=ANS_SCALAR\n"
                   "      ANS(1)=ANS_SCALAR\n"
                   "      END\n")
    return result


def transform(source, split):
    """Add a same-owner ENTRY and checked wrappers; all old text is retained.

    Strip the marked additions and recover the exact original source bytes.
    Fail closed on template drift, extra returns, labels or name collisions.
    """
    if "BT_" in source or "BORN-TRANSACTION" in source:
        raise ValueError("Reserved transaction names already present")
    old = extract_aggregate(source, split)
    if len(re.findall(r"(?mi)^\s*RETURN\s*$", old)) != 1:
        raise ValueError("Re-audit all aggregate returns before transformation")
    if re.search(r"(?m)^\s*90(?:01|10|20)\s", old):
        raise ValueError("Reserved transaction labels already present")
    new = once(old, "      IMPLICIT NONE\n", block("IEEE_IMPORT",
        "      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE,\n"
        "     $ IEEE_VALUE,IEEE_QUIET_NAN\n") + "      IMPLICIT NONE\n")
    name = "SMATRIX_SPLITORDERS" if split else "SMATRIX"
    # This insertion is after every old declaration and before the first old
    # executable statement. The checked ENTRY shares DATA learning with SMATRIX.
    anchor = "c----------\nc   Check if helreset mode is on\nc---------\n"
    new = once(new, anchor, block("ENTRY_AND_SNAPSHOT",
        "      LOGICAL BT_CHECKED_CALL,BT_OK,BT_HELRESET_SAVED\n"
        "      INTEGER BT_NTRY_SAVED(NFLAV)\n"
        "      LOGICAL BT_GOODHEL_SAVED(NCOMB,NFLAV)\n"
        "      BT_CHECKED_CALL=.FALSE.\n"
        "      GOTO 9001\n"
        "      ENTRY " + PREFIX + name + "_CHECKED(P,FLAV_IDX,ANS,BT_OK)\n"
        "      BT_CHECKED_CALL=.TRUE.\n"
        "      BT_OK=.FALSE.\n"
        "      BT_NTRY_SAVED=NTRY\n"
        "      BT_GOODHEL_SAVED=GOODHEL\n"
        "      BT_HELRESET_SAVED=HELRESET\n"
        "9001 CONTINUE\n") + anchor)
    new = once(new, "        RETURN\n", block("EARLY_RETURN",
        "      IF (BT_CHECKED_CALL) GOTO 9010\n") + "        RETURN\n")
    if split:
        call = "              CALL " + PREFIX + "MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX, T)\n"
        new = once(new, call, call + guard("RAW_T", "ALL(IEEE_IS_FINITE(T))"))
        new = once(new, "              BUFF=0D0\n",
                   guard("WEIGHTED_T", "ALL(IEEE_IS_FINITE(T))") + "              BUFF=0D0\n")
        learn = "              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN\n"
        new = once(new, learn,
                   guard("LEARNING_BUFFER", "IEEE_IS_FINITE(BUFF)") +
                   guard("PARTIAL_SUM", "ALL(IEEE_IS_FINITE(ANS))") + learn)
    else:
        call = "                    T=" + PREFIX + "MATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX)\n"
        new = once(new, call, call + guard("RAW_T", "IEEE_IS_FINITE(T)"))
        selected = "                    IF(POLARIZATIONS(0,0).eq.-1.or." + PREFIX + "IS_BORN_HEL_SELECTED(IHEL)) THEN\n"
        new = once(new, selected, guard("WEIGHTED_T", "IEEE_IS_FINITE(T)") + selected)
        learn = "                    IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN\n"
        new = once(new, learn, guard("PARTIAL_SUM", "IEEE_IS_FINITE(ANS)") + learn)
    finite = "ALL(IEEE_IS_FINITE(ANS))" if split else "IEEE_IS_FINITE(ANS)"
    new = once(new, "      END\n", block("FINALIZE",
        "9010 CONTINUE\n"
        "      IF (BT_CHECKED_CALL) THEN\n"
        "        IF (.NOT.(" + finite + ")) GOTO 9020\n"
        "        BT_OK=.TRUE.\n"
        "      ENDIF\n"
        "      RETURN\n"
        "9020 CONTINUE\n"
        "      NTRY=BT_NTRY_SAVED\n"
        "      GOODHEL=BT_GOODHEL_SAVED\n"
        "      HELRESET=BT_HELRESET_SAVED\n"
        "      BT_OK=.FALSE.\n"
        "      ANS=IEEE_VALUE(0D0,IEEE_QUIET_NAN)\n"
        "      RETURN\n") + "      END\n")
    result = once(source, old, new) + block("CHECKED_WRAPPERS", "\n" + wrapper(split))
    if strip_additions(result) != source:
        raise AssertionError("Original numerical/selection statements changed")
    return result


def source_inputs():
    meta = json.loads((HERE/"inputs/source.json").read_text())
    sources = {}
    for record in meta["sources"]:
        data = (HERE/record["copy"]).read_bytes()
        if digest(data) != record["sha256"]:
            raise ValueError("Frozen input changed: " + record["copy"])
        sources[Path(record["path"]).name] = data.decode()
    return meta, sources


def parameters(nexternal=4, split=False):
    """Only the aggregate is rendered; no synthetic HELAS/model body is claimed."""
    return dict(proc_prefix="TX_", nexternal=nexternal, ncomb=4,
                nincoming=2, nflav=2, nSqAmpSplitOrders=3, hel_avg_factor=4,
                beamone_helavgfactor=2, beamtwo_helavgfactor=2,
                chosen_so_configs=".TRUE.,.FALSE.,.TRUE.", info_lines="",
                process_lines="C Pure source/control fixture, not a model.",
                den_factor_line="      DATA IDEN/4/", helicity_lines="")


def build_outputs():
    meta, sources = source_inputs()
    result = {}
    patch = []
    for split, name in NAMES.items():
        old = sources[name]
        new = transform(old, split)
        result["generated/"+name] = new
        result["generated/"+name+".aggregate"] = extract_aggregate(new, split)
        rel = next(r["path"] for r in meta["sources"] if Path(r["path"]).name == name)
        patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile="a/"+rel, tofile="b/"+rel))
        for n in [3, 5]:
            p = parameters(n, split)
            kind = "split" if split else "scalar"
            result["rendered/"+kind+"_n"+str(n)+".f"] = (
                extract_aggregate(new, split) + "\n" + wrapper(split)) % p
    result["patches/learning-transaction.patch"] = "".join(patch)
    result["api.json"] = json.dumps({
        "schema": 1, "revision": REVISION, "native_qualified": False,
        "split_host": "<prefix>SMATRIXHEL_SPLITORDERS_CHECKED(P,HEL,FLAV_IDX,ANS,OK)",
        "scalar_host": "<prefix>SMATRIXHEL_CHECKED(P,HEL,FLAV_IDX,ANS,OK)",
        "split_entry": "<prefix>SMATRIX_SPLITORDERS_CHECKED(P,FLAV_IDX,ANS,OK)",
        "scalar_entry": "<prefix>SMATRIX_CHECKED(P,FLAV_IDX,ANS,OK)",
        "types": {"P": "REAL*8(0:3,NEXTERNAL)", "ANS split": "REAL*8(0:NSQAMPSO)",
                  "ANS scalar": "REAL*8", "HEL and FLAV_IDX": "INTEGER", "OK": "LOGICAL"},
        "no_split_companion": "SMATRIXHEL_SPLITORDERS_CHECKED is also emitted for no-split output; ANS(0:1) receives the checked scalar result in both slots, including quiet NaN on failure; OK propagates unchanged.",
        "opt_in": "Only supported private MadLoop Born output selects transform; ordinary tree exports and canonical templates are untouched.",
        "shared_owner": "Checked ENTRY shares the existing aggregate's DATA NTRY/GOODHEL; both entries reset BT_CHECKED_CALL explicitly. No duplicate learning owner or COMMON initialization.",
        "success": "OK=true; all original arithmetic statements and successful learning mutations retained in order. Wrapper retains original USERHEL=-1 reset, including invalid-flavor zero.",
        "rejection": "OK=false; restore entire NTRY/GOODHEL and HELRESET snapshot taken before any reset; wrapper restores its entry USERHEL; output is quiet NaN. Reject raw/weighted T, split BUFF, every partial/final output slot if nonfinite.",
        "ieee": "Classify finite values without clearing flags or changing traps/rounding. Enabled traps can terminate before rollback/return. Quiet NaN output is a failure marker, not a physics result. Host owns entry-flag admission and exception recovery.",
        "limits": "Not a rollback of arbitrary MATRIX scratch/amp2/color-flow effects. Scope requires supported scalar standalone Born; no concurrency/reentrancy guarantee beyond original saved/COMMON state. Finite guards add unmeasured overhead; compiler/IEEE/bit-identity qualification remains open.",
    }, indent=2)+"\n"
    result["proof.json"] = json.dumps({"schema": 1, "revision": REVISION,
        "original_full_sources_recovered_by_removing_only_marked_additions": True,
        "old_return_count_per_aggregate": 1,
        "same_saved_learning_owner": True,
        "fixture_limit": "Aggregate/wrapper template substitution only; no actual graph, compiler, numerical integrand or IEEE hardware execution. Existing long statements need normal Fortran writer formatting or fixed-line-length-none.",
        "source_hashes": {name:digest(text.encode()) for name,text in sources.items()}},indent=2)+"\n"
    result["revision.json"] = json.dumps({"schema":1,"revision":REVISION,
        "change":"Use collision-checked four-digit fixed-form labels 9001, 9010, 9020; the current FortranWriter places a five-digit label into continuation column six.",
        "previous_manifest":"history/revision-1/manifest.json",
        "previous_manifest_sha256":digest((HERE/"history/revision-1/manifest.json").read_bytes()),
        "previous_sources_preserved":["generate.py","test_generate.py","README.md","tests.json"],
        "formatter_metadata":"inputs/formatter.json",
        "formatter_metadata_sha256":digest((HERE/"inputs/formatter.json").read_bytes()),
        "native_qualified":False},indent=2)+"\n"
    return {name:text.encode() for name,text in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"artifacts")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    dest = args.output.resolve()
    if not dest.is_relative_to(HERE) or dest == HERE:
        parser.error("Output must remain inside private learning-transaction")
    outputs = build_outputs()
    outputs["manifest.json"] = (json.dumps({"schema":1,"revision":REVISION,
        "generator_sha256":digest(Path(__file__).read_bytes()),
        "inputs_sha256":digest((HERE/"inputs/source.json").read_bytes()),
        "files":{name:digest(data) for name,data in sorted(outputs.items())}},indent=2)+"\n").encode()
    for name,data in outputs.items():
        path=dest/name
        if args.check:
            if not path.is_file() or path.read_bytes()!=data:
                raise SystemExit("Artifact differs: "+str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
    print(("Verified" if args.check else "Generated")+" %d source artifacts; no native work." % len(outputs))


if __name__ == "__main__":
    main()
