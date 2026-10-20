#!/usr/bin/env python3
"""Apply the isolated guarded factor to proved QCD3 FFV call sites only."""
import sys
sys.dont_write_bytecode = True

from collections import Counter
import importlib.util
from pathlib import Path
import re

import factor

spec = importlib.util.spec_from_file_location("sealed_sharing_prototype",
    factor.PREVIOUS/"prototype.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def transform(directory, trace=True):
    directory = Path(directory)
    original = (directory/"polynomial.f").read_text()
    module = re.search(r"(?m)^      MODULE (\w+)POLYNOMIAL_CONSTANTS$",original)
    assert module
    prefix = module[1]
    gen = previous.FortranPolynomialRoutines(4,updater_max_rank=1,proc_prefix=prefix,
        sub_prefix="MP_"+prefix,coef_format="complex*32")
    pattern = re.compile(r"CALL(MP_\w*(UPDATE|TERMINAL)_WL_(\d+)_1)"
        r"\(WL\(1,0,1,(-?\d+)\),(\d+),COEFS,(\d+),(\d+),"
        r"WL\(1,0,1,(-?\d+)\)(,WL_ENTRY_INEXACT)?\)$")
    files, entries, helpers, vertices = {}, [], {}, {}
    for path in sorted(directory.glob("mp_coef_construction_*.f")):
        text = path.read_text()
        lines = text.splitlines(True)
        rows = previous.statements(text)
        vertex = None
        replacements = []
        for number,(index,label,row) in enumerate(rows):
            found = re.fullmatch(r"CALL(MP_\w+)\(.*?,COEFS\)",row)
            if found:
                vertex = found[1]
                continue
            if row == "CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)":
                continue
            match = pattern.fullmatch(row)
            if (match and 2<=int(match[3])<=4 and match[6]==match[7]=="4"
                    and vertex in ("MP_FFV1L1_2","MP_FFV1L2_1")):
                terminal = match[2]=="TERMINAL"
                assert bool(match[9]) == terminal
                assert int(match[4]) != int(match[8])
                assert 0<=int(match[5])<=4
                vertex_path = directory.parents[1]/"Source/DHELAS"/(vertex+".f")
                identity = factor.proof(vertex)
                assert previous.sha(vertex_path.read_text()) == identity["source_sha256"]
                vertices[vertex] = dict(source_sha256=identity["source_sha256"],
                    relative_source=str(vertex_path.relative_to(directory.parents[1])))
                suffix = "_FACTOR_"+vertex.removeprefix("MP_")
                new = match[1]+suffix
                assert len(new)<=63
                key = int(match[3]),terminal,vertex
                helpers[key] = match[1],new,suffix
                newrow = "CALL "+row[4:].replace(match[1],new,1)
                if not terminal:
                    newrow = newrow[:-1]+",WL_ENTRY_INEXACT)"
                    newrow = "CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n"+newrow
                rendered = previous.render(newrow).splitlines(True)
                rendered[0] = label[:5]+rendered[0][5:]
                # Consume only this logical statement, keeping all comments and
                # the next statement (including its label) unchanged.
                end = index+1
                while end<len(lines) and lines[end].strip() and lines[end][0] not in "cC*!":
                    if len(lines[end])<6 or lines[end][5] in " 0":
                        break
                    end += 1
                replacements.append((index,end,"".join(rendered)))
                entries.append(dict(file=path.name,line=index+1,label=label[:5],
                    vertex=vertex,rank=key[0],terminal=terminal,
                    input_wavefunction=int(match[4]),output_wavefunction=int(match[8]),
                    original=match[1],replacement=new,
                    added_entry_flag_query=not terminal))
            if row.startswith("CALL") or "COEFS" in row:
                vertex = None
        if replacements:
            # This targeted experiment uses existing terminal-aware batches;
            # it must not introduce a new IEEE procedure boundary implicitly.
            assert text.count("      USE, INTRINSIC :: IEEE_EXCEPTIONS")==1
            assert text.count("      LOGICAL WL_ENTRY_INEXACT")==1
            for start,end,newtext in reversed(replacements):
                lines[start:end] = [newtext]
            files[path.name] = "".join(lines)
    assert entries
    source = original
    bound_helpers = []
    count = len(helpers)
    for index,(key,(old,new,suffix)) in enumerate(sorted(helpers.items()),1):
        rank,terminal,vertex = key
        public = gen.write_compact_wl_updater(rank,1,terminal=terminal)
        if not terminal:
            public = public.split("\n\nSUBROUTINE "+old+"_DENSE(",1)[0]
        assert previous.render(public).strip() in original
        body = factor.generate(gen,rank,vertex,terminal,suffix)
        if trace:
            body = body.replace("  implicit none","  implicit none\n"
                f"  INTEGER*8 FACTOR_COUNTS(2,{count})\n"
                "  COMMON/FACTORIZATION_COUNTS/FACTOR_COUNTS",1)
            assert body.count("C FACTOR_FAST_PATH")==1
            body = body.replace("C FACTOR_FAST_PATH",
                f"C FACTOR_FAST_PATH\nFACTOR_COUNTS(1,{index})=FACTOR_COUNTS(1,{index})+1")
            body,hits = re.subn(
                r"(?m)^(\s*CALL MP_\w*(?:UPDATE|TERMINAL)_WL_\d+_1(?:_DENSE)?\(.*)$",
                rf"  FACTOR_COUNTS(2,{index})=FACTOR_COUNTS(2,{index})+1\n\1",body)
            assert hits>=5
        source += "\n"+previous.render(body)
        bound_helpers.append(dict(index=index,rank=rank,terminal=terminal,
            vertex=vertex,original=old,replacement=new))
    assert source.startswith(original)
    files["polynomial.f"] = source
    if trace:
        driver = (directory/"check_sa.f").read_text()
        assert driver.count("      IMPLICIT NONE")==1
        driver = driver.replace("      IMPLICIT NONE","      IMPLICIT NONE\n"
            f"      INTEGER*8 FACTOR_COUNTS(2,{count})\n"
            "      COMMON/FACTORIZATION_COUNTS/FACTOR_COUNTS\n"
            f"      DATA FACTOR_COUNTS/{2*count}*0/",1)
        marker = "      OPEN(6,FILE='/dev/null',STATUS='OLD')"
        assert driver.count(marker)==1
        driver = driver.replace(marker,marker+"\n      FACTOR_COUNTS=0",1)
        marker = "      ELAPSED=DBLE(T1-T0)/DBLE(RATE)"
        assert driver.count(marker)==1
        driver = driver.replace(marker,marker+
            "\n      OPEN(74,FILE='factor_trace.dat',STATUS='REPLACE')\n"
            f"      DO I=1,{count}\n"
            "         WRITE(74,'(3I20)') I,FACTOR_COUNTS(:,I)\n"
            "      ENDDO\n      CLOSE(74)",1)
        files["check_sa.f"] = driver
    proof = dict(trace=trace,helpers=bound_helpers,calls=entries,vertices=vertices,
        call_counts=dict(Counter(("terminal" if e["terminal"] else "full")+
                               "-rank"+str(e["rank"]) for e in entries)),
        symbolic_proof_sha256=factor.PROOF_SHA256,
        original_polynomial_sha256=previous.sha(original),
        original_polynomial_preserved_as_prefix=True,
        generator_sha256=previous.sha(
            (factor.gather.ROOT/"madgraph/various/q_polynomial.py").read_text()),
        scope="Private isolated QP derivative factor, changed association and "
              "zero-sign contract. Actual vertex source hashes and original "
              "public routine renderings checked; distinct input/output WL; "
              "existing batch IEEE boundaries preserved. Every added full "
              "private call receives a newly queried entry-INEXACT flag.")
    return files,proof
