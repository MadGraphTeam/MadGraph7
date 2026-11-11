"""Pure source-contract tests; no Fortran compilation, generation or execution.

Qualification limits: the explicitly approved diagnostic driver policy uses
FORCE_STABILITY_CHECK(.FALSE.), five ordinary bootstrap public calls, and at
most one measurement call. Natural initialization may still be incomplete;
the driver then retains all evidence and stops without retry or measurement.
INIT state, status and all raw outputs are evidence, not accuracy or trace
qualification. The caller must retain negative/unknown codes, verify actual
modes in STATE and trace records, and review initialization DATA/layout in each
current generated matrix. Positive fixed-mode codes can be unchecked and may
hide failed tensor contributions. No native result, backend success, parity or
fully initialized run is claimed by these source-only tests.

Run from repository root with the pinned environment and Python -B.
"""
from pathlib import Path
import re
import unittest

SOURCE = Path(__file__).with_name("driver.f.in")


def render(prefix="ML5_0_", nexternal=5, nborn=1, instrumented=True):
    text = SOURCE.read_text()
    control = (f"      CALL {prefix}PRIVATE_KTRACE_SET(LUN,POINT,ENABLED)"
               if instrumented else "      CONTINUE")
    for key, value in (("@PREFIX@", prefix), ("@NEXTERNAL@", nexternal),
                       ("@NBORN@", nborn), ("@CONTROL_CALL@", control)):
        text = text.replace(key, str(value))
    return text


def statements(text):
    """Read only these fixed-form private sources; reject overlong code."""
    out = []
    for line in text.splitlines():
        if not line or line[0] in "Cc*!":
            continue
        if len(line) > 72:
            raise ValueError("fixed-form statement exceeds column 72")
        if len(line) < 6:
            raise ValueError("malformed fixed-form statement")
        if line[5] not in " 0":
            if not out:
                raise ValueError("orphan continuation")
            out[-1] += line[6:].strip()
        else:
            out.append(line[6:].strip())
    return out


def check_contract(text):
    code = statements(text)
    joined = "\n".join(code)
    if re.search(r"@[A-Z_]+@", text):
        raise ValueError("unrendered token")
    calls = [s for s in code if re.match(r"CALL \w+SLOOPMATRIX_THRES\(", s)]
    if len(calls) != 2:
        raise ValueError("public call budget")
    if not all("SLOOPMATRIX_THRES(P," in s for s in calls):
        raise ValueError("working-point argument missing")
    if code.count("DO BOOT_CALL=1,5") != 1:
        raise ValueError("five-entry bootstrap budget missing")
    if "IF(DID_RUN.OR.SEQUENCE.NE.1.OR.NLOOPS.NE.1) GOTO 800" not in code:
        raise ValueError("one RUN budget missing")
    first, second = [i for i, s in enumerate(code) if s in calls]
    loop_start = code.index("DO BOOT_CALL=1,5")
    if code[loop_start + 1] != "P=BOOT":
        raise ValueError("bootstrap does not reload saved point")
    if code[first:second].count("P=POINT") != 1:
        raise ValueError("measurement does not reload saved point")
    if "DID_RUN=.TRUE." not in code[first:second]:
        raise ValueError("run latch missing")
    if sum(s == "CODE=-987654321" for s in code) != 2:
        raise ValueError("missing return-code sentinel")
    if sum(s == "ACC=-1D0" for s in code) != 2:
        raise ValueError("missing accuracy sentinel")
    if "FORCE_STABILITY_CHECK(.FALSE.)" not in joined:
        raise ValueError("diagnostic stability policy changed")
    for pattern in (r"^(?:CTMODE|CTMODERUN|CTMODEINIT|CHECKPHASE|HELDOUBLECHECKED)\s*=",
                    r"^FPE_IN_(?:DP|QP)_REDUCTION\s*="):
        if re.search(pattern, joined, re.M):
            raise ValueError("MadLoop COMMON mutation")
    if re.search(r"\b(?:LOOPRES|LOOPCOEFS|MP_LOOPRES|RAW)\b", joined):
        raise ValueError("unqualified result buffer read")
    controls = [s for s in code[:first]
                if s.startswith("CALL KTRACE_DRIVER_CONTROL(")]
    if controls != ["CALL KTRACE_DRIVER_CONTROL(.FALSE.,76,0)"]:
        raise ValueError("bootstrap tracing enabled")
    before_measurement = [s for s in code[first:second]
                          if s.startswith("CALL KTRACE_DRIVER_CONTROL(")]
    if before_measurement != [
            "CALL KTRACE_DRIVER_CONTROL(TRACE_ON.EQ.1,76,POINT_ID)"]:
        raise ValueError("measurement trace context missing")
    after_measurement = code[second + 1:second + 3]
    if after_measurement != ["CALL SYSTEM_CLOCK(T1)",
                             "CALL KTRACE_DRIVER_CONTROL(.FALSE.,76,0)"]:
        raise ValueError("measurement trace context leaks")
    gate = ["IF(.NOT.BOOT_READY) THEN",
            "WRITE(79,'(A)') 'UNQUALIFIED_INCOMPLETE_BOOTSTRAP'",
            "FLUSH(79)", "WRITE(74,'(A)') 'ERROR bootstrap-incomplete'",
            "FLUSH(74)", "CALL ML5_0_EXIT_MADLOOP()", "STOP 27", "ENDIF"]
    gate_start = code.index(gate[0])
    if code[gate_start:gate_start + len(gate)] != gate:
        raise ValueError("initialization failure gate altered")
    if not first < gate_start < second:
        raise ValueError("initialization gate misplaced")
    return code


class DriverSourceTests(unittest.TestCase):
    def test_cms_ew_both_variants_fixed_form(self):
        for nexternal in (4, 5):
            for enabled in (False, True):
                with self.subTest(nexternal=nexternal, enabled=enabled):
                    check_contract(render(nexternal=nexternal,
                                          instrumented=enabled))

    def test_control_is_only_variant_difference(self):
        enabled = statements(render())
        disabled = statements(render(instrumented=False))
        differences = [(left, right) for left, right in zip(enabled, disabled)
                       if left != right]
        self.assertEqual(len(enabled), len(disabled))
        self.assertEqual(differences, [(
            "CALL ML5_0_PRIVATE_KTRACE_SET(LUN,POINT,ENABLED)", "CONTINUE")])

    def test_guard_mutations_are_rejected(self):
        text = render()
        mutations = (
            ("IF(DID_RUN.OR.SEQUENCE.NE.1.OR.NLOOPS.NE.1)",
             "IF(SEQUENCE.LT.1.OR.NLOOPS.LT.1)"),
            ("DID_RUN=.TRUE.", "DID_RUN=.FALSE."),
            ("FORCE_STABILITY_CHECK(.FALSE.)", "FORCE_STABILITY_CHECK(.TRUE.)"),
            ("DO BOOT_CALL=1,5", "DO BOOT_CALL=1,6"),
            ("P=BOOT", "P=POINT"),
            ("STOP 27", "CONTINUE"),
            ("CALL KTRACE_DRIVER_CONTROL(.FALSE.,76,0)",
             "CALL KTRACE_DRIVER_CONTROL(.TRUE.,76,0)"),
            ("CODE=-987654321", "CODE=0"),
            ("ACC=-1D0", "ACC=0D0"),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                self.assertIn(old, text)
                with self.assertRaises(ValueError):
                    check_contract(text.replace(old, new, 1))
        for line in ("      CHECKPHASE=.FALSE.\n",
                     "      FPE_IN_DP_REDUCTION=.FALSE.\n",
                     "      CALL ML5_0_SLOOPMATRIX_THRES(P,A,T,B,C)\n"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    check_contract(text.replace(" 100  CONTINUE", line +
                                                " 100  CONTINUE", 1))

    def test_full_split_orders_and_raw_fields_preserved(self):
        code = check_contract(render())
        self.assertIn("DO J=0,NDIM", code)
        self.assertIn("ESTIMATE=ACC(0)", code)
        self.assertIn("IF(J.LE.NSO) ESTIMATE=ACC(J)", code)
        self.assertTrue(any("TRANSFER(ANS(:,J),0_8,4)" in s for s in code))
        self.assertTrue(any("TRANSFER(ESTIMATE,0_8)" in s for s in code))
        for name in ("FPE_IN_DP_REDUCTION", "FPE_IN_QP_REDUCTION",
                     "CHECKPHASE", "HELDOUBLECHECKED"):
            self.assertTrue(any(f"TRANSFER({name},0)" in s for s in code))
        self.assertIn("INIT_READY=.NOT.CHECKPHASE.AND.HELDOUBLECHECKED", code)

    def test_strict_inputs_and_worker_protocol(self):
        code = check_contract(render())
        for statement in (
                "IF(TRIM(ADJUSTL(LINE)).NE.'1') STOP 26",
                "IF(NTOKENS.NE.4) STOP 26",
                "IF(IAND(WORD,EXP_MASK).EQ.EXP_MASK) STOP 26",
                "IF(LEN_TRIM(LINE).NE.0) STOP 26",
                "CALL GET_COMMAND_ARGUMENT(6,ARG)"):
            self.assertIn(statement, code)
        for marker in ("'READY '", "'DONE '", "'BYE'",
                       "'UNQUALIFIED_INCOMPLETE_BOOTSTRAP'",
                       "'UNQUALIFIED_INCOMPLETE_MEASUREMENT'"):
            self.assertTrue(any(marker in s for s in code))
        self.assertEqual(sum("ADVANCE='NO'" in s for s in code), 3)
        self.assertNotIn("DO K=1,1000", "\n".join(code))


if __name__ == "__main__":
    unittest.main()
