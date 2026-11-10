"""Source contracts only: no compiler, model, loader or amplitude execution."""
import re
import struct
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "stability_worker.f"


def statements(text):
    """Join fixed-form continuations, excluding comments and whitespace."""
    records = []
    for line in text.splitlines():
        if not line or line[0] in "Cc*!":
            continue
        code = line[6:]
        if len(line) > 5 and line[5] not in " 0":
            if not records:
                raise ValueError("Continuation without an initial statement")
            records[-1] += code
        else:
            records.append(code)
    return [re.sub(r"\s+", "", record).upper() for record in records]


class DriverSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = TEMPLATE.read_text()
        cls.code = statements(cls.text)
        cls.joined = "\n".join(cls.code)

    def test_only_declared_render_parameters(self):
        self.assertEqual(set(re.findall(r"@[A-Z_]+@", self.text)),
                         {"@PREFIX@", "@NEXTERNAL@", "@NBORN@"})
        for prefix, nexternal, nborn in (("ML5_0_", 4, 1),
                                          ("ML5_37_", 5, 3)):
            rendered = self.text.replace("@PREFIX@", prefix).replace(
                "@NEXTERNAL@", str(nexternal)).replace("@NBORN@", str(nborn))
            self.assertNotRegex(rendered, r"@[A-Z_]+@")
            self.assertIn(f"NEXTERNAL={nexternal},NSQSO_BORN={nborn}", rendered)
            self.assertIn(f"CALL {prefix}SLOOPMATRIX_THRES", rendered)

    def test_all_noncomment_fixed_form_lines_fit(self):
        # The process makefile currently permits longer lines. Keep this new
        # template portable to ordinary fixed-form width without that flag.
        for number, line in enumerate(self.text.splitlines(), 1):
            if line and line[0] not in "Cc*!":
                self.assertLessEqual(len(line), 72, f"line {number}")

    def test_exact_point_count_and_bounded_bootstrap(self):
        self.assertIn("PARAMETER(REQUIRED_POINTS=100,MAX_BOOTSTRAP=16)", self.code)
        self.assertIn("IF(IOS.NE.0.OR.NPOINTS.NE.REQUIRED_POINTS)STOP12", self.code)
        self.assertIn("DOBOOTSTRAP=1,MAX_BOOTSTRAP", self.code)
        self.assertIn("POINT_INDEX=1+MOD(BOOTSTRAP-1,NPOINTS)", self.code)
        self.assertIn("IF(.NOT.READY)THEN", self.code)
        self.assertIn("'STABILITY_BOOTSTRAP_LIMIT'", self.joined)

    def test_primary_is_three_methods_on_every_original_point(self):
        self.assertIn("DOK=1,NPOINTS", self.code)
        self.assertIn("DOMETHOD=1,3", self.code)
        self.assertIn("P=POINTS(:,:,K)", self.code)
        self.assertIn("MODE=1", self.code)
        self.assertIn("IF(METHOD.EQ.2)MODE=2", self.code)
        self.assertIn("P(1,I)=-POINTS(2,I,K)", self.code)
        self.assertIn("P(2,I)=POINTS(1,I,K)", self.code)
        self.assertNotIn("P(0,I)=", self.joined)
        self.assertNotIn("P(3,I)=", self.joined)

    def test_no_automatic_or_quadruple_precision_mode(self):
        for statement in self.code:
            self.assertNotRegex(statement, r"CTMODE(?:INIT|RUN)?=(-1|[456])(?:$|[^0-9])")
        self.assertIn("CALL@PREFIX@FORCE_STABILITY_CHECK(.FALSE.)", self.code)
        self.assertNotIn("FORCE_STABILITY_CHECK(.TRUE.)", self.joined)
        self.assertIn("IF(CODE.EQ.110)THEN", self.code)
        self.assertIn("IF(CODE.NE.130)THEN", self.code)
        self.assertIn("IF(CODE.NE.110)THEN", self.code)

    def test_card_and_backend_cannot_silently_change(self):
        self.assertIn("CALLSTABILITY_CHECK_CARD(BACKEND,1)", self.code)
        self.assertIn("CALLSTABILITY_CHECK_CARD(BACKEND,MODE)", self.code)
        for guard in ("CTMODEINIT.NE.1", "CTMODERUN.NE.MODE",
                      "MLREDUCTIONLIB(1).NE.BACKEND", "MLREDUCTIONLIB(2).NE.0"):
            self.assertIn(guard, self.joined)

    def test_rows_keep_total_all_orders_and_raw_accuracy(self):
        self.assertIn("ALLOCATE(ANS(0:3,0:NDIM),ACC(0:NSO))", self.code)
        self.assertIn("IF(NDIM.NE.MAX(NSO,NSQSO_BORN))STOP13", self.code)
        self.assertIn("DOJ=0,NDIM", self.code)
        self.assertIn("ESTIMATE=ACC(0)", self.code)
        self.assertIn("IF(J.LE.NSO)ESTIMATE=ACC(J)", self.code)
        self.assertIn("WRITE(75,'(6I12,5ES26.17E3)')PHASE,CALL_INDEX,"
                      "POINT_INDEX,METHOD,J,CODE,ANS(:,J),ESTIMATE", self.code)
        self.assertIn("ANS=0D0", self.code)
        self.assertIn("ACC=-1D0", self.code)

    def test_diagnostics_remain_visible_and_calls_are_delimited(self):
        self.assertNotIn("/DEV/NULL", self.joined)
        self.assertNotRegex(self.joined, r"(?:OPEN|CLOSE)\(6[,)]")
        call = self.code.index("CALL@PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)")
        begin = next(i for i, s in enumerate(self.code) if "'STABILITY_CALL_BEGIN'" in s)
        end = next(i for i, s in enumerate(self.code) if "'STABILITY_CALL_END'" in s)
        self.assertLess(begin, call)
        self.assertLess(call, end)
        self.assertIn("FLUSH(6)", self.code[begin:call])
        self.assertIn("FLUSH(6)", self.code[end:end + 3])

    def test_result_files_are_new_and_completion_is_last(self):
        for name in ("STABILITY-VALUES.DAT", "STABILITY-DIMENSIONS.DAT",
                     "STABILITY-GROUPS.DAT"):
            statement = next(s for s in self.code if f"FILE='{name}'" in s)
            self.assertIn("STATUS='NEW'", statement)
        self.assertIn("WRITE(76,'(5I12)')1,NPOINTS,NDIM,NSO,NSQSO_BORN", self.code)
        completion = self.code.index("WRITE(75,'(A,2I12)')'#COMPLETE',BOOTSTRAP,3*NPOINTS")
        primary_reject = self.code.index("IF(CODE.NE.110)THEN")
        self.assertGreater(completion, primary_reject)
        self.assertIn("FLUSH(75)", self.code[completion:completion + 3])

    def body(self, name):
        start = next(i for i, line in enumerate(self.code)
                     if line.startswith("SUBROUTINE" + name + "("))
        end = self.code.index("END", start)
        return self.code[start:end + 1]

    def test_group_layout_is_case_specific_and_rank_bounded(self):
        self.assertIn("INCLUDE'LOOP_MAX_COEFS.INC'", self.code)
        self.assertIn("INCLUDE'STABILITY-LAYOUT.INC'", self.code)
        self.assertIn("IF(NSO.NE.NSQUAREDSO.OR.NLOOPGROUPS.LE.0)STOP13", self.code)
        self.assertIn("IF(STABILITY_NCOEFS(I).LE.0.OR."
                      "STABILITY_NCOEFS(I).GT.LOOPMAXCOEFS)STOP13", self.code)
        snapshot = self.body("STABILITY_GROUP_SNAPSHOT")
        self.assertIn("COMPLEX*16LOOPCOEFS(0:LOOPMAXCOEFS-1,"
                      "NSQUAREDSO,NLOOPGROUPS)", snapshot)
        self.assertIn("COMMON/@PREFIX@LCOEFS/LOOPCOEFS", snapshot)
        self.assertIn("COMPLEX*16LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)", snapshot)
        self.assertIn("COMMON/@PREFIX@LOOPRES/LOOPRES,S", snapshot)
        self.assertIn("DOK=0,STABILITY_NCOEFS(I)-1", snapshot)

    def test_group_snapshot_never_reads_unconstructed_coefficients(self):
        snapshot = self.body("STABILITY_GROUP_SNAPSHOT")
        gate = snapshot.index("IF(NSO.NE.NSQUAREDSO.OR..NOT.LOOP_REQ_SO_DONE.OR."
                              "SQSO_TARGET.NE.-1.OR.FILTER_SO)GOTO200")
        first_read = snapshot.index("WORDS=TRANSFER(LOOPCOEFS(K,J,I),WORDS)")
        self.assertLess(gate, first_read)
        active = snapshot.index("IF(NZ.GT.0)THEN")
        stable_read = snapshot.index("STABLE=MERGE(1,0,S(J,I))")
        self.assertLess(active, stable_read)
        self.assertIn("STABLE=-1", snapshot[:active])
        self.assertNotRegex("\n".join(snapshot), r"(?m)^(S|LOOPCOEFS|LOOPRES)\([^\n]*=")
        self.assertIn("COMMON/@PREFIX@SOCHOICE/SQSO_TARGET", snapshot)

    def test_native_latches_are_observed_before_after_without_reset(self):
        call = self.code.index("CALL@PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)")
        before = self.code.index("CALLSTABILITY_FLAG_BEFORE(PHASE,CALL_INDEX,POINT_INDEX,METHOD)")
        after = self.code.index("CALLSTABILITY_GROUP_SNAPSHOT(PHASE,CALL_INDEX,POINT_INDEX,METHOD,NSO)")
        self.assertLess(before, call)
        self.assertLess(call, after)
        for name in ("FPE_IN_DP_REDUCTION", "FPE_IN_QP_REDUCTION"):
            self.assertNotRegex(self.joined, r"(?m)^" + name + r"=")
        self.assertNotIn("IEEE_SET_", self.joined)
        self.assertIn("WRITE(77,'(A,6I12)')'B',PHASE,CALL_INDEX,POINT_INDEX,"
                      "METHOD,MERGE(1,0,FPE_IN_DP_REDUCTION),"
                      "MERGE(1,0,FPE_IN_QP_REDUCTION)", self.code)

    def test_group_classification_uses_integer_bits_only(self):
        snapshot = self.body("STABILITY_GROUP_SNAPSHOT")
        joined = "\n".join(snapshot)
        self.assertIn("PARAMETER(MAG_MASK=INT(Z'7FFFFFFFFFFFFFFF',INT64))", snapshot)
        self.assertIn("PARAMETER(EXP_MASK=INT(Z'7FF0000000000000',INT64))", snapshot)
        self.assertIn("IF(IAND(WORDS(L),MAG_MASK).NE.0_INT64)THEN", snapshot)
        self.assertIn("IF(IAND(WORDS(L),EXP_MASK).EQ.EXP_MASK)NF_COEF=NF_COEF+1", snapshot)
        for operation in ("ABS(", "DBLE(", "AIMAG(", "IEEE_IS_", "IEEE_SET_"):
            self.assertNotIn(operation, joined)
        # Explicit IEEE edge cases describe the mask contract independently
        # of any Fortran execution: signed zero is inactive; sNaN is active
        # and nonfinite without ever converting it to a floating value.
        for word, nonzero, finite in ((0, False, True),
                                      (0x8000000000000000, False, True),
                                      (0x0000000000000001, True, True),
                                      (0x7FF0000000000000, True, False),
                                      (0x7FF0000000000001, True, False)):
            self.assertEqual(bool(word & 0x7FFFFFFFFFFFFFFF), nonzero)
            self.assertEqual((word & 0x7FF0000000000000) != 0x7FF0000000000000, finite)
        self.assertEqual(struct.unpack('=Q', struct.pack('=d', -0.0))[0],
                         0x8000000000000000)

    def test_group_journal_is_complete_and_bad_groups_do_not_abort(self):
        snapshot = self.body("STABILITY_GROUP_SNAPSHOT")
        self.assertIn("DOI=1,NLOOPGROUPS", snapshot)
        self.assertIn("DOJ=1,NSQUAREDSO", snapshot)
        self.assertIn("WRITE(77,'(A,11I12,8(1X,Z16.16))')'G',PHASE,"
                      "CALL_INDEX,POINT_INDEX,METHOD,J,I,STABILITY_NCOEFS(I),"
                      "NZ,NF_COEF,STABLE,NF_RES,FIRST_NZ,RES_WORDS", snapshot)
        self.assertIn("WRITE(77,'(A,11I12)')'C',PHASE,CALL_INDEX,POINT_INDEX,"
                      "METHOD,COVERAGE,ACTIVE,FAILED,COEF_BAD,RES_BAD,FPE_DP,FPE_QP", snapshot)
        self.assertEqual([line for line in snapshot if "STOP" in line],
                         ["IF(STORAGE_SIZE(LOOPCOEFS(0,1,1)).NE.128)STOP19"])
        self.assertIn("WRITE(77,'(A,2I12)')'#COMPLETE',BOOTSTRAP,3*NPOINTS", self.code)

    def test_no_data_generation_or_born_normalization(self):
        self.assertNotIn("RAMBO", self.joined)
        self.assertNotIn("RANDOM", self.joined)
        self.assertNotIn("AO2PI", self.joined)
        self.assertNotRegex(self.joined, r"ANS\([^\n]*\)/")


if __name__ == "__main__":
    unittest.main()
