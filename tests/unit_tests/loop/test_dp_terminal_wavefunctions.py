"""The private DP updater requires a complete, fail-closed consumer proof.

These fixtures are source text only: no compiler, native executable, reference
checkout or benchmark artifact is needed.
"""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from madgraph.iolibs.file_writers import FortranWriter
from madgraph.loop.dp_terminal_proof import validate_dp_terminal_wavefunctions


class DPTerminalConsumerProofTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='madloop-dp-proof-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.counter = 0

    def fixture(self, prefix='TEST_', directory=None):
        self.counter += 1
        root = directory or self.work / ('case_%d' % self.counter)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'coef_specs.inc').write_text(
            '      INTEGER MAXLWFSIZE,LOOPMAXCOEFS,VERTEXMAXCOEFS\n'
            '      PARAMETER(MAXLWFSIZE=4,LOOPMAXCOEFS=15,VERTEXMAXCOEFS=5)\n')

        def declaration(alias='WL'):
            momentum = 'DP_PL' if alias == 'DP_WL' else 'PL'
            return (
                '      INTEGER NLOOPWAVEFUNCS\n'
                '      PARAMETER(NLOOPWAVEFUNCS=2)\n'
                '      COMPLEX*16 %s(MAXLWFSIZE,0:LOOPMAXCOEFS-1,\n'
                '     $ MAXLWFSIZE,-1:NLOOPWAVEFUNCS)\n'
                '      COMPLEX*16 %s(0:3,-1:NLOOPWAVEFUNCS)\n'
                '      COMMON/%sWL/%s,%s\n'
            ) % (alias, momentum, prefix, alias, momentum)

        seed_lines = (
            '(I,J,K,-1)=(0.0D0,0.0D0)',
            '(I,J,K,0)=(1.0D0,0.0D0)',
            '(I,J,K,0)=(0.0D0,0.0D0)',
            '(1,0,2,-1)=DCMPLX(-1.0D0,0.0D0)',
            '(2,0,1,-1)=DCMPLX(1.0D0,0.0D0)',
            '(3,0,4,-1)=DCMPLX(1.0D0,0.0D0)',
            '(4,0,3,-1)=DCMPLX(-1.0D0,0.0D0)',
        )
        for name, alias in (('loop_matrix.f', 'WL'),
                            ('mp_compute_loop_coefs.f', 'DP_WL')):
            (root / name).write_text(
                '      SUBROUTINE SEEDS()\n'
                '      USE %sPOLYNOMIAL_CONSTANTS\n' % prefix + declaration(alias) +
                ''.join('      ' + alias + line + '\n' for line in seed_lines) +
                '      END\n')
        (root / 'helas_calls_ampb_1.f').write_text(
            '      SUBROUTINE BORN()\n'
            '      USE %sPOLYNOMIAL_CONSTANTS\n' % prefix + declaration() + '      END\n')
        (root / 'coef_construction_1.f').write_text(
            '      SUBROUTINE CONSTRUCT()\n'
            '      USE %sPOLYNOMIAL_CONSTANTS\n' % prefix +
            '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n' + declaration() +
            '      LOGICAL WL_ENTRY_INEXACT\n'
            '      CALL %sUPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,1))\n'
            '      CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
            '      CALL %sDP_TERMINAL_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,2),WL_ENTRY_INEXACT)\n'
            '      CALL %sCREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1,1,1,1,1)\n'
            '      END\n' % (prefix, prefix, prefix))
        template = (Path(__file__).resolve().parents[3] /
                    'madgraph/iolibs/template_files/loop_optimized/polynomial.inc')
        create = '\n'.join(line for line in template.read_text().split('## }else{', 1)[0].splitlines()
                           if not line.lstrip().startswith('##'))
        create %= dict(mp_prefix='', proc_prefix=prefix, nbornamps=1,
                       real_format='REAL*8', complex_format='COMPLEX*16',
                       zero_def='0.0D0', one_def='1.0D0', nloopamps=1,
                       nloop_groups=1, ncomb=1, nSO=1, nSquaredSO=1,
                       nAmpSO=1, color_matrix_size='NBORNAMPS', kind=8)
        rendered = root / 'rendered_create.inc'
        with FortranWriter(str(rendered)) as writer:
            writer.writelines(create)
        create = rendered.read_text()
        rendered.unlink()
        (root / 'polynomial.f').write_text(
            '      MODULE %sPOLYNOMIAL_CONSTANTS\n'
            "      INCLUDE 'coef_specs.inc'\n"
            '      INTEGER NCOEF_R(0:2)\n'
            '      DATA NCOEF_R/1,5,15/\n'
            '      END MODULE %sPOLYNOMIAL_CONSTANTS\n'
            '      SUBROUTINE %sMERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)\n'
            '      USE %sPOLYNOMIAL_CONSTANTS\n'
            '      INTEGER I,J\n'
            '      COMPLEX*16 WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)\n'
            '      INTEGER R,LCUT_SIZE\n'
            '      COMPLEX*16 CONST\n'
            '      COMPLEX*16 OUT(0:LOOPMAXCOEFS-1)\n'
            '      DO I=1,LCUT_SIZE\n'
            '        DO J=0,NCOEF_R(R)-1\n'
            '          OUT(J)=OUT(J)+WL(I,J,I)*CONST\n'
            '        ENDDO\n'
            '      ENDDO\n'
            '      END\n' % (prefix, prefix, prefix, prefix) + create)
        expected = {2: dict(loop_mother_rank=1, vertex_rank=1,
                           loop_mother_number=1, lcut_size=4, in_size=4,
                           out_size=4, out=2)}
        return root, prefix, expected

    def validate(self, fixture, **kwargs):
        root, prefix, expected = fixture
        return validate_dp_terminal_wavefunctions(str(root), prefix, expected, **kwargs)

    def replace(self, root, name, old, new):
        path = root / name
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def test_known_graph_and_empty_process_prefix(self):
        for prefix in ('TEST_', ''):
            with self.subTest(prefix=prefix):
                self.assertTrue(self.validate(self.fixture(prefix)))

    def test_standard_intrinsic_import_and_user_named_variables(self):
        fixture = self.fixture()
        self.replace(fixture[0], 'loop_matrix.f',
                     '      USE TEST_POLYNOMIAL_CONSTANTS',
                     '      USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE\n'
                     '      USE TEST_POLYNOMIAL_CONSTANTS')
        self.replace(fixture[0], 'loop_matrix.f', '      INTEGER NLOOPWAVEFUNCS',
                     '      INTEGER USERHEL,USER_STAB_PREC\n'
                     '      INTEGER NLOOPWAVEFUNCS')
        self.replace(fixture[0], 'loop_matrix.f', '      END\n',
                     '      USERHEL=1\n      USER_STAB_PREC=-1\n      END\n')
        self.assertTrue(self.validate(fixture))

    def test_manifest_must_include_every_source(self):
        fixture = self.fixture()
        root = fixture[0]
        manifest = sorted(path.name for path in root.glob('*.f'))
        self.assertTrue(self.validate(fixture, source_manifest=manifest))
        for omitted in ('mp_compute_loop_coefs.f', 'helas_calls_ampb_1.f',
                        'polynomial.f'):
            with self.subTest(omitted=omitted):
                self.assertFalse(self.validate(
                    fixture, source_manifest=[name for name in manifest if name != omitted]))

    def test_unknown_duplicate_and_later_wavefunction_consumers(self):
        mutations = (
            ('WL(1,0,1,0),4,COEFS', 'WL(1,0,1,99),4,COEFS'),
            ('WL(1,0,1,1))', 'WL(1,0,1,2))'),
            ('WL(1,0,1,1),4,COEFS', 'WL(1,0,1,2),4,COEFS'),
            ('CREATE_LOOP_COEFS(WL(1,0,1,2)', 'CREATE_LOOP_COEFS(WL(1,0,1,99)'),
        )
        for old, new in mutations:
            with self.subTest(mutation=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_construction_1.f', old, new)
                self.assertFalse(self.validate(fixture))
        for call in (
            'CALL TEST_UPDATE_WL_0_1(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))',
            'CALL TEST_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))',
            'CALL HIDDEN(WL)', 'CALL HIDDEN(WL(1,0,1,2))',
        ):
            with self.subTest(call=call):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_construction_1.f',
                             '      END\n', '      ' + call + '\n      END\n')
                self.assertFalse(self.validate(fixture))

    def test_expected_terminal_shape_and_identity_are_binding(self):
        for key, value in (('out', 3), ('loop_mother_number', 0), ('vertex_rank', 2),
                           ('lcut_size', 3), ('in_size', 3), ('out_size', 3)):
            with self.subTest(key=key):
                fixture = self.fixture()
                fixture[2][2][key] = value
                self.assertFalse(self.validate(fixture))

    def test_common_alias_and_storage_layout_changes_rejected(self):
        changes = (
            ('COMMON/TEST_WL/DP_WL,DP_PL', 'COMMON/TEST_WL/OTHER,DP_PL'),
            ('COMMON/TEST_WL/DP_WL,DP_PL', 'COMMON/TEST_WL/DP_PL,DP_WL'),
            ('COMPLEX*16 DP_WL', 'COMPLEX*32 DP_WL'),
            ('MAXLWFSIZE,-1:NLOOPWAVEFUNCS)', 'MAXLWFSIZE,0:NLOOPWAVEFUNCS)'),
            ('DP_PL(0:3,-1:NLOOPWAVEFUNCS)', 'DP_PL(0:4,-1:NLOOPWAVEFUNCS)'),
            ('COMMON/TEST_WL/DP_WL,DP_PL', 'COMMON/OTHER/X/TEST_WL/DP_WL,DP_PL'),
            ('COMMON/TEST_WL/DP_WL,DP_PL', 'COMMON/TEST_WL/DP_WL,DP_PL/OTHER/X'),
        )
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'mp_compute_loop_coefs.f', old, new)
                self.assertFalse(self.validate(fixture))

    def test_array_extents_must_resolve_to_the_private_helper_contract(self):
        changes = (('MAXLWFSIZE=4', 'MAXLWFSIZE=16'),
                   ('LOOPMAXCOEFS=15', 'LOOPMAXCOEFS=14'),
                   ('VERTEXMAXCOEFS=5', 'VERTEXMAXCOEFS=4'),
                   ('MAXLWFSIZE=4,', ''))
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_specs.inc', old, new)
                self.assertFalse(self.validate(fixture))

    def test_rank_coefficient_table_requires_canonical_literal_layout(self):
        declaration = '      INTEGER NCOEF_R(0:2)\n'
        table = '      DATA NCOEF_R/1,5,15/\n'
        changes = (
            (declaration, declaration.replace('(0:2)', '(1:3)')),
            (declaration, declaration.replace('(0:2)', '(0:1)')),
            (declaration, declaration.replace('(0:2)', '(0:3)')),
            (declaration, declaration.replace('INTEGER', 'REAL*8')),
            (declaration, declaration.replace('(0:2)', '(0:1+1)')),
            (declaration, ''),
            (declaration, declaration + declaration),
            (table, table.replace('1,5,15', '1,5,16')),
            (table, table.replace('1,5,15', '1,5,14')),
            (table, table.replace('1,5,15', '1,6,15')),
            (table, table.replace('1,5,15', '1,5,10+5')),
            (table, ''),
            (table, table + table),
        )
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'polynomial.f', old, new)
                self.assertFalse(self.validate(fixture))

    def test_rank_coefficient_table_must_cover_terminal_rank_and_fit_storage(self):
        fixture = self.fixture()
        self.replace(fixture[0], 'polynomial.f', 'NCOEF_R(0:2)', 'NCOEF_R(0:1)')
        self.replace(fixture[0], 'polynomial.f', 'DATA NCOEF_R/1,5,15/',
                     'DATA NCOEF_R/1,5/')
        self.assertFalse(self.validate(fixture))
        fixture = self.fixture()
        self.replace(fixture[0], 'polynomial.f', 'NCOEF_R(0:2)', 'NCOEF_R(0:3)')
        self.replace(fixture[0], 'polynomial.f', 'DATA NCOEF_R/1,5,15/',
                     'DATA NCOEF_R/1,5,15,35/')
        self.assertFalse(self.validate(fixture))
        self.replace(fixture[0], 'coef_specs.inc', 'LOOPMAXCOEFS=15', 'LOOPMAXCOEFS=35')
        self.assertTrue(self.validate(fixture))

    def test_local_rank_coefficient_declaration_or_assignment_rejected(self):
        mutations = (
            ('polynomial.f', '      COMPLEX*16 LOOP_WF(',
             '      INTEGER NCOEF_R(0:2)\n'),
            ('coef_construction_1.f', '      COMPLEX*16 WL(',
             '      INTEGER NCOEF_R(0:2)\n'),
            ('polynomial.f', '      CALL TEST_MERGE_WL(LOOP_WF',
             '      NCOEF_R(2)=16\n'),
            ('coef_construction_1.f', '      CALL TEST_CREATE_LOOP_COEFS(',
             '      NCOEF_R(2)=16\n'),
        )
        for name, marker, addition in mutations:
            with self.subTest(source=name, addition=addition):
                fixture = self.fixture()
                self.replace(fixture[0], name, marker, addition + marker)
                self.assertFalse(self.validate(fixture))

    def test_rank_coefficient_table_cannot_be_modified_or_escape_elsewhere(self):
        for statement in ('NCOEF_R(2)=16', 'CALL HIDDEN(NCOEF_R)'):
            with self.subTest(statement=statement):
                fixture = self.fixture()
                (fixture[0] / 'born_matrix.f').write_text(
                    '      SUBROUTINE OTHER()\n'
                    '      USE TEST_POLYNOMIAL_CONSTANTS\n'
                    '      ' + statement + '\n      END\n')
                self.assertFalse(self.validate(fixture))

    def test_create_rank_and_cut_size_cannot_change_or_escape(self):
        for statement in ('RANK=99', 'LCUT_SIZE=5', 'CALL HIDDEN(LCUT_SIZE)'):
            with self.subTest(statement=statement):
                fixture = self.fixture()
                marker = '      CALL TEST_MERGE_WL(LOOP_WF'
                self.replace(fixture[0], 'polynomial.f', marker,
                             '      ' + statement + '\n' + marker)
                self.assertFalse(self.validate(fixture))

    def test_labeled_capture_or_private_call_rejected(self):
        for routine in ('IEEE_GET_FLAG', 'TEST_DP_TERMINAL_WL_1_1'):
            with self.subTest(routine=routine):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_construction_1.f',
                             '      CALL ' + routine, ' 1001 CALL ' + routine)
                self.assertFalse(self.validate(fixture))

    def test_seed_alias_cannot_access_terminal_or_escape(self):
        for new in ('DP_WL(I,J,K,2)=(0.0D0,0.0D0)', 'CALL HIDDEN(DP_WL)',
                    'EQUIVALENCE(DP_WL,ESCAPE)', 'X=DP_WL(1,0,2,2)'):
            with self.subTest(statement=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'mp_compute_loop_coefs.f',
                             'DP_WL(I,J,K,-1)=(0.0D0,0.0D0)', new)
                self.assertFalse(self.validate(fixture))

    def test_fixed_form_identifier_blanks_cannot_hide_consumers(self):
        mutations = (
            ('helas_calls_ampb_1.f', 'X=W L(1,0,2,2)'),
            ('helas_calls_ampb_1.f', 'CALL HIDDEN(W L)'),
            ('helas_calls_ampb_1.f', 'EQUI VALENCE(W L,ESCAPE)'),
            ('mp_compute_loop_coefs.f', 'X=DP_ WL(1,0,2,2)'),
            ('coef_construction_1.f', 'CALL HIDDEN(WL_ENTRY_ INEXACT)'),
            ('coef_construction_1.f', 'COMMON/FLAG_ESCAPE/WL_ENTRY_ INEXACT'),
        )
        for name, statement in mutations:
            with self.subTest(statement=statement):
                fixture = self.fixture()
                self.replace(fixture[0], name, '      END\n',
                             '      ' + statement + '\n      END\n')
                self.assertFalse(self.validate(fixture))
        fixture = self.fixture()
        self.replace(fixture[0], 'polynomial.f',
                     '      CALL TEST_MERGE_WL(LOOP_WF',
                     '      X=LOOP_ WF(1,0,2)\n'
                     '      CALL TEST_MERGE_WL(LOOP_WF')
        self.assertFalse(self.validate(fixture))

    def test_fixed_form_include_keyword_blanks_cannot_hide_consumers(self):
        fixture = self.fixture()
        (fixture[0] / 'hidden.inc').write_text('      X=WL(1,0,2,2)\n')
        self.replace(fixture[0], 'helas_calls_ampb_1.f', '      END\n',
                     "      IN CLUDE 'hidden.inc'\n      END\n")
        self.assertFalse(self.validate(fixture))

    def test_internal_consumer_cannot_shadow_certified_external_routine(self):
        fixture = self.fixture()
        self.replace(fixture[0], 'coef_construction_1.f', '      END\n',
                     '      CONTAINS\n'
                     '      SUBROUTINE TEST_CREATE_LOOP_COEFS(A,R,S,C)\n'
                     '      COMPLEX*16 A(4,0:14,4)\n'
                     '      INTEGER R,S,C\n'
                     '      WRITE(*,*) A(1,0,2)\n'
                     '      END SUBROUTINE TEST_CREATE_LOOP_COEFS\n'
                     '      END\n')
        self.assertFalse(self.validate(fixture))

    def test_imported_procedure_and_generic_cannot_shadow_consumer(self):
        bindings = (
            ('SUBROUTINE TEST_CREATE_LOOP_COEFS(A,R,S,C)', ''),
            ('RECURSIVE SUBROUTINE TEST_CREATE_LOOP_COEFS(A,R,S,C)', ''),
            ('SUBROUTINE HIDDEN(A,R,S,C)',
             '      INTERFACE TEST_CREATE_LOOP_COEFS\n'
             '      MODULE PROCEDURE HIDDEN\n'
             '      END INTERFACE\n'),
        )
        for declaration, interface in bindings:
            with self.subTest(declaration=declaration):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_construction_1.f',
                             '      USE TEST_POLYNOMIAL_CONSTANTS',
                             '      USE OTHER_CONSUMER\n'
                             '      USE TEST_POLYNOMIAL_CONSTANTS')
                (fixture[0] / 'born_matrix.f').write_text(
                    '      MODULE OTHER_CONSUMER\n' + interface +
                    '      CONTAINS\n      ' + declaration + '\n'
                    '      COMPLEX*16 A(4,0:14,4)\n'
                    '      INTEGER R,S,C\n'
                    '      WRITE(*,*) A(1,0,2)\n'
                    '      END SUBROUTINE\n'
                    '      END MODULE OTHER_CONSUMER\n')
                self.assertFalse(self.validate(fixture))
        with self.subTest(import_binding='external operator and consumer'):
            fixture = self.fixture()
            self.replace(fixture[0], 'coef_construction_1.f',
                         '      USE TEST_POLYNOMIAL_CONSTANTS',
                         '      USE EVIL, ONLY: OPERATOR(==), TEST_CREATE_LOOP_COEFS\n'
                         '      USE TEST_POLYNOMIAL_CONSTANTS')
            self.assertFalse(self.validate(fixture))

    def test_local_bounds_cannot_override_the_certified_module(self):
        mutations = (
            ('coef_construction_1.f', '      COMPLEX*16 WL(', '16'),
            ('mp_compute_loop_coefs.f', '      COMPLEX*16 DP_WL(', '16'),
            ('polynomial.f', '      COMPLEX*16 LOOP_WF(', '16-0'),
        )
        for name, declaration, value in mutations:
            with self.subTest(source=name, value=value):
                fixture = self.fixture()
                self.replace(fixture[0], name, declaration,
                             '      INTEGER MAXLWFSIZE\n'
                             '      PARAMETER(MAXLWFSIZE=' + value + ')\n' + declaration)
                self.assertFalse(self.validate(fixture))

    def test_mp_common_wavefunction_extent_must_match_dp_extent(self):
        fixture = self.fixture()
        self.replace(fixture[0], 'mp_compute_loop_coefs.f',
                     'PARAMETER(NLOOPWAVEFUNCS=2)', 'PARAMETER(NLOOPWAVEFUNCS=1)')
        self.assertFalse(self.validate(fixture))

    def test_c_binding_cannot_alias_the_common_block_under_another_name(self):
        fixture = self.fixture()
        (fixture[0] / 'born_matrix.f').write_text(
            '      SUBROUTINE HIDDEN()\n'
            '      COMPLEX*16 A(4,0:14,4,-1:2),P(0:3,-1:2)\n'
            '      COMMON/ESCAPE/A,P\n'
            "      BIND(C,NAME='test_wl_') /ESCAPE/\n"
            '      WRITE(*,*) A(1,0,2,2)\n'
            '      END\n')
        self.assertFalse(self.validate(fixture))

    def test_flag_capture_must_be_immediate_and_unmodified(self):
        capture = '      CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
        changes = ('', capture + '      CALL OTHER()\n',
                   '      CALL IEEE_SET_FLAG(IEEE_INEXACT,.TRUE.)\n',
                   '      CALL IEEE_SET_FLAG(IEEE_INEXACT,.TRUE.)\n' + capture,
                   capture + '      WL_ENTRY_INEXACT=.TRUE.\n',
                   capture.replace('IEEE_INEXACT', 'IEEE_OVERFLOW'),
                   capture.replace('WL_ENTRY_INEXACT', 'OTHER_FLAG'))
        for changed in changes:
            with self.subTest(capture=changed):
                fixture = self.fixture()
                self.replace(fixture[0], 'coef_construction_1.f', capture, changed)
                self.assertFalse(self.validate(fixture))

    def test_comments_do_not_separate_capture_from_call(self):
        fixture = self.fixture()
        self.replace(fixture[0], 'coef_construction_1.f',
                     '      CALL TEST_DP_TERMINAL_WL_1_1',
                     'C     This is a non-executable comment.\n\n'
                     'c         $MG5_CITATION_DIR is ordinary comment text.\n'
                     'c$$$      CALL HIDDEN(WL)\n'
                     '      CALL TEST_DP_TERMINAL_WL_1_1')
        self.assertTrue(self.validate(fixture))

    def test_included_declarations_are_resolved(self):
        fixture = self.fixture()
        root = fixture[0]
        (root / 'ordinary.inc').write_text('      INTEGER UNUSED_PARAMETER\n')
        self.replace(root, 'coef_construction_1.f', '      LOGICAL WL_ENTRY_INEXACT',
                     "      INCLUDE 'ordinary.inc'\n      LOGICAL WL_ENTRY_INEXACT")
        self.assertTrue(self.validate(fixture))

    def test_hidden_consumer_or_common_alias_in_include_rejected(self):
        for source in ('      X=WL(1,0,2,2)\n',
                       '      CALL HIDDEN(WL)\n',
                       '      COMMON/TEST_WL/OTHER,OTHER_PL\n'):
            with self.subTest(source=source):
                fixture = self.fixture()
                root = fixture[0]
                (root / 'hidden.inc').write_text(source)
                self.replace(root, 'coef_construction_1.f', '      LOGICAL WL_ENTRY_INEXACT',
                             "      INCLUDE 'hidden.inc'\n      LOGICAL WL_ENTRY_INEXACT")
                self.assertFalse(self.validate(fixture))

    def test_missing_cyclic_and_escaping_include_rejected(self):
        for include in ('missing.inc', 'cycle.inc', '../outside.inc'):
            with self.subTest(include=include):
                fixture = self.fixture()
                root = fixture[0]
                (root.parent / 'outside.inc').write_text('      INTEGER OUTSIDE\n')
                (root / 'cycle.inc').write_text("      INCLUDE 'cycle.inc'\n")
                self.replace(root, 'coef_construction_1.f', '      LOGICAL WL_ENTRY_INEXACT',
                             "      INCLUDE '%s'\n      LOGICAL WL_ENTRY_INEXACT" % include)
                self.assertFalse(self.validate(fixture))

    def test_external_include_symlink_rejected(self):
        fixture = self.fixture()
        root = fixture[0]
        target = self.work / 'outside.inc'
        target.write_text('      INTEGER OUTSIDE\n')
        (root / 'external.inc').symlink_to(target)
        self.replace(root, 'coef_construction_1.f', '      LOGICAL WL_ENTRY_INEXACT',
                     "      INCLUDE 'external.inc'\n      LOGICAL WL_ENTRY_INEXACT")
        self.assertFalse(self.validate(fixture))

    def test_source_symlink_preserves_each_include_search_context(self):
        export = self.work / 'symlink_export'
        fixture = self.fixture(directory=export / 'SubProcesses/P0_test')
        (export / 'Source/MODEL').mkdir(parents=True)
        shared = export / 'SubProcesses/MadLoopCommons.f'
        shared.write_text('      SUBROUTINE SHARED()\n'
                          "      INCLUDE 'shared.inc'\n      END\n")
        shared.with_name('shared.inc').write_text('      INTEGER UNUSED\n')
        (fixture[0] / shared.name).symlink_to('../' + shared.name)
        (fixture[0] / 'shared.inc').write_text(
            '      COMPLEX*16 HIDDEN_WL(4,0:14,4,-1:2)\n'
            '      COMMON/TEST_WL/HIDDEN_WL\n'
            '      X=HIDDEN_WL(1,0,2,2)\n')
        self.assertFalse(self.validate(fixture))

    def test_include_roots_inside_export_are_part_of_consumer_closure(self):
        export = self.work / 'include_export'
        fixture = self.fixture(directory=export / 'SubProcesses/P0_test')
        model = export / 'Source/MODEL'
        model.mkdir(parents=True)
        included = model / 'ordinary.inc'
        included.write_text('      INTEGER UNUSED_PARAMETER\n')
        self.replace(fixture[0], 'coef_construction_1.f', '      LOGICAL WL_ENTRY_INEXACT',
                     "      INCLUDE 'ordinary.inc'\n      LOGICAL WL_ENTRY_INEXACT")
        self.assertTrue(self.validate(fixture, include_roots=(str(model),)))
        included.write_text('      X=WL(1,0,2,2)\n')
        self.assertFalse(self.validate(fixture, include_roots=(str(model),)))

    def test_model_source_is_scanned_and_cannot_be_omitted_from_manifest(self):
        export = self.work / 'model_export'
        fixture = self.fixture(directory=export / 'SubProcesses/P0_test')
        model = export / 'Source/MODEL'
        model.mkdir(parents=True)
        source = model / 'couplings.f'
        source.write_text('      SUBROUTINE COUPLINGS()\n      END\n')
        manifest = [str(path) for path in fixture[0].glob('*.f')]
        self.assertTrue(self.validate(fixture, source_manifest=manifest + [str(source)]))
        self.assertFalse(self.validate(fixture, source_manifest=manifest))
        source.write_text('      SUBROUTINE COUPLINGS()\n'
                          '      COMMON/TEST_WL/OTHER,OTHER_PL\n'
                          '      CALL HIDDEN(OTHER)\n      END\n')
        self.assertFalse(self.validate(fixture, source_manifest=manifest + [str(source)]))

    def test_unrecognized_added_units_and_gnumakefile_rejected(self):
        for name in ('custom.f', 'custom.f90', 'GNUmakefile'):
            with self.subTest(name=name):
                fixture = self.fixture()
                (fixture[0] / name).write_text('      SUBROUTINE CUSTOM()\n      END\n')
                self.assertFalse(self.validate(fixture))

    def test_build_definitions_cannot_inject_sources_or_unsafe_flags(self):
        definitions = (
            'FFLAGS+=-ffast-math\n',
            'FFLAGS += -ffast-math\\\n -O2\n',
            'evil.o: hidden.inc\n\t$(FC) -x f77 -c hidden.inc -o evil.o\n',
            'EXTRA=$(shell cat hidden.mk)\n',
        )
        for index, text in enumerate(definitions):
            with self.subTest(definitions=text):
                export = self.work / ('build_export_%d' % index)
                fixture = self.fixture(directory=export / 'SubProcesses/P0_test')
                (export / 'Source').mkdir()
                (export / 'Source/make_opts').write_text(text)
                self.assertFalse(self.validate(fixture))

    def test_modified_merger_or_create_consumer_rejected(self):
        changes = (
            ('WL(I,J,I)*CONST', 'WL(I,J,1)*CONST'),
            ('OUT(J)=OUT(J)+WL(I,J,I)*CONST', 'OUT(J)=(OUT(J)+WL(I,J,I))*CONST'),
            ('DO I=1,LCUT_SIZE', 'DO I=LCUT_SIZE,1,-1'),
            ('CALL TEST_MERGE_WL(LOOP_WF', 'CALL HIDDEN(LOOP_WF'),
            ('      CALL TEST_MERGE_WL(LOOP_WF',
             '      CALL HIDDEN(LOOP_WF)\n      CALL TEST_MERGE_WL(LOOP_WF'),
            ('      CALL TEST_MERGE_WL(LOOP_WF',
             '      EQUIVALENCE(LOOP_WF,ESCAPE)\n      CALL TEST_MERGE_WL(LOOP_WF'),
        )
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.fixture()
                self.replace(fixture[0], 'polynomial.f', old, new)
                self.assertFalse(self.validate(fixture))

    def test_matching_template_cannot_authorize_an_extra_wavefunction_read(self):
        template = (Path(__file__).resolve().parents[3] /
                    'madgraph/iolibs/template_files/loop_optimized/polynomial.inc')
        original_read_text = Path.read_text
        marker = '      DO I=1,NAMPSO'
        mutation = '      CFTOT=LOOP_WF(1,0,2)\n' + marker

        def changed_template(path, *args, **kwargs):
            text = original_read_text(path, *args, **kwargs)
            if path.resolve() == template:
                self.assertIn(marker, text)
                return text.replace(marker, mutation, 1)
            return text

        with mock.patch.object(Path, 'read_text', new=changed_template):
            # Rendering and validating against the same mutated template makes
            # template equality insufficient to establish diagonal-only reads.
            fixture = self.fixture()
            self.assertIn('CFTOT=LOOP_WF(1,0,2)',
                          (fixture[0] / 'polynomial.f').read_text())
            self.assertFalse(self.validate(fixture))

    def test_source_manifest_cannot_add_external_unit(self):
        fixture = self.fixture()
        root = fixture[0]
        outside = self.work / 'outside.f'
        outside.write_text('      SUBROUTINE OUTSIDE()\n      END\n')
        manifest = [path.name for path in root.glob('*.f')] + [str(outside)]
        self.assertFalse(self.validate(fixture, source_manifest=manifest))

    def higher_fixture(self, process_rank=4, mother_rank=3, prefix='TEST_'):
        """Use the actual generator, independent of the proof's monomial code."""
        from madgraph.various.q_polynomial import FortranPolynomialRoutines
        from madgraph.various.q_polynomial import get_number_of_coefs_for_rank

        root, prefix, _ = self.fixture(prefix)
        count = get_number_of_coefs_for_rank(process_rank)
        self.replace(root, 'coef_specs.inc', 'LOOPMAXCOEFS=15',
                     'LOOPMAXCOEFS=%d' % count)
        (root / 'loop_max_coefs.inc').write_text('C Bounds are in coef_specs.inc.\n')
        constants = FortranPolynomialRoutines(
            process_rank, updater_max_rank=1, sub_prefix=prefix).write_polynomial_constant_module()
        rendered = root / 'rendered_constants.inc'
        with FortranWriter(str(rendered)) as writer:
            writer.writelines(constants)
        polynomial = root / 'polynomial.f'
        text = polynomial.read_text()
        end = text.index('\n', text.index('END MODULE')) + 1
        polynomial.write_text(rendered.read_text() + text[end:] +
            '      SUBROUTINE ' + prefix + 'LOOKUP()\n'
            '      USE ' + prefix + 'POLYNOMIAL_CONSTANTS\n'
            '      INTEGER NEW_POSITION,L,M\n'
            '      NEW_POSITION=COMB_COEF_POS(L,M)\n'
            '      END\n')
        rendered.unlink()
        output = mother_rank + 1
        for path in root.glob('*.f'):
            path.write_text(path.read_text().replace('NLOOPWAVEFUNCS=2',
                                                     'NLOOPWAVEFUNCS=%d' % output))
        batch = root / 'coef_construction_1.f'
        text = batch.read_text()
        calls = ''.join(
            '      CALL %sUPDATE_WL_%d_1(WL(1,0,1,%d),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,%d))\n' % (prefix, rank, rank, rank + 1)
            for rank in range(mother_rank))
        calls += (
            '      CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
            '      CALL %sDP_TERMINAL_WL_%d_1(WL(1,0,1,%d),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,%d),WL_ENTRY_INEXACT)\n'
            '      CALL %sCREATE_LOOP_COEFS(WL(1,0,1,%d),%d,4,1,1,1,1,1)\n'
            '      END\n' % (prefix, mother_rank, mother_rank, output,
                             prefix, output, output))
        batch.write_text(text[:text.index('      CALL ')] + calls)
        expected = {output: dict(loop_mother_rank=mother_rank, vertex_rank=1,
                                loop_mother_number=mother_rank, lcut_size=4,
                                in_size=4, out_size=4, out=output)}
        return root, prefix, expected

    def test_higher_terminal_process_ranks_three_through_six(self):
        for rank in (3, 4, 5, 6):
            for mother in (2, 3):
                for prefix in ('TEST_', ''):
                    with self.subTest(process_rank=rank, mother_rank=mother, prefix=prefix):
                        fixture = self.higher_fixture(rank, mother, prefix)
                        self.assertEqual(self.validate(fixture), mother + 1 <= rank)

    def test_higher_gate_does_not_change_single_rank_one_validation(self):
        # The original 1x1 fixture has no combination table and still passes.
        self.assertTrue(self.validate(self.fixture()))
        # The original path also permits storage above the new higher-rank gate.
        fixture = self.fixture()
        self.replace(fixture[0], 'coef_specs.inc', 'LOOPMAXCOEFS=15', 'LOOPMAXCOEFS=330')
        self.replace(fixture[0], 'coef_specs.inc', 'VERTEXMAXCOEFS=5', 'VERTEXMAXCOEFS=15')
        self.assertTrue(self.validate(fixture))

    def test_higher_gate_rejects_other_process_ranks_and_vertex_storage(self):
        for rank in (2, 7):
            with self.subTest(rank=rank):
                self.assertFalse(self.validate(self.higher_fixture(rank, 2)))
        for old, new in (('LOOPMAXCOEFS=70', 'LOOPMAXCOEFS=71'),
                         ('VERTEXMAXCOEFS=5', 'VERTEXMAXCOEFS=15')):
            with self.subTest(new=new):
                fixture = self.higher_fixture()
                self.replace(fixture[0], 'coef_specs.inc', old, new)
                self.assertFalse(self.validate(fixture))

    def test_higher_combination_table_layout_and_values_are_literal(self):
        from madgraph.loop.dp_terminal_proof import _compact

        changes = (
            ('INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)',
             'INTEGER COMB_COEF_POS(1:LOOPMAXCOEFS,0:4)'),
            ('INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)',
             'INTEGER COMB_COEF_POS(0:LOOPMAXCOEFS-1,1:5)'),
            ('DATACOMB_COEF_POS(0,0:4)/0,1,2,3,4/',
             'DATA COMB_COEF_POS(0,0:4)/1,1,2,3,4/'),
            ('DATACOMB_COEF_POS(14,0:4)/14,31,32,33,34/',
             'DATA COMB_COEF_POS(14,0:4)/14,31,32,33,35/'),
            ('DATACOMB_COEF_POS(34,0:4)/34,66,67,68,69/',
             'DATA COMB_COEF_POS(34,0:4)/34,66,67,68,70/'),
            # Even the unused final rows must retain their canonical values.
            ('DATACOMB_COEF_POS(69,0:4)/69,122,123,124,125/',
             'DATA COMB_COEF_POS(69,0:4)/69,122,123,124,124/'),
        )
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.higher_fixture()
                path = fixture[0] / 'polynomial.f'
                lines = path.read_text().splitlines(True)
                match = next(line for line in lines if _compact(line) == old)
                self.replace(fixture[0], path.name, match, '      ' + new + '\n')
                self.assertFalse(self.validate(fixture))
        for mode in ('missing', 'duplicate'):
            with self.subTest(mode=mode):
                fixture = self.higher_fixture()
                path = fixture[0] / 'polynomial.f'
                line = next(line for line in path.read_text().splitlines(True)
                            if _compact(line).startswith('DATACOMB_COEF_POS(14,'))
                self.replace(fixture[0], path.name, line, '' if mode == 'missing' else line + line)
                self.assertFalse(self.validate(fixture))

    def test_higher_combination_table_cannot_be_written_shadowed_or_escape(self):
        statements = (
            'COMB_COEF_POS(0,0)=1', 'CALL HIDDEN(COMB_COEF_POS)',
            'INTEGER COMB_COEF_POS(0:69,0:4)', 'COMMON/ESCAPE/COMB_COEF_POS',
            'EQUIVALENCE(COM B_COEF_POS,ESCAPE)', 'P=C_LOC(COMB_COEF_POS)',
            'USE TEST_POLYNOMIAL_CONSTANTS,ONLY:ALIAS=>COMB_COEF_POS',
            'NEW_POSITION=COMB_COEF_POS(L,HIDDEN())',
        )
        for statement in statements:
            with self.subTest(statement=statement):
                fixture = self.higher_fixture()
                (fixture[0] / 'born_matrix.f').write_text(
                    '      SUBROUTINE OTHER()\n'
                    '      USE TEST_POLYNOMIAL_CONSTANTS\n'
                    '      ' + statement + '\n      END\n')
                self.assertFalse(self.validate(fixture))

    def test_higher_combination_readers_bind_only_known_modules(self):
        for addition in ('USE EVIL', 'USE EVIL, ONLY: OPERATOR(==)',
                         'USE IEEE_ARITHMETIC', 'INTERFACE HIDDEN',
                         'INTEGER COMB_COEF_POS(0:69,0:4)'):
            with self.subTest(addition=addition):
                fixture = self.higher_fixture()
                self.replace(fixture[0], 'polynomial.f', '      INTEGER NEW_POSITION,L,M',
                             '      ' + addition + '\n      INTEGER NEW_POSITION,L,M')
                self.assertFalse(self.validate(fixture))
        fixture = self.higher_fixture()
        self.replace(fixture[0], 'polynomial.f', '      INTEGER NEW_POSITION,L,M',
                     '      USE, INTRINSIC :: IEEE_ARITHMETIC\n'
                     '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n'
                     '      INTEGER NEW_POSITION,L,M')
        self.assertTrue(self.validate(fixture))

    def test_higher_capture_rank_and_consumer_checks_remain_binding(self):
        changes = (
            ('      CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n', ''),
            ('DP_TERMINAL_WL_3_1(', 'DP_TERMINAL_WL_2_1('),
            ('DP_TERMINAL_WL_3_1(', 'DP_TERMINAL_WL_3_0('),
            ('CREATE_LOOP_COEFS(WL(1,0,1,4),4,4,',
             'CREATE_LOOP_COEFS(WL(1,0,1,4),3,4,'),
            ('WL(1,0,1,3),4,COEFS,4,4,', 'WL(1,0,1,3),4,COEFS,1,4,'),
            ('      END\n', '      CALL HIDDEN(WL(1,0,1,4))\n      END\n'),
        )
        for old, new in changes:
            with self.subTest(mutation=new):
                fixture = self.higher_fixture()
                self.replace(fixture[0], 'coef_construction_1.f', old, new)
                self.assertFalse(self.validate(fixture))
        fixture = self.higher_fixture()
        self.replace(fixture[0], 'coef_construction_1.f', '      CALL TEST_CREATE_LOOP_COEFS(',
                     '      CONTINUE\n      CALL TEST_CREATE_LOOP_COEFS(')
        self.assertTrue(self.validate(fixture))

    def test_higher_reader_internal_bit_views_cannot_alias_the_table(self):
        fixture = self.higher_fixture()
        marker = '      NEW_POSITION=COMB_COEF_POS(L,M)\n      END\n'
        helpers = (
            '      NEW_POSITION=COMB_COEF_POS(L,M)\n'
            '      CONTAINS\n'
            '      LOGICAL FUNCTION SAFE(X)\n'
            '      REAL*8 X\n'
            '      INTEGER*8 ONE\n'
            '      PARAMETER(ONE=1)\n'
            '      ASSOCIATE(WORDS=>TRANSFER(X,ONE))\n'
            '      SAFE=WORDS.EQ.0\n'
            '      END ASSOCIATE\n'
            '      END FUNCTION SAFE\n'
            '      END\n')
        self.replace(fixture[0], 'polynomial.f', marker, helpers)
        self.assertTrue(self.validate(fixture))
        self.replace(fixture[0], 'polynomial.f', 'WORDS=>TRANSFER(X,ONE)',
                     'WORDS=>COMB_COEF_POS')
        self.assertFalse(self.validate(fixture))

    def test_private_references_outside_certified_graph_are_rejected(self):
        for make_fixture, pair in ((self.fixture, '1_1'), (self.higher_fixture, '3_1')):
            references = (
                'CALL TEST_DP_TERMINAL_WL_%s(A,4,B,4,4,C,.TRUE.)' % pair,
                'X=TEST_DP_TERMINAL_WL_%s(A)' % pair,
                'EXTERNAL TEST_DP_TERMINAL_WL_%s' % pair,
                'CALL CALLBACK(TEST_DP_TERMINAL_WL_%s)' % pair,
                'PROCEDURE(TEST_DP_TERMINAL_WL_%s),POINTER::P' % pair,
                'CALL TEST_DP_TERMINAL_WL_4_1(A,4,B,4,4,C,.TRUE.)',
                'CALL TEST_DP_TERMINAL_WL_3_0(A,4,B,4,4,C,.TRUE.)',
            )
            for statement in references:
                with self.subTest(pair=pair, statement=statement):
                    fixture = make_fixture()
                    (fixture[0] / 'born_matrix.f').write_text(
                        '      SUBROUTINE OTHER(A,B,C)\n'
                        '      COMPLEX*16 A(4,0:69,4),B(4,0:4,4),C(4,0:69,4)\n'
                        '      ' + statement + '\n      END\n')
                    self.assertFalse(self.validate(fixture))
            with self.subTest(pair=pair, source='polynomial.f'):
                fixture = make_fixture()
                with (fixture[0] / 'polynomial.f').open('a') as output:
                    output.write('      SUBROUTINE OTHER(A,B,C)\n'
                                 '      CALL TEST_DP_TERMINAL_WL_%s(A,4,B,4,4,C,.TRUE.)\n'
                                 '      END\n' % pair)
                self.assertFalse(self.validate(fixture))

    def test_private_definition_names_and_included_call_replay_are_binding(self):
        fixture = self.higher_fixture()
        with (fixture[0] / 'polynomial.f').open('a') as output:
            output.write('      SUBROUTINE TEST_DP_TERMINAL_WL_3_1(A)\n'
                         '      END SUBROUTINE TEST_DP_TERMINAL_WL_3_1\n')
        self.assertTrue(self.validate(fixture))
        with (fixture[0] / 'polynomial.f').open('a') as output:
            output.write('      SUBROUTINE TEST_DP_TERMINAL_WL_3_1(A)\n      END\n')
        self.assertFalse(self.validate(fixture))
        fixture = self.higher_fixture()
        with (fixture[0] / 'polynomial.f').open('a') as output:
            output.write('      SUBROUTINE TEST_DP_TERMINAL_WL_4_1(A)\n      END\n')
        self.assertFalse(self.validate(fixture))
        fixture = self.higher_fixture()
        (fixture[0] / 'born_matrix.f').write_text(
            "      INCLUDE 'coef_construction_1.f'\n")
        self.assertFalse(self.validate(fixture))


if __name__ == '__main__':
    unittest.main()
