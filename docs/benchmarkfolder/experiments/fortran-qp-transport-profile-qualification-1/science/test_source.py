"""Pure source-transform tests. No Fortran compiler or numerical invocation."""
import importlib.util
from pathlib import Path
import re
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('private_profile_prepare',HERE/'prepare.py')
prep=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rendered=prep.render()

    def test_exact_frozen_render(self):
        for name,text in self.rendered.items():
            self.assertEqual(text,(HERE/'source-preview'/name).read_text())

    def test_learning_stream_unchanged(self):
        original=prep.SOURCE['check_sa.f'][0].read_text()
        changed=self.rendered['check_sa.f']
        begin="      CALL SETPARA('param_card.dat')"
        end='      PREVIOUS_SEQUENCE=0\n'
        self.assertEqual(original[original.index(begin):original.index(end)+len(end)],
                         changed[changed.index(begin):changed.index(end)+len(end)])
        self.assertNotIn("IF(TRIM(COMMAND).NE.'RUN')",changed)
        self.assertNotIn("'DONE '",changed)
        self.assertIn('DO QP_POINT=1,16,15',changed)

    def test_original_numerical_assignments_and_returns_preserved(self):
        for name in ['TIRUST_interface.f','TIR_interface.f','mtirust.f90']:
            original=prep.SOURCE[name][0].read_text()
            changed=self.rendered[name]
            # Existing arithmetic, branch guards and numerical RETURN lines
            # are preserved verbatim, including all continuation lines.
            for line in original.splitlines():
                stripped=line.strip()
                if stripped.startswith(('RESULT(', 'RES(', 'TIRCOEFS(', 'INVARIANTS(',
                    'PREFIX_SUM=', 'P_TMP(', 'coefs(', 'read(text,', 'write(text,',
                    'IF(.NOT.', 'if (.not.', 'IF(STATUS.', 'if (status', 'RETURN')):
                    self.assertIn(line,changed)

    def test_no_rust_or_physics_policy_change(self):
        self.assertEqual(set(self.rendered),{'TIRUST_interface.f','TIR_interface.f','check_sa.f','mtirust.f90'})
        self.assertNotIn('ml_tirust_profile_', '\n'.join(self.rendered.values()))
        self.assertIn('CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)',self.rendered['check_sa.f'])

    def test_fixed_line_width(self):
        for name,text in self.rendered.items():
            if name.endswith('.f'):
                self.assertFalse([(i,len(line)) for i,line in enumerate(text.splitlines(),1)
                                  if len(line)>132 and not line.startswith('C')])

    def test_no_scalar_transport_change(self):
        changed=self.rendered['mtirust.f90']
        for name in ['pack','unpack']:
            pattern=r'  subroutine '+name+r'\(.*?  end subroutine\n'
            self.assertEqual(re.search(pattern,prep.SOURCE['mtirust.f90'][0].read_text(),re.S).group(),
                             re.search(pattern,changed,re.S).group())


if __name__ == '__main__':
    unittest.main()
