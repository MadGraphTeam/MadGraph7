"""Checks the compiler cannot make: calls of external NLO routines in other
files, and card values that FKSParamReader must reject."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


def fixed_form_statements(path):
    """Statements of a fixed-form source, continuation lines joined and
    comments dropped."""
    statements = []
    for line in path.read_text(errors='replace').splitlines():
        if not line.strip() or line[:1] in 'cC*!' or line.lstrip().startswith('!'):
            continue
        if len(line) > 5 and line[5] not in ' 0' and statements:
            statements[-1] += line[6:]
        else:
            statements.append(line)
    return [s.split('!')[0] for s in statements]


def call_arguments(statement, name):
    """Top-level arguments of 'call name(...)' in a statement, or None."""
    match = re.search(r'(?i)\bcall\s+%s\s*\(' % name, statement)
    if not match:
        return None
    depth, start, arguments = 1, match.end(), []
    for position in range(match.end(), len(statement)):
        character = statement[position]
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth == 0:
                arguments.append(statement[start:position].strip())
                return arguments
        elif character == ',' and depth == 1:
            arguments.append(statement[start:position].strip())
            start = position + 1
    raise ValueError('Unbalanced call of %s: %s' % (name, statement))


class TestNLOCallArguments(unittest.TestCase):
    def test_every_add_wgt_call_passes_type_orders_and_three_weights(self):
        calls = 0
        for path in sorted(TEMPLATE.glob('*.f')):
            for statement in fixed_form_statements(path):
                arguments = call_arguments(statement, 'add_wgt')
                if arguments is None:
                    continue
                calls += 1
                self.assertEqual(len(arguments), 5, '%s: %s' % (path.name, statement.strip()))
                # The coupling orders are an integer array, never a weight.
                self.assertRegex(arguments[1], r'^[A-Za-z_]\w*$', '%s: %s' % (path.name, statement.strip()))
        self.assertGreater(calls, 10)


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestFKSParamRanges(unittest.TestCase):
    def test_precision_virtual_at_run_time_is_checked(self):
        with tempfile.TemporaryDirectory(prefix='mg7-fks-params-') as temporary:
            work = Path(temporary)
            (work / 'orders.inc').write_text('      integer nsplitorders\n      parameter(nsplitorders=1)\n')
            (work / 'check.f90').write_text(
                'program check\nuse FKSParams\nimplicit none\n'
                "call FKSParamReader('card.dat',.false.,.true.)\n"
                "print *,'READ',PrecisionVirtualAtRunTime\nend program\n")
            result = subprocess.run(['gfortran', '-I', str(work), str(TEMPLATE / 'FKSParams.f90'),
                                     str(work / 'check.f90'), '-o', str(work / 'check')],
                                    cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for value, accepted in (('-1d0', True), ('1d-3', True), ('-5d0', False)):
                with self.subTest(value=value):
                    (work / 'card.dat').write_text('#PrecisionVirtualAtRunTime\n%s\n' % value)
                    result = subprocess.run([str(work / 'check')], cwd=work, capture_output=True,
                                            text=True, timeout=30)
                    output = result.stdout + result.stderr
                    if accepted:
                        self.assertIn('READ', output)
                    else:
                        self.assertIn('PrecisionVirtualAtRunTime must be >= -1.0d0', output)
                        self.assertNotIn('READ', output)


if __name__ == '__main__':
    unittest.main()
