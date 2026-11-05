"""IO snapshots retain helper identity independently of allocator suffix width."""
import io
import unittest

import tests.IOTests as io_tests


class TestFlavourIdentifierComparison(unittest.TestCase):
    short = '      CALL FFV(W(1),FLV_9,FLV_10,W(5))\n'
    wrapped = ('      CALL FFV(W(1),FLV_133,FLV_134,W(5)\n'
               '     $       )\n')

    def compare(self, actual, reference):
        io_tests.IOTestManager().assertFileContains(io.StringIO(actual), reference)

    def test_allocator_width_does_not_change_logical_call(self):
        self.compare(self.short, self.wrapped)
        self.compare(self.wrapped, self.short)

    def test_missing_parenthesis_is_detected(self):
        with self.assertRaises(AssertionError):
            self.compare(self.wrapped.replace('     $       )\n', ''), self.short)

    def test_changed_helper_reuse_is_detected(self):
        with self.assertRaises(AssertionError):
            self.compare(self.wrapped.replace('FLV_134', 'FLV_133'), self.short)

    def test_other_source_formatting_remains_compared(self):
        with self.assertRaises(AssertionError):
            self.compare('      CALL FOO(X,\n     $ Y)\n', '      CALL FOO(X,Y)\n')
