"""The way IOTests compare and store their reference files (tests/IOTests.py):
comment-insensitive comparison of Fortran sources, the 'once' target rule and
the shared storage of identical reference files."""
import os
import tempfile
import unittest

import tests.IOTests as IOTests

Manager = IOTests.IOTestManager


class IOTestReferencesTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='iotest-references-')
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def write(self, folder, test, name, content):
        directory = os.path.join(self.root, folder, test)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, name)
        if os.path.islink(file_path):
            os.remove(file_path)
        with open(file_path, 'w') as stream:
            stream.write(content)
        return file_path

    def read(self, file_path):
        with open(file_path) as stream:
            return stream.read()

    def shared(self):
        shared = os.path.join(self.root, Manager._shared_folder)
        return sorted(os.listdir(shared)) if os.path.isdir(shared) else []

    def test_iotest_fortran_comments_are_ignored(self):
        reference = "C     old comment\n      X = 1\n\n      CALL F(X)\n"
        output = ("C     new, longer comment\nc     and a second line\n"
                  "      X = 1\n      ! free-form comment\n      CALL F(X)\n")
        manager = Manager()
        self.assertTrue(manager.reference_accepts(reference, output, 'a.f'))
        self.assertTrue(manager.reference_accepts(reference, output, 'b.inc'))
        # Code changes still fail, and other file types keep comment lines.
        self.assertFalse(manager.reference_accepts(
            reference, output.replace('X = 1', 'X = 2'), 'a.f'))
        self.assertFalse(manager.reference_accepts(reference, output, 'card.dat'))

    def test_iotest_once_rule_keeps_first_file_of_each_name(self):
        files = ['P0_b/sborn_sf.f', 'P0_a/sborn_sf.f', 'P0_c/maxparticles.inc',
                 'P0_a/maxparticles.inc']
        self.assertEqual(sorted(Manager.beyond_first_of_name(files)),
                         ['P0_b/sborn_sf.f', 'P0_c/maxparticles.inc'])

    def test_iotest_identical_references_are_shared(self):
        a = self.write('F1', 'T1', 'x.f', 'same\n')
        b = self.write('F2', 'T2', 'x.f', 'same\n')
        c = self.write('F2', 'T2', 'y.f', 'unique\n')
        Manager.share_identical_references(self.root)
        self.assertEqual(self.shared(), ['F1%T1%x.f'])
        for file_path in (a, b):
            self.assertTrue(os.path.islink(file_path))
            self.assertFalse(os.path.isabs(os.readlink(file_path)))
            self.assertEqual(self.read(file_path), 'same\n')
        self.assertFalse(os.path.islink(c))

        # A change applied to every user edits the shared file only.
        self.write('F1', 'T1', 'x.f', 'changed\n')
        self.write('F2', 'T2', 'x.f', 'changed\n')
        Manager.share_identical_references(self.root)
        self.assertEqual(self.shared(), ['F1%T1%x.f'])
        self.assertTrue(os.path.islink(a) and os.path.islink(b))
        self.assertEqual(self.read(b), 'changed\n')

        # A change in one user only makes both regular files again.
        self.write('F2', 'T2', 'x.f', 'different\n')
        Manager.share_identical_references(self.root)
        self.assertEqual(self.shared(), [])
        self.assertFalse(os.path.islink(a) or os.path.islink(b))
        self.assertEqual((self.read(a), self.read(b)), ('changed\n', 'different\n'))


if __name__ == '__main__':
    unittest.main()
