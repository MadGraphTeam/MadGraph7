################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Integrating a selection of the subprocesses.

The compiled channel functions refer to a matrix element by its position in
the context, so a run over some of the subprocesses has to load them at their
position in the full list, and name their channels by their global index. Both
are what lets the gridpacks of several such runs be merged into one.
"""
from __future__ import absolute_import
import types
import unittest
from unittest import mock

from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

# Importing launch without a compiled madspace starts the madspace build, and
# test_manager imports every unit test module while collecting: check first.
launch = None
if mg7_bootstrap.madspace_is_installed():
    try:
        import madgraph.iolibs.template_files.mg7.launch as launch
    except Exception as error:
        launch = None
        import_error = error

SUBPROCESS_DATA = [{'me_path': 'lib/me%d.so' % i} for i in range(5)]
# merge groups list their subprocesses in another order than the output does
MERGED_DATA = [{'subprocesses': [3, 0]}, {'subprocesses': [1]},
               {'subprocesses': [4, 2]}]


def process_stub(merged, selection=()):
    process = types.SimpleNamespace(
        subprocess_data=SUBPROCESS_DATA,
        merged_subprocess_data=MERGED_DATA if merged else None,
        custom_function_paths=[['cut', 'cut.py', {}]],
        run_card={'phasespace': {'subprocess_selection': list(selection)}},
        compile_matrix_elements=lambda: ['simd_128'],
    )
    process.me_layout = types.MethodType(launch.MadgraphProcess.me_layout, process)
    return process


@unittest.skipIf(launch is None, 'madspace not available')
class TestSubprocessSelection(unittest.TestCase):

    def test_layout_follows_the_subprocess_order(self):
        paths, offsets = process_stub(False).me_layout()
        self.assertEqual(paths, [d['me_path'] for d in SUBPROCESS_DATA])
        self.assertEqual(offsets, [0, 1, 2, 3, 4])

    def test_layout_follows_the_merge_groups(self):
        """The matrix elements are loaded group by group, so that is the order
        data.json has to list them in."""
        paths, offsets = process_stub(True).me_layout()
        self.assertEqual(paths, ['lib/me%d.so' % i for i in (3, 0, 1, 4, 2)])
        self.assertEqual(offsets, [0, 2, 3])

    def init_subprocesses(self, merged, selection):
        process = process_stub(merged, selection)
        created = []
        with mock.patch.object(
                launch, 'MadgraphSubprocess',
                side_effect=lambda *args, **kwargs: created.append((args, kwargs))
                or types.SimpleNamespace(subproc_id=args[2])):
            launch.MadgraphProcess.init_subprocesses(process)
        return process, created

    def test_selection_keeps_global_ids_and_positions(self):
        process, created = self.init_subprocesses(False, [3, 1])
        self.assertEqual([args[2] for args, _ in created], [1, 3])
        # after the one custom function
        self.assertEqual([kwargs['me_index'] for _, kwargs in created], [2, 4])

    def test_merged_selection_positions(self):
        process, created = self.init_subprocesses(True, [2])
        self.assertEqual([args[2] for args, _ in created], [2])
        self.assertEqual(created[0][1]['me_index'], 1 + 3)

    def test_no_selection_selects_everything(self):
        process, created = self.init_subprocesses(False, [])
        self.assertEqual([args[2] for args, _ in created], [0, 1, 2, 3, 4])

    def test_selection_beyond_the_subprocesses(self):
        with self.assertRaises(ValueError):
            self.init_subprocesses(True, [3])
