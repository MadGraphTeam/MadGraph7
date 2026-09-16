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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Runs the Python examples embedded in the MadSpace user documentation.

Each ``docs/source/madspace/examples/*.rst`` page carries its example as one or
more ``.. code-block:: python`` directives, with prose in between. This module
extracts those blocks (in document order), concatenates them into one script
and runs it as a subprocess, so the documentation cannot silently drift from a
working example. There is one test method per example page.

Every test method self-skips when its dependencies (madspace, an LHAPDF set, a
GPU, ...) are unavailable, so it can be run on a laptop that only has a subset
installed. CI instead selects test methods by name, one per job, so that a
missing dependency there is a real failure rather than a silent skip --
``test_manager.py`` treats a skipped test the same as a failed one.

Run locally with e.g.::

    ./tests/test_manager.py test_madspace_example_simple_mappings -pA -t0 -l INFO
"""

from __future__ import absolute_import

import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

from madgraph import MG5DIR

pjoin = os.path.join

_EXAMPLES_DIR = pjoin(MG5DIR, 'docs', 'source', 'madspace', 'examples')
_MADSPACE_INSTALL = pjoin(MG5DIR, 'madspace', 'install')

_CODE_BLOCK_RE = re.compile(r'^(?P<indent>[ \t]*)\.\.[ \t]+code-block::[ \t]*python[ \t]*$')


def extract_python_blocks(rst_path):
    """Return the bodies of every ``.. code-block:: python`` directive in
    *rst_path*, in document order, dedented. A block whose directive carries
    ``:class: no-run`` is skipped, so a page can show a snippet it does not
    want executed."""
    with open(rst_path) as f:
        lines = f.read().splitlines()

    blocks = []
    i = 0
    while i < len(lines):
        m = _CODE_BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        base_indent = len(m.group('indent'))
        i += 1

        options = []
        while i < len(lines) and lines[i].strip().startswith(':'):
            options.append(lines[i].strip())
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        body = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                body.append('')
                i += 1
                continue
            if len(line) - len(line.lstrip()) <= base_indent:
                break
            body.append(line)
            i += 1
        while body and not body[-1].strip():
            body.pop()

        if not any(':class: no-run' in o for o in options):
            blocks.append(textwrap.dedent('\n'.join(body)))

    return blocks


def has_madspace():
    try:
        import madspace
        return hasattr(madspace, 'PhaseSpaceMapping')
    except ImportError:
        return False


def run_doc_example(test, page_name):
    """Extract the python blocks of ``examples/<page_name>.rst``, concatenate
    them into one script, and run it as a subprocess. Fails the test if the
    script raises or if the page has no python blocks at all (a moved or
    renamed page must fail loudly, not silently pass)."""
    rst_path = pjoin(_EXAMPLES_DIR, page_name + '.rst')
    if not os.path.isfile(rst_path):
        test.fail('documentation page not found: %s' % rst_path)

    blocks = extract_python_blocks(rst_path)
    test.assertTrue(blocks, 'no python code-block found in %s' % rst_path)
    script = '\n'.join(blocks)

    env = dict(os.environ)
    if os.path.isdir(_MADSPACE_INSTALL):
        env['PYTHONPATH'] = os.pathsep.join(
            [_MADSPACE_INSTALL] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = pjoin(tmp_dir, page_name.replace('-', '_') + '.py')
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=tmp_dir, env=env, capture_output=True, text=True,
        )

    test.assertEqual(
        result.returncode, 0,
        'example %s exited with code %s\n--- stdout ---\n%s\n--- stderr ---\n%s'
        % (page_name, result.returncode, result.stdout, result.stderr),
    )


class TestMadSpaceExamples(unittest.TestCase):

    def test_madspace_example_simple_mappings(self):
        """docs/source/madspace/examples/simple-mappings.rst -- constructing
        and calling a Rambo/Chili PhaseSpaceMapping. Needs madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'simple-mappings')

    def test_madspace_example_cuts(self):
        """docs/source/madspace/examples/cuts.rst -- building fiducial cuts
        and folding them into a PhaseSpaceMapping. Needs madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'cuts')

    def test_madspace_example_diagram_mapping(self):
        """docs/source/madspace/examples/diagram-mapping.rst -- a mapping
        for one Feynman diagram, t-channel plus resonant s-channel. Needs
        madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'diagram-mapping')

    def test_madspace_example_integration_order(self):
        """docs/source/madspace/examples/integration-order.rst -- multiple
        s-channel on-shell configurations and the propagator integration
        order, plus printing a Topology. Needs madspace only."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'integration-order')


if __name__ == '__main__':
    unittest.main()
