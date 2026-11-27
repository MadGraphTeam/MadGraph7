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
"""The SVG drawings stay light: smooth curls and waves, tenth-of-a-pixel
relative coordinates, gzip-compressed files.

The C++ outputs (madmatrix standalone, mg7) draw every diagram of every
subprocess; with a dense polyline per gluon (10 points per half-curl, two
decimals, absolute coordinates) and the drawings written twice in plain text,
p p > 5j weighed 1.6 GB of drawings for 21 MB of sources."""

from __future__ import absolute_import

import gzip
import math
import os
import re
import shutil
import tempfile

import madgraph.iolibs.drawing_svg as draw_svg
import tests.unit_tests as unittest


def _segments(path):
    """(start, [(c1, c2, end)]) in absolute coordinates, read back from the
    `d` attribute written by _bezier_path (M, then relative c commands)."""
    d = re.search(r'd="([^"]*)"', path).group(1)
    head, *cmds = d.split('c')
    x, y = (float(v) for v in head[1:].split())
    start, segs = (x, y), []
    for cmd in cmds:
        v = [float(t) for t in cmd.split()]
        c1 = (x + v[0], y + v[1])
        c2 = (x + v[2], y + v[3])
        end = (x + v[4], y + v[5])
        segs.append((c1, c2, end))
        x, y = end
    return start, segs


def _max_deviation(start, segs, exact):
    """Largest distance from the Bézier curve to the densely sampled exact
    curve."""
    worst, p0 = 0.0, start
    for c1, c2, p3 in segs:
        for k in range(1, 20):
            t = k / 20.
            x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * c1[0]
                 + 3 * (1 - t) * t * t * c2[0] + t ** 3 * p3[0])
            y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * c1[1]
                 + 3 * (1 - t) * t * t * c2[1] + t ** 3 * p3[1])
            worst = max(worst, min(math.hypot(x - a, y - b)
                                   for a, b in exact))
        p0 = p3
    return worst


class TestSvgCurls(unittest.TestCase):

    X1, Y1, X2, Y2 = 30., 100., 200., 160.

    def test_gluon_is_two_segments_per_half_curl_and_follows_the_curve(self):
        x1, y1, x2, y2 = self.X1, self.Y1, self.X2, self.Y2
        dist, xl, yl, xt, yt = draw_svg._basis(x1, y1, x2, y2)
        fn = max(2, 2 * round(dist / (2 * draw_svg._Fr)))
        n = draw_svg._Fnopoints * fn

        def exact(i):
            t, a = i / n, i * math.pi / draw_svg._Fnopoints
            return (x1 + (x2 - x1) * t + xt * (1 - math.cos(a))
                    + xl * math.sin(a),
                    y1 + (y2 - y1) * t + yt * (1 - math.cos(a))
                    + yl * math.sin(a))
        path = draw_svg._svg_gluon(x1, y1, x2, y2)
        start, segs = _segments(path)
        self.assertEqual(len(segs), 2 * fn)
        dense = [exact(n * k / 4000.) for k in range(4001)]
        # well below the 1.5-pixel stroke (rounding to 0.1 px included)
        self.assertLess(_max_deviation(start, segs, dense), 0.2)
        # the path ends where the line ends: relative steps do not drift
        self.assertAlmostEqual(segs[-1][2][0], x2, delta=0.06)
        self.assertAlmostEqual(segs[-1][2][1], y2, delta=0.06)
        # an eighth of the former 10-points-per-half-curl polyline
        self.assertLess(len(path), 2500)

    def test_photon_follows_the_curve(self):
        x1, y1, x2, y2 = self.X1, self.Y1, self.X2, self.Y2
        dist, xl, yl, xt, yt = draw_svg._basis(x1, y1, x2, y2)
        fn = max(1, round(dist / (2 * draw_svg._Fr)))
        n = draw_svg._Fnopoints * fn

        def exact(i):
            t, w = i / (2 * n), math.sin(i * math.pi / draw_svg._Fnopoints)
            return (x1 + (x2 - x1) * t + xt * w / 2,
                    y1 + (y2 - y1) * t + yt * w / 2)
        start, segs = _segments(draw_svg._svg_photon(x1, y1, x2, y2))
        self.assertEqual(len(segs), 4 * fn)
        dense = [exact(2 * n * k / 4000.) for k in range(4001)]
        self.assertLess(_max_deviation(start, segs, dense), 0.2)

    def test_coordinates_have_one_decimal(self):
        text = (draw_svg._svg_gluon(self.X1, self.Y1, self.X2, self.Y2)
                + draw_svg._svg_fermion(self.X1, self.Y1, self.X2, self.Y2)
                + draw_svg._svg_ghost(self.X1, self.Y1, self.X2, self.Y2))
        self.assertFalse(re.search(r'\d\.\d\d', text), text[:300])
        self.assertNotIn('-0 ', text)


class TestMultiSvgFiles(unittest.TestCase):
    """diagrams.svgz and diagrams.json.gz: gzip files, the same bytes for the
    same diagrams (no timestamp in the gzip header)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='svgdraw_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write(self, name):
        import madgraph.interface.master_interface as master_interface
        cmd = master_interface.MasterCmd()
        cmd.exec_cmd('import model sm')
        cmd.exec_cmd('generate u u~ > g a --use_crossing=False')
        amp = cmd._curr_amps[0]
        stem = os.path.join(self.tmpdir, name)
        draw_svg.MultiSVGDiagramDrawer(
            amp.get('diagrams'), stem,
            model=amp.get('process').get('model'), amplitude=True).draw()
        return stem

    def test_files_are_gzip_and_reproducible(self):
        first, second = self._write('a'), self._write('b')
        for plain in ('.svg', '.json'):     # no uncompressed copy left
            self.assertFalse(os.path.exists(first + plain), plain)
        for ext in ('.svgz', '.json.gz'):
            with open(first + ext, 'rb') as fa, open(second + ext, 'rb') as fb:
                self.assertEqual(fa.read(), fb.read(), ext)
        with gzip.open(first + '.svgz', 'rt') as fsock:
            svg = fsock.read()
        self.assertTrue(svg.startswith('<?xml'))
        self.assertIn('</svg>', svg)
        import json
        with gzip.open(first + '.json.gz', 'rt') as fsock:
            data = json.load(fsock)
        self.assertEqual([d['diagram_number'] for d in data],
                         list(range(1, len(data) + 1)))
        self.assertTrue(all(d['svg'].startswith('<?xml') for d in data))


if __name__ == '__main__':
    unittest.main()
