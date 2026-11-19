"""Legacy and contextual MadLoop backend statistics remain distinguishable."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from xml.dom.minidom import parseString

from madgraph import MG5DIR, MadGraph5Error
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.madevent.sum_html import RunStatistics, decode_madloop_backend_usage
from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd


class TestNinjaRSStatistics(unittest.TestCase):
    @staticmethod
    def xml(extra=""):
        return parseString("""<statistics>
          <u_return_code>0,0,7,0,0,0,0,0,11,0</u_return_code>
          <t_return_code>0,10,8,0,0</t_return_code>
          <h_return_code>0,0,10,8,0</h_return_code>
          <average_time>0.001</average_time>
          <cumulated_time>0.018</cumulated_time>
          <max_prec>1e-8</max_prec><min_prec>1e-12</min_prec>
          <n_evals>18</n_evals>%s</statistics>""" % extra).documentElement

    def test_old_xml_is_unchanged(self):
        stats = RunStatistics()
        stats.load_statistics(self.xml())
        self.assertEqual(stats["PJFry_usage"], 7)
        self.assertEqual(stats["Ninja_QP_usage"], 11)
        self.assertEqual(stats["NinjaRS_DP_usage"], 0)
        self.assertEqual(stats["NinjaRS_QP_usage"], 0)

    def test_mixed_xml_and_aggregation(self):
        stats = RunStatistics()
        stats.load_statistics(self.xml(
            "<ninjars_dp_usage>9</ninjars_dp_usage>"
            "<ninjars_qp_usage>5</ninjars_qp_usage>"))
        self.assertEqual(stats["PJFry_usage"], 2)
        self.assertEqual(stats["Ninja_QP_usage"], 2)
        self.assertEqual(stats["NinjaRS_DP_usage"], 9)
        self.assertEqual(stats["NinjaRS_QP_usage"], 5)
        self.assertIn("NinjaRS_DP", stats.nice_output("test"))
        old = RunStatistics()
        old.load_statistics(self.xml())
        stats.aggregate_statistics(old)
        self.assertEqual(stats["PJFry_usage"], 9)
        self.assertEqual(stats["Ninja_QP_usage"], 13)
        self.assertEqual(stats["NinjaRS_DP_usage"], 9)

    def test_optional_counters_and_non_mutating_decoder(self):
        units = [0, 0, 7, 0, 0, 0, 0, 0, 11, 0]
        for _ in range(2):
            usage = decode_madloop_backend_usage(units, 9, 5)
            self.assertEqual(usage['Ninja_QP'], 2)
            self.assertEqual(usage['PJFry'], 2)
            self.assertEqual(sum(usage.values()), sum(units))
            self.assertEqual(units[8], 11)
            self.assertEqual(units[2], 7)
        stats = RunStatistics()
        for extra, expected in [
                ('<ninjars_dp_usage>9</ninjars_dp_usage>', (2, 7, 9, 0)),
                ('<ninjars_qp_usage>5</ninjars_qp_usage>', (11, 2, 0, 5)),
                ('', (11, 7, 0, 0))]:
            # Reusing the reader must also clear optional counters absent from
            # the next file, without altering the old-file interpretation.
            stats.load_statistics(self.xml(extra))
            self.assertEqual(tuple(stats[key] for key in (
                'Ninja_QP_usage', 'PJFry_usage', 'NinjaRS_DP_usage',
                'NinjaRS_QP_usage')), expected)

    legacy_log = """Satistics from MadLoop:
 Total points tried: 18
 Stability unknown: 0
 Stable PS point: 18
 Unstable PS point (and rescued): 0
 Exceptional PS point (unstable and not rescued): 0
 Double precision used: 10
 Quadruple precision used: 8
 Initialization phase-space points: 0
 Unknown return code (100): 0
 Unknown return code (10): 0
 #Unit 2 = 7
 #Unit 8 = 11
"""

    def test_nlo_logs_old_and_mixed(self):
        log = self.legacy_log
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            for extra, expected in [
                ("", False),
                ("#NinjaRS DP = 9\n#NinjaRS QP = 5\n", True),
            ]:
                path.write_text(log + extra)
                _, report = aMCatNLOCmd.compile_advanced_stats(
                    SimpleNamespace(me_dir=directory),
                    [str(path)], [], "")
                self.assertEqual("NinjaRS (double precision)" in report, expected)
                self.assertEqual("NinjaRS (quadruple precision)" in report, expected)
                if expected:
                    self.assertRegex(report, r"PJFry\+\+\s+2(?:\n|$)")
                    self.assertRegex(report, r"Ninja \(quadruple precision\)\s+2(?:\n|$)")

    def test_format_two_xml_uses_unit_two_without_alias_subtraction(self):
        xml = self.xml('<return_code_version>2</return_code_version>'
                       '<ninjars_dp_usage>9</ninjars_dp_usage>'
                       '<ninjars_qp_usage>5</ninjars_qp_usage>')
        xml.getElementsByTagName('u_return_code')[0].firstChild.data = '0,0,14,0,0,0,0,0,11,0'
        stats = RunStatistics()
        stats.load_statistics(xml)
        self.assertEqual(stats['PJFry_usage'], 0)
        self.assertEqual(stats['Ninja_QP_usage'], 11)
        self.assertEqual(stats['NinjaRS_DP_usage'], 9)
        self.assertEqual(stats['NinjaRS_QP_usage'], 5)
        old = RunStatistics()
        old.load_statistics(self.xml())
        stats.aggregate_statistics(old)
        self.assertEqual(stats['PJFry_usage'], 7)
        self.assertEqual(stats['Ninja_QP_usage'], 22)

    def test_initialization_precision_counts_remain_distinct(self):
        old = RunStatistics()
        new = RunStatistics()
        for stats, extra in [(old, ''), (new,
                '<return_code_version>2</return_code_version>'
                '<ninjars_dp_usage>3</ninjars_dp_usage>'
                '<ninjars_qp_usage>4</ninjars_qp_usage>')]:
            xml = self.xml(extra)
            xml.getElementsByTagName('t_return_code')[0].firstChild.data = '0,5,7,12,15'
            stats.load_statistics(xml)
            self.assertEqual(stats['DP_init_usage'], 12)
            self.assertEqual(stats['QP_init_usage'], 15)
        new.aggregate_statistics(old)
        self.assertEqual(new['DP_init_usage'], 24)
        self.assertEqual(new['QP_init_usage'], 30)

    def test_unknown_format_and_inconsistent_new_counters_are_rejected(self):
        units = [0, 0, 14, 0, 0, 0, 0, 0, 11, 0]
        for version, dp, qp in [(3, 9, 5), (2, 9, 4), (2, -1, 15)]:
            with self.subTest(version=version, dp=dp, qp=qp):
                with self.assertRaises(MadGraph5Error):
                    decode_madloop_backend_usage(units, dp, qp, version)
        self.assertEqual(units[2], 14)

    def test_mixed_version_logs_are_decoded_before_aggregation(self):
        old = self.legacy_log + '#NinjaRS DP = 9\n#NinjaRS QP = 5\n'
        new = self.legacy_log.replace('#Unit 2 = 7', '#Unit 2 = 14')
        new += '#ReturnCodeVersion = 2\n#NinjaRS DP = 9\n#NinjaRS QP = 5\n'
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('old.log', 'new.log')]
            for path, content in zip(paths, (old, new)):
                path.write_text(content)
            _, report = aMCatNLOCmd.compile_advanced_stats(
                SimpleNamespace(me_dir=directory), list(map(str, paths)), [], '')
            for label, count in [('PJFry++', 2), ('Ninja (quadruple precision)', 13),
                                 ('NinjaRS (double precision)', 18),
                                 ('NinjaRS (quadruple precision)', 10)]:
                self.assertRegex(report, re.escape(label) + r'\s+' + str(count) + r'(?:\n|$)')
            for version in ('8', '-2', 'invalid'):
                paths[1].write_text(new.replace('#ReturnCodeVersion = 2',
                                               '#ReturnCodeVersion = ' + version))
                with self.subTest(version=version):
                    with self.assertRaises(MadGraph5Error):
                        aMCatNLOCmd.compile_advanced_stats(
                            SimpleNamespace(me_dir=directory), [str(paths[1])], [], '')


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMadLoopStatisticsRoundTrip(unittest.TestCase):
    """Run actual Fortran tally/writer blocks through both Python readers."""

    # Current unit 2 is only NinjaRS. Include synthetic H/T combinations to
    # ensure the counters use T, independently of the controller's current H
    # choices: 312 denotes DP precision, whereas 222 denotes QP use.
    samples = [(100, 1), (211, 2), (213, 4), (214, 5),
               (215, 6), (216, 7), (217, 8), (328, 9), (329, 10),
               (212, 11), (232, 12), (312, 13), (322, 14), (342, 15), (222, 16)]
    expected = {'CutTools_DP': 2, 'PJFry': 0, 'IREGI': 4, 'Golem': 5,
                'Samurai': 6, 'Ninja': 7, 'COLLIER': 8, 'Ninja_QP': 9,
                'CutTools_QP': 10, 'NinjaRS_DP': 36, 'NinjaRS_QP': 45}

    def run_producer(self, work, tally, writer):
        codes = [code for code, count in self.samples for _ in range(count)]
        source = """
      program test_statistics
      implicit none
      integer i,j,u,t,h,return_code,ret_code,outUnit
      integer NINJARS_DP_USAGE,NINJARS_QP_USAGE
      integer U_RETURN_CODES(0:9),T_RETURN_CODES(0:9)
      integer H_RETURN_CODES(0:9),n1(0:9)
      integer ntot,nsun,nsps,nups,neps,nddp,nqdp,nini,n100,n10
      NINJARS_DP_USAGE=0
      NINJARS_QP_USAGE=0
      U_RETURN_CODES=0
      T_RETURN_CODES=0
      H_RETURN_CODES=0
      n1=0
      ntot=0
      nsun=0
      nsps=0
      nups=0
      neps=0
      nddp=0
      nqdp=0
      nini=0
      n100=0
      n10=0
      outUnit=6
      do i=1,%d
      read(*,*) return_code
      ret_code=return_code
""" % len(codes)
        source += tally + '\n      enddo\n' + writer + '\n      end\n'
        with FortranWriter(str(work / 'statistics.f')) as output:
            output.writelines(source)
        compiled = subprocess.run(
            ['gfortran', '-ffixed-line-length-132', '-fcheck=all',
             'statistics.f', '-o', 'statistics'], cwd=work,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(compiled.returncode, 0, compiled.stdout)
        run = subprocess.run([str(work / 'statistics')], cwd=work,
                             input='\n'.join(map(str, codes)) + '\n', text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(run.returncode, 0, run.stdout)
        return run.stdout

    def test_lo_grouped_and_ungrouped_xml_round_trip(self):
        root = Path(MG5DIR)
        source = (root / 'Template/LO/Source/dsample.f').read_text()
        writer = source[source.index("      write(outUnit,33) '<u_return_code>'"):
                        source.index("      write(outUnit,33) '<t_return_code>'")]
        writer += source[source.index('33    FORMAT('):source.index(
            '      end subroutine', source.index('33    FORMAT('))]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in ('matrix_loop_induced_madevent.inc',
                         'matrix_loop_induced_madevent_group.inc'):
                with self.subTest(producer=name):
                    source = (root / 'madgraph/iolibs/template_files' / name).read_text()
                    tally = source[source.index('\tU = mod(RETURN_CODE,10)'):
                                   source.index('\tMAX_PREC = max(PREC, MAX_PREC)')]
                    output = self.run_producer(work, tally, writer)
                    # All backend data comes from the production Fortran writer;
                    # unrelated timing/stability fields use the existing fixture.
                    xml = TestNinjaRSStatistics.xml()
                    old = xml.getElementsByTagName('u_return_code')[0]
                    xml.removeChild(old)
                    generated = parseString('<counts>' + output + '</counts>')
                    for node in list(generated.documentElement.childNodes):
                        xml.appendChild(node)
                    stats = RunStatistics()
                    stats.load_statistics(xml)
                    self.assertEqual({key: stats[key + '_usage']
                                      for key in self.expected}, self.expected)

    def test_nlo_fixed_order_and_matched_log_round_trip(self):
        root = Path(MG5DIR) / 'Template/NLO/SubProcesses'
        source = (root / 'BinothLHA.f').read_text()
        tally = source[source.index('c Update the statistics using the MadLoop return code'):
                       source.index('c Write out the unstable, non-rescued')]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for name in ('driver_mintFO.f', 'driver_mintMC.f'):
                with self.subTest(producer=name):
                    source = (root / name).read_text()
                    start = source.index('      if (ntot.ne.0) then')
                    # Extract the statistics conditional itself, including its
                    # nested bucket conditional, independently of whatever
                    # timing/reporting statements follow it in the driver.
                    writer_lines = []
                    depth = 0
                    for line in source[start:].splitlines(keepends=True):
                        writer_lines.append(line)
                        statement = line[6:].strip().lower()
                        if re.match(r'if\s*\(.*\)\s*then\b', statement):
                            depth += 1
                        elif re.match(r'end\s*if\b', statement):
                            depth -= 1
                            if depth == 0:
                                break
                    else:
                        self.fail('Unterminated MadLoop statistics block in ' + name)
                    writer = ''.join(writer_lines)
                    output = self.run_producer(work, tally, writer)
                    path = work / 'log.txt'
                    path.write_text(output)
                    _, report = aMCatNLOCmd.compile_advanced_stats(
                        SimpleNamespace(me_dir=directory), [str(path)], [], '')
                    for label, count in [
                            ('Not identified (CTModeRun != -1)', 1),
                            ('CutTools (double precision)', 2),
                            ('IREGI', 4), ('Golem95', 5), ('Samurai', 6),
                            ('Ninja (double precision)', 7), ('COLLIER', 8),
                            ('Ninja (quadruple precision)', 9),
                            ('CutTools (quadruple precision)', 10),
                            ('NinjaRS (double precision)', 36),
                            ('NinjaRS (quadruple precision)', 45)]:
                        self.assertRegex(report, re.escape(label) +
                                         r'\s+' + str(count) + r'(?:\n|$)')
                    self.assertNotIn('PJFry++', report)
                    old_path = work / 'historical.log'
                    old_path.write_text(TestNinjaRSStatistics.legacy_log)
                    _, mixed = aMCatNLOCmd.compile_advanced_stats(
                        SimpleNamespace(me_dir=directory),
                        [str(path), str(old_path)], [], '')
                    self.assertRegex(mixed, r'PJFry\+\+\s+7(?:\n|$)')
                    self.assertRegex(mixed, r'Ninja \(quadruple precision\)\s+20(?:\n|$)')


if __name__ == "__main__":
    unittest.main()
