"""Host colour-sum defaults and incompatible-build protection."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock

from madgraph.iolibs import export_cpp
from madmatrix import output


class TestHostBlasBuild(unittest.TestCase):
    def policy(self, flags):
        exporter = object.__new__(output.ProcessExporterMadMatrix)
        with mock.patch.object(export_cpp.ProcessExporterCPP,
                               'get_makefile_replace_dict', return_value={'model': 'test'}), \
             mock.patch.object(exporter.oneprocessclass,
                               'blas_available_flags', return_value=flags):
            return exporter.get_makefile_replace_dict(None)

    def makefile(self, directory, flags='-lblas'):
        template = (Path(__file__).resolve().parents[3] / 'madgraph/iolibs/'
                    'template_files/madmatrix/madmatrix.mk').read_text()
        path = Path(directory) / 'makefile'
        path.write_text(template % self.policy(flags) +
                        '\n.PHONY: probe\nprobe:\n'
                        '\t@echo "$(CPPBLAS)|$(TAG)|$(CPPBLASLIBFLAGS)|$(CPPBLASCXXFLAGS)"\n')
        return path

    def make(self, directory, target='probe', setting=None, question=False):
        command = ['make', '--no-print-directory', '-s', 'BACKEND=scalar',
                   'FPTYPE=d', 'HELINL=0', 'HRDCOD=0', 'USEBUILDDIR=1',
                   'CXX=true', target]
        if setting is not None:
            command.append('CPPBLAS=' + setting)
        if question:
            command.append('-q')
        return subprocess.run(command, cwd=directory, capture_output=True,
                              text=True, timeout=30)

    def test_availability_alone_does_not_select_host_blas(self):
        for flags, expected in [('', 'hasNoBlas'), ('-lblas', 'hasNoBlas'),
                                ('-framework Accelerate', 'hasBlas')]:
            with self.subTest(flags=flags):
                policy = self.policy(flags)
                self.assertEqual(policy['cpp_blas_default'], expected)
                self.assertEqual(policy['cpp_blas_libflags'], flags)

    def test_explicit_override_keeps_both_paths_available(self):
        with mock.patch.object(output.ProcessExporterMadMatrix.oneprocessclass,
                               'blas_is_available', return_value=True):
            exporter = output.ProcessExporterMadMatrix.oneprocessclass
            self.assertTrue(exporter.cpp_blas_wanted_for(120))
            self.assertFalse(exporter.cpp_blas_wanted_for(24))
        for flags, default in [('-lblas', 'hasNoBlas'),
                               ('-framework Accelerate', 'hasBlas')]:
            with tempfile.TemporaryDirectory() as directory:
                self.makefile(directory, flags)
                for setting in (None, 'hasBlas', 'hasNoBlas'):
                    result = self.make(directory, setting=setting)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    selected, tag, links, defines = result.stdout.strip().splitlines()[-1].split('|')
                    self.assertEqual(selected, setting or default)
                    self.assertIn(selected, tag)
                    # make joins the flags with a separating blank.
                    self.assertEqual(links.strip(), flags if selected == 'hasBlas' else '')
                    self.assertEqual(bool(defines), selected == 'hasBlas')

    def test_switching_host_blas_rejects_stale_objects(self):
        for first, second in [('hasBlas', 'hasNoBlas'),
                              ('hasNoBlas', 'hasBlas')]:
            with self.subTest(first=first), tempfile.TemporaryDirectory() as directory:
                self.makefile(directory)
                initial = self.make(directory, setting=first)
                self.assertEqual(initial.returncode, 0, initial.stderr)
                tag = initial.stdout.strip().splitlines()[-1].split('|')[1]
                build = Path(directory) / 'build.scalar_d'
                build.mkdir()
                (build / ('.build.' + tag)).touch()
                changed = self.make(directory, setting=second)
                self.assertEqual(changed.returncode, 0, changed.stderr)
                new_tag = changed.stdout.strip().splitlines()[-1].split('|')[1]
                result = self.make(directory, 'build.scalar_d/.build.' + new_tag,
                                   setting=second)
                self.assertNotEqual(result.returncode, 0,
                                    'Changing the colour implementation reused old objects')
                self.assertIn('old builds exist', result.stdout + result.stderr)

    def test_nlo_dispatch_uses_strict_flags_without_changing_lo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root/'SubProcesses/PMM_B1_R1'
            process.mkdir(parents=True)
            for relative in ('backend/cpu', 'backend/common', 'src/rambo', 'src'):
                folder = root/relative
                folder.mkdir(parents=True, exist_ok=True)
                (folder/'fixture.h').write_text('// Dependency fixture\n')
            for name in ('CPPProcess.cc', 'CPPProcess.h'):
                (process/name).write_text('// Compile-command fixture\n')
            (root/'src/fixture.cc').write_text('// Shared model fixture\n')
            source_template = (Path(__file__).resolve().parents[3] /
                'madgraph/iolibs/template_files/madmatrix/madmatrix_src.mk').read_text()
            (root/'src/makefile').write_text(source_template % {'model': 'test'})
            makefile = self.makefile(process)
            with makefile.open('a') as stream:
                stream.write('\n.DEFAULT_GOAL := compile_probe\n'
                             '.PHONY: compile_probe\n'
                             'compile_probe: $(BUILDDIR)/CPPProcess.o\n'
                             '\t$(MAKE) -C ../../src $(MADMATRIX_BUILDDIR)/fixture.o\n')
            command = ['make', '--no-print-directory', '-n', 'BACKEND=scalar',
                       'FPTYPE=d', 'HELINL=0', 'HRDCOD=0', 'USEBUILDDIR=1',
                       'CXX=true']
            lo = subprocess.run(command, cwd=process, capture_output=True,
                                text=True, timeout=30)
            self.assertEqual(lo.returncode, 0, lo.stderr)

            native = root/'SubProcesses/P0_test'
            native.mkdir()
            exporter = object.__new__(output.ProcessExporterMadMatrixNLOReal)
            exporter._write_bridge_config({
                'library_model': 'test', 'nexternal': 2,
                'split_order_names': ['QCD'],
                'real_matrix_elements': [{
                    'library_process_id': process.name,
                    'process_fingerprint': 'fixture',
                    'local_squared_orders': [[2]], 'local_flavours': [[2, -2]],
                    'backend_capabilities': {'cuda': True},
                }],
            }, str(native))
            nlo = subprocess.run(command + ['-f', 'nlo_real.mk', 'libraries'],
                                 cwd=native, capture_output=True,
                                 text=True, timeout=30)
            self.assertEqual(nlo.returncode, 0, nlo.stderr)
            for result, strict in ((lo, False), (nlo, True)):
                compile_lines = [shlex.split(line) for line in result.stdout.splitlines()
                                 if any(' -c '+name+' ' in line for name in
                                        ('CPPProcess.cc', 'fixture.cc'))]
                self.assertEqual(len(compile_lines), 2, result.stdout)
                for flags in compile_lines:
                    self.assertEqual('-fno-fast-math' in flags, strict)
                    self.assertEqual('-ffast-math' in flags, not strict)
                    self.assertIn('-DMADMATRIX_FPTYPE_DOUBLE', flags)
                if strict:
                    self.assertIn('_strict1', result.stdout)
                else:
                    self.assertNotIn('_strict1', result.stdout)

            # Reusing fast-math objects must fail instead of silently relinking.
            probe = self.make(process)
            tag = probe.stdout.strip().splitlines()[-1].split('|')[1]
            build = process/'build.scalar_d'
            build.mkdir()
            (build/('.build.'+tag)).touch()
            stale = subprocess.run(command[0:2] + ['-s', 'BACKEND=scalar',
                'FPTYPE=d', 'HELINL=0', 'HRDCOD=0', 'USEBUILDDIR=1',
                'CXX=true', 'NLO_REAL_STRICT=1',
                'build.scalar_d/.build.'+tag+'_strict1'], cwd=process,
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(stale.returncode, 0)
            self.assertIn('old builds exist', stale.stdout + stale.stderr)

    def test_arithmetic_header_change_invalidates_process_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root / 'SubProcesses/PMM_B1_R1'
            arithmetic = root / 'src/Arithmetics'
            process.mkdir(parents=True)
            arithmetic.mkdir(parents=True)
            for relative in ('backend/cpu', 'backend/common', 'src/rambo', 'src'):
                folder = root/relative
                folder.mkdir(parents=True, exist_ok=True)
                (folder/'fixture.h').write_text('// Dependency fixture\n')
            self.makefile(process)
            sources = [process/'CPPProcess.cc', process/'CPPProcess.h',
                       arithmetic/'Invariant.h']
            for source in sources:
                source.write_text('// Dependency fixture\n')
            result = self.make(process)
            self.assertEqual(result.returncode, 0, result.stderr)
            tag = result.stdout.strip().splitlines()[-1].split('|')[1]
            build = process/'build.scalar_d'
            build.mkdir()
            marker = build/('.build.'+tag)
            marker.touch()
            for source in sources + list(root.rglob('fixture.h')) + [marker]:
                os.utime(source, (1600000000, 1600000000))
            target = build/'CPPProcess.o'
            target.touch()
            os.utime(target, (1600000010, 1600000010))
            fresh = self.make(process, 'build.scalar_d/CPPProcess.o', question=True)
            self.assertEqual(fresh.returncode, 0, fresh.stderr)
            os.utime(arithmetic/'Invariant.h', (1600000020, 1600000020))
            stale = self.make(process, 'build.scalar_d/CPPProcess.o', question=True)
            self.assertEqual(stale.returncode, 1, stale.stderr)
