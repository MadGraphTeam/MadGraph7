"""The f2py recipes of the MadLoop standalone makefile suit both f2py backends.

The meson backend (numpy >= 1.26, and the only one on Python >= 3.12) differs
from distutils in ways that make a recipe fail to build, or build a module
that does not import:
- it names a meson variable after each -l library ('.' becomes '_'), so
  -lstdc++, -lc++ and a library named after an output with a '-' stop the
  build;
- it links .a files given as sources before its objects, where they resolve
  nothing;
- it compiles copies of the sources in a temporary directory, so INCLUDE files
  are not found and files with the same name overwrite each other;
- it takes the Fortran compiler from FC and ignores --f77exec.
matrix2py, allmatrix2py and all_matrix2py are therefore all built the same
way: a shared library, linked by the C++ compiler, carries the objects and the
C++ runtime, and f2py only wraps it, through LDFLAGS and one -l.
The recipes are read with ``make -n`` in a stub output, so neither a compiler
nor numpy is needed.
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from madgraph import MG5DIR

SUBPROCESSES = Path(MG5DIR) / 'Template/loop_material/StandAlone/SubProcesses'
# numpy.f2py._backends._meson: "<lib with '.' replaced by '_'> = declare_dependency(...)"
MESON_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
# a restricted model names its output this way: the '-' must not reach -l
OUTPUT = 'PROC_sm-no_b_mass_0'
PLATFORMS = (('Linux', '-lstdc++', 'so'), ('Darwin', '-lc++', 'dylib'))


@unittest.skipUnless(shutil.which('make'), 'make is required')
class LoopF2PYMakefileTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='loop-f2py-makefile-')
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)

    def write_output(self, uname, stdlib, dylibext, extra_tir_libs=''):
        """A loop standalone output with Ninja and two subprocesses, whose
        sources and objects exist so that make -n prints the f2py recipes."""
        out = self.output / uname / OUTPUT
        (out / 'Source').mkdir(parents=True)
        (out / 'Source/make_opts').write_text(
            'FC=gfortran\nCXX=c++\nF2PY=f2py\nSTDLIB=%s\nUNAME=%s\nlibext=a\n'
            'dylibext=%s\nDYNLIBFLAG=%s\n'
            % (stdlib, uname, dylibext,
               '-dynamiclib' if uname == 'Darwin' else '-shared -fPIC'))
        (out / 'lib').mkdir()
        for lib in ('dhelas', 'model', 'cts', 'iregi'):
            (out / 'lib' / ('lib%s.a' % lib)).touch()
        ninja = out / 'HEPTools/lib'
        ninja.mkdir(parents=True)
        for lib in ('ninja', 'avh_olo'):
            (ninja / ('lib%s.a' % lib)).touch()
        sub = out / 'SubProcesses'
        sub.mkdir()
        shutil.copy(SUBPROCESSES / 'makefile', sub / 'makefile')
        definitions = (SUBPROCESSES / 'MadLoop_makefile_definitions.inc').read_text()
        (sub / 'MadLoop_makefile_definitions').write_text(definitions % {
            'link_tir_libs': '-liregi -L%s/ -lninja -L%s/ -lavh_olo %s' % (
                ninja, ninja, extra_tir_libs),
            'tir_libs': '$(LIBDIR)libiregi.$(libext) %s/libninja.$(libext) '
                        '%s/libavh_olo.$(libext)' % (ninja, ninja),
            'tir_dylibs': '%s/libninja.$(dylibext)' % ninja,
            'tir_include': '-I %s/../include' % ninja,
            'prefix': 'P', 'doto': '%.o', 'dotf': '%.f',
            'rpaths_libs': '-Wl,-rpath,%s ' % ninja})
        common = ['MadLoopParamReader.f', 'MadLoopCommons.f', 'mg5_citation.f']
        files = common + ['all_matrix.f', 'f2py_wrapper.f']
        for pdir in ('P0_uux_z', 'P1_ddx_z'):
            (sub / pdir).mkdir()
            (sub / pdir / 'makefile').symlink_to('../makefile')
            files += ['%s/%s' % (pdir, name) for name in common + ['f2py_wrapper.f']]
        for name in files:
            (sub / name).touch()
        for name in files:
            (sub / name).with_suffix('.o').touch()
        return sub

    def dry_run(self, directory, target):
        result = subprocess.run(
            ['make', '-n', 'SHELL=%s' % (shutil.which('bash') or '/bin/sh'), target],
            cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout.splitlines()

    def f2py_calls(self, lines):
        calls = []
        for line in lines:
            words = shlex.split(line)
            if 'f2py' not in words:
                continue
            start = words.index('f2py')
            env = dict(word.split('=', 1) for word in words[:start])
            calls.append((env, words[start + 1:], line))
        self.assertTrue(calls, '\n'.join(lines))
        return calls

    def library_link(self, lines, library, dylibext):
        """The C++ compiler line that links lib<library>.<dylibext>."""
        name = 'lib%s.%s' % (library, dylibext)
        links = [shlex.split(line) for line in lines
                 if line.startswith('c++ ') and name in shlex.split(line)]
        self.assertEqual(len(links), 1, '\n'.join(lines))
        return links[0]

    def check_build(self, lines, uname, stdlib, dylibext, sources):
        """Check the f2py calls of a recipe, and return the environment of the
        build call with its run path made independent of the directory."""
        calls = self.f2py_calls(lines)
        for env, args, line in calls:
            for arg in args:
                self.assertNotIn(arg, ('-lc++', '-lstdc++'), line)
                if arg.startswith('-l'):
                    self.assertRegex(arg[2:].replace('.', '_'), MESON_IDENTIFIER,
                                     'meson cannot name this library:\n' + line)
                self.assertFalse(arg.startswith('-Wl'), 'f2py rejects linker options:\n' + line)
                self.assertFalse(arg.endswith(('.a', '.o')),
                                 'the library carries the objects:\n' + line)
        (env, args, line), = [call for call in calls if '-c' in call[1]]
        self.assertEqual([arg for arg in args if arg.endswith(('.f', '.f90', '.pyf'))],
                         sources, line)
        self.assertEqual(env.get('FC'), 'gfortran', line)
        ldflags = env.get('LDFLAGS', '').split()
        self.assertNotIn(stdlib, ldflags, 'the library carries the C++ runtime:\n' + line)
        here = ldflags[0]
        self.assertTrue(here.startswith('-Wl,-rpath,'), line)
        (library,) = [arg[2:] for arg in args if arg.startswith('-l')]
        link = self.library_link(lines, library, dylibext)
        self.assertIn(stdlib, link)
        self.assertIn('mg5_citation.o', link)
        return {key: value.replace(here[len('-Wl,-rpath,'):], '<HERE>')
                for key, value in env.items()}, link

    def test_recipes_match_all_matrix2py(self):
        for uname, stdlib, dylibext in PLATFORMS:
            with self.subTest(uname=uname):
                sub = self.write_output(uname, stdlib, dylibext)
                reference, liball = self.check_build(
                    self.dry_run(sub, 'all_matrix2py.so'), uname, stdlib, dylibext,
                    ['f2py_wrapper.f'])
                self.assertIn('all_matrix.o', liball)

                lines = self.dry_run(sub, 'allmatrix2py.so')
                env, link = self.check_build(lines, uname, stdlib, dylibext,
                                             ['allmatrix2py.pyf'])
                self.assertEqual(env, reference)
                self.assertEqual(link, liball)
                (signature,) = [args for _, args, _ in self.f2py_calls(lines) if '-h' in args]
                for wrapper in ('P0_uux_z/f2py_wrapper.f', 'P1_ddx_z/f2py_wrapper.f'):
                    self.assertIn(wrapper, signature)

                pdir = sub / 'P0_uux_z'
                lines = self.dry_run(pdir, 'matrix2py.so')
                env, link = self.check_build(lines, uname, stdlib, dylibext,
                                             ['matrix2py.pyf'])
                self.assertEqual(env, reference)
                self.assertIn('f2py_wrapper.o', link)
                (signature,) = [args for _, args, _ in self.f2py_calls(lines) if '-h' in args]
                self.assertIn('f2py_wrapper.f', signature)

    def test_ldflags_carry_run_paths_but_no_loop_library(self):
        """meson links LDFLAGS into its compiler sanity check and runs it in its
        build directory. TIRust with internal dependencies links a shared
        library found only through a relative run path, so LDFLAGS must keep
        the run path but not the library, which the wrapped library carries."""
        relative = {'Linux': '-Wl,-rpath,\\$$ORIGIN:\\$$ORIGIN/../lib',
                    'Darwin': '-Wl,-rpath,@loader_path -Wl,-rpath,@loader_path/../lib'}
        for uname, stdlib, dylibext in PLATFORMS:
            with self.subTest(uname=uname):
                run_path = relative[uname]
                libraries = ['/gcc/libgcc.a', '-lgcc_s', '-ltirust', '-lmadloop_tirust', '-ldl']
                sub = self.write_output(
                    uname, stdlib, dylibext,
                    ' '.join(libraries[:2] + ['-L$(LIBDIR)'] + libraries[2:] + [run_path]))
                # shlex keeps the backslash of \$ inside double quotes; a shell drops it
                expected = run_path.replace('\\$$', '$').split()
                loop_libraries = libraries + ['-lcts', '-liregi', '-lninja', '-lavh_olo']
                for directory, target in ((sub, 'all_matrix2py.so'),
                                          (sub, 'allmatrix2py.so'),
                                          (sub / 'P0_uux_z', 'matrix2py.so')):
                    lines = self.dry_run(directory, target)
                    (env, args, line), = [call for call in self.f2py_calls(lines)
                                          if '-c' in call[1]]
                    ldflags = env['LDFLAGS'].replace('\\$', '$').split()
                    for flag in expected:
                        self.assertIn(flag, ldflags, line)
                    self.assertEqual([flag for flag in ldflags if flag in loop_libraries],
                                     [], line)
                    (library,) = [arg[2:] for arg in args if arg.startswith('-l')]
                    link = self.library_link(lines, library, dylibext)
                    for flag in libraries:
                        self.assertIn(flag, link)


if __name__ == '__main__':
    unittest.main()
