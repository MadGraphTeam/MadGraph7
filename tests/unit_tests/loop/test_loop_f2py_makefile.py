"""The f2py recipes of the MadLoop standalone makefile suit both f2py backends.

The meson backend (numpy >= 1.26, and the only one on Python >= 3.12) differs
from distutils in ways that make a recipe fail to build, or build a module
that does not import:
- it names a meson variable after each -l library ('.' becomes '_'), so
  -lstdc++ and -lc++ stop the build;
- it links .a files given as sources before its objects, where they resolve
  nothing;
- it compiles copies of the sources (and copies the objects) in a temporary
  directory, so INCLUDE files are found only through -I, and files with the
  same name overwrite each other;
- it takes the Fortran compiler from FC and ignores --f77exec.
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


@unittest.skipUnless(shutil.which('make'), 'make is required')
class LoopF2PYMakefileTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='loop-f2py-makefile-')
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)

    def write_output(self, uname, stdlib):
        """A loop standalone output with Ninja and two subprocesses, whose
        sources and objects exist so that make -n prints the f2py recipes."""
        out = self.output / uname
        (out / 'Source').mkdir(parents=True)
        (out / 'Source/make_opts').write_text(
            'FC=gfortran\nF2PY=f2py\nSTDLIB=%s\nUNAME=%s\nlibext=a\ndylibext=%s\n'
            % (stdlib, uname, 'dylib' if uname == 'Darwin' else 'so'))
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
            'link_tir_libs': '-liregi -L%s/ -lninja -L%s/ -lavh_olo' % (ninja, ninja),
            'tir_libs': '$(LIBDIR)libiregi.$(libext) %s/libninja.$(libext) '
                        '%s/libavh_olo.$(libext)' % (ninja, ninja),
            'tir_dylibs': '%s/libninja.$(dylibext)' % ninja,
            'tir_include': '-I %s/../include' % ninja,
            'prefix': 'P', 'doto': '%.o', 'dotf': '%.f',
            'rpaths_libs': '-Wl,-rpath,%s ' % ninja})
        files = ['MadLoopParamReader.f', 'MadLoopCommons.f', 'all_matrix.f',
                 'mg5_citation.f']
        for pdir in ('P0_uux_z', 'P1_ddx_z'):
            (sub / pdir).mkdir()
            (sub / pdir / 'makefile').symlink_to('../makefile')
            files += ['%s/f2py_wrapper.f' % pdir, '%s/mg5_citation.f' % pdir]
        for name in files:
            (sub / name).touch()
        for name in files:
            (sub / name).with_suffix('.o').touch()
        return sub

    def f2py_calls(self, directory, target):
        result = subprocess.run(
            ['make', '-n', 'SHELL=%s' % (shutil.which('bash') or '/bin/sh'), target],
            cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        calls = []
        for line in result.stdout.splitlines():
            words = shlex.split(line)
            if 'f2py' not in words:
                continue
            start = words.index('f2py')
            env = dict(word.split('=', 1) for word in words[:start])
            calls.append((env, words[start + 1:], line))
        self.assertTrue(calls, result.stdout)
        return calls

    def check_call(self, env, args, line, uname, stdlib):
        for arg in args:
            if arg.startswith('-l'):
                self.assertRegex(arg[2:].replace('.', '_'), MESON_IDENTIFIER,
                                 'meson cannot name this library:\n' + line)
            self.assertFalse(arg.startswith('-Wl'), 'f2py rejects linker options:\n' + line)
            self.assertFalse(arg.endswith('.a'), 'meson links archives first:\n' + line)
        if '-c' not in args:
            return
        self.assertEqual(env.get('FC'), 'gfortran', line)
        ldflags = env.get('LDFLAGS', '').split()
        self.assertEqual(ldflags[0], '$LDFLAGS', 'the user LDFLAGS are dropped:\n' + line)
        self.assertIn(stdlib, ldflags, line)
        # The distutils backend puts LDFLAGS before the objects, where a linker
        # with --as-needed drops the C++ runtime that a static Ninja needs.
        # Apple's ld has no --no-as-needed and keeps every library it is given.
        before_stdlib = ldflags[:ldflags.index(stdlib)]
        self.assertEqual('-Wl,--no-as-needed' in before_stdlib, uname != 'Darwin', line)
        self.assertIn('-lMadLoop', args, line)
        self.assertTrue(any(Path(arg).name == 'mg5_citation.o' for arg in args), line)
        sources = [arg for arg in args if arg.endswith(('.f', '.f90', '.pyf'))]
        self.assertEqual(len({Path(arg).name for arg in sources}), len(sources),
                         'meson compiles copies of these in one directory:\n' + line)
        for arg in args:
            if arg.endswith('.o'):
                self.assertTrue('/' not in arg or Path(arg).is_absolute(),
                                'meson looks for it in its build directory:\n' + line)

    def test_matrix2py_recipe(self):
        for uname, stdlib in (('Linux', '-lstdc++'), ('Darwin', '-lc++')):
            with self.subTest(uname=uname):
                pdir = self.write_output(uname, stdlib) / 'P0_uux_z'
                (call,) = self.f2py_calls(pdir, 'matrix2py.so')
                env, args, line = call
                self.check_call(env, args, line, uname, stdlib)
                self.assertIn('f2py_wrapper.f', args, line)
                # meson compiles a copy of the wrapper, away from its includes
                includes = [Path(arg[2:]) for arg in args if arg.startswith('-I')]
                self.assertIn(pdir.resolve(), [path.resolve() for path in includes], line)

    def test_allmatrix2py_recipe(self):
        for uname, stdlib in (('Linux', '-lstdc++'), ('Darwin', '-lc++')):
            with self.subTest(uname=uname):
                sub = self.write_output(uname, stdlib)
                signature, build = self.f2py_calls(sub, 'allmatrix2py.so')
                for env, args, line in (signature, build):
                    self.check_call(env, args, line, uname, stdlib)
                # Each wrapper is compiled in its own directory, for its own
                # nexternal.inc, and the module is built from the signatures.
                self.assertIn('-h', signature[1], signature[2])
                self.assertIn('P0_uux_z/f2py_wrapper.f', signature[1], signature[2])
                self.assertIn('P1_ddx_z/f2py_wrapper.f', signature[1], signature[2])
                env, args, line = build
                self.assertIn('allmatrix2py.pyf', args, line)
                objects = {Path(arg).resolve().relative_to(sub.resolve()).as_posix()
                           for arg in args if arg.endswith('.o')}
                self.assertEqual(objects, {'all_matrix.o', 'mg5_citation.o',
                                           'P0_uux_z/f2py_wrapper.o',
                                           'P1_ddx_z/f2py_wrapper.o'}, line)


if __name__ == '__main__':
    unittest.main()
