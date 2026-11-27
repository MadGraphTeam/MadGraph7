"""Unsupported toolchains must fail before Cargo, including parallel make."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BRIDGE = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('make') and shutil.which('bash'), 'requires GNU make and bash')
class NinjaRSBuildPreflight(unittest.TestCase):
    def run_make(self, host='Linux', compiler='GNU Fortran',
                 target='x86_64-pc-linux-gnu', cargo=True):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            shutil.copy(BRIDGE / 'Makefile', work / 'Makefile')
            (work / 'mninjars.f90').write_text('end\n')
            binary = work / 'bin'
            binary.mkdir()
            scripts = {
                'uname': '#!/bin/sh\nprintf "%s\\n" "$PREFLIGHT_HOST"\n',
                'fc': '''#!/bin/sh
printf 'fc %s\n' "$*" >> "$PREFLIGHT_LOG"
case "$1" in
    --version) printf '%s\n' "$PREFLIGHT_COMPILER";;
    -dumpmachine) printf '%s\n' "$PREFLIGHT_TARGET";;
    *) touch mninjars.o;;
esac
''',
                'cargo': '''#!/bin/sh
printf 'cargo %s\n' "$*" >> "$PREFLIGHT_LOG"
printf 'note: native-static-libs: -ldl\n'
''',
            }
            for name, script in scripts.items():
                (binary / name).write_text(script)
                (binary / name).chmod(0o755)
            log = work / 'commands.log'
            environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                               PREFLIGHT_HOST=host, PREFLIGHT_COMPILER=compiler,
                               PREFLIGHT_TARGET=target, PREFLIGHT_LOG=str(log))
            result = subprocess.run(
                ['make', '-j8', 'rust', 'mninjars.o', 'FC=' + str(binary / 'fc'),
                 'CARGO=' + str(binary / ('cargo' if cargo else 'absent-cargo'))],
                cwd=work, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return result.returncode, result.stdout, log.read_text().splitlines() if log.exists() else []

    def test_unsupported_host_fails_without_starting_any_compiler(self):
        code, output, commands = self.run_make(host='Darwin')
        self.assertNotEqual(code, 0)
        self.assertIn('supported on Linux', output)
        self.assertEqual(commands, [])

    def test_non_gnu_and_non_linux_targets_fail_before_building(self):
        for compiler, target in [('Intel Fortran', 'x86_64-pc-linux-gnu'),
                                 ('GNU Fortran', 'x86_64-apple-darwin')]:
            with self.subTest(compiler=compiler, target=target):
                code, output, commands = self.run_make(compiler=compiler, target=target)
                self.assertNotEqual(code, 0, output)
                self.assertFalse(any(line.startswith('cargo ') or ' -c ' in line
                                     for line in commands), commands)

    def test_missing_cargo_fails_before_fortran_compilation(self):
        code, output, commands = self.run_make(cargo=False)
        self.assertNotEqual(code, 0)
        self.assertIn('requires Cargo/Rust', output)
        self.assertFalse(any(' -c ' in line for line in commands), commands)

    def test_parallel_build_waits_for_compiler_checks(self):
        code, output, commands = self.run_make()
        self.assertEqual(code, 0, output)
        self.assertEqual(commands[:2], ['fc --version', 'fc -dumpmachine'])
        self.assertTrue(any(line.startswith('cargo rustc') for line in commands[2:]))
        self.assertTrue(any(' -c ' in line for line in commands[2:]))


if __name__ == '__main__':
    unittest.main()
