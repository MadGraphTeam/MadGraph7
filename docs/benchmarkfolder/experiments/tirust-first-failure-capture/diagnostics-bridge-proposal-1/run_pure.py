"""Run source/synthetic tests with native process/loading and writes denied."""
import sys
import unittest


def deny(event, args):
    if event.startswith(("subprocess.", "ctypes.dlopen", "os.exec", "os.spawn", "os.posix_spawn")) or event in ("os.system", "os.fork", "os.forkpty"):
        raise RuntimeError("Native execution/loading denied: " + event)
    if event == "open":
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(x in mode for x in "wax+")) or (isinstance(flags, int) and flags & 0x643):
            raise RuntimeError("Writes denied during pure tests")


sys.addaudithook(deny)
suite = unittest.defaultTestLoader.discover(str(__import__("pathlib").Path(__file__).parent), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
