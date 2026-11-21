"""Negative controls for the native-dispatch acceptance criterion."""
import io
from pathlib import Path
import struct
import tempfile
import unittest

from tests.acceptance_tests.mc_batch_trace import audit, read_job, TraceError


def word(value):
    return struct.pack(">d", value).hex().upper()


def trace(mode, scalar=False):
    samples = [" ".join(map(word, [1,1,energy,0,0,0])) for energy in (5,9)]
    lines = ["MG7 MC batch trace v1"]
    for owner in (1,2):
        lines.extend([f"DRAW {mode} {owner}",
                      f"REQUEST {owner} {owner} 1 3 1 {samples[owner-1]}"])
    for call, owners in enumerate(((1,),(2,)) if scalar else ((1,2),),1):
        lines.extend([f"BIND {call} "+ " ".join(map(str,owners)),f"PROVIDER {call} {len(owners)} 3 umami"])
        for lane,owner in enumerate(owners,1):
            lines.extend([f"INPUT {call} {lane} 1 {samples[owner-1]}",
                          f"OUTPUT {call} {lane} {word(10*owner)}"])
        lines.append(f"MAP {call} 1 1 1")
        for lane,owner in enumerate(owners,1):
            result = " ".join([word(10*owner)]*2)
            lines.extend([f"SCATTER {call} {lane} {owner} {result}",f"CONSUME {owner} {owner} {result}"])
    lines.extend([f"COMMIT {mode} {owner}" for owner in (1,2)])
    if mode == 2:
        lines.append("WRITE 2 1")
    return "\n".join(lines)+"\n"


class NativeDispatchGate(unittest.TestCase):
    def test_all_phases_require_genuine_shared_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/str(mode) for mode in range(3)]
            for mode,path in enumerate(paths):
                path.write_text(trace(mode))
            self.assertEqual(len(audit(paths)),3)
            paths[1].write_text(trace(1,scalar=True))
            self.assertEqual(len(audit(paths,require_batch=False)),3)
            with self.assertRaisesRegex(TraceError, "mode 1 has no single UMAMI call"):
                audit(paths)

    def test_physical_identity_and_order_ownership(self):
        good = trace(0)
        inputs = [line for line in good.splitlines() if line.startswith("INPUT")]
        faults = [
            (inputs[1],inputs[0].replace("INPUT 1 1","INPUT 1 2"),"momentum"),
            ("SCATTER 1 1 1","SCATTER 1 1 2","scatter owner"),
            ("OUTPUT 1 1 "+word(10),"OUTPUT 1 1 "+word(20),"physical result"),
            ("CONSUME 2 2","CONSUME 1 2","consumer owner"),
            ("COMMIT 0 2","COMMIT 0 1","commit"),
            ("COMMIT 0 2\n","","unfinished"),
            ("CONSUME 1 1","COMMIT 0 1\nCONSUME 1 1","before its physical result"),
        ]
        for before,after,message in faults:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TraceError,message):
                    read_job(io.StringIO(good.replace(before,after)))

    def test_duplicate_generation_write(self):
        with self.assertRaisesRegex(TraceError,"duplicate event"):
            read_job(io.StringIO(trace(2)+"WRITE 2 1\n"))

    def test_every_accepted_candidate_is_written_or_counted_as_surplus(self):
        trace2 = trace(2).replace("COMMIT 2 1\n", "COMMIT 2 1\nACCEPT 2 1\n")
        trace2 = trace2.replace("COMMIT 2 2\n", "COMMIT 2 2\nACCEPT 2 2\n")
        good = trace2 + "SURPLUS 2 2\n"
        self.assertEqual(read_job(io.StringIO(good))["counts"]["ACCEPT"], 2)
        for bad in (trace2,good.replace("ACCEPT 2 2","REJECT 2 2"),good.replace("ACCEPT 2 1\n","")):
            with self.assertRaises(TraceError):
                read_job(io.StringIO(bad))
