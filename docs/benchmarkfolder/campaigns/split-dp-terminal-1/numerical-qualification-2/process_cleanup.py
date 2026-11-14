"""Linux child-session cleanup with observed lineage and process-start guards."""
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "parent_pid": int(fields[1]),
                "process_group": int(fields[2]), "session": int(fields[3]),
                "start_time_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(record):
    current = identity(record["pid"])
    return (current is not None and current["state"] != "Z"
            and all(current[key] == record[key] for key in
                    ("pid", "start_time_ticks", "process_group", "session")))


class SignalRequests:
    """Defer interruption until a newly spawned child's handle is recorded."""
    def __init__(self):
        self.requested = []
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda number, frame: self.requested.append(number))
        return self

    def __exit__(self, *exception):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def check(self):
        if self.requested:
            raise InterruptedError(f"Supervisor received signal {self.requested[0]}")


class OwnedChild:
    def __init__(self, process):
        self.process = process
        root = identity(process.pid)
        if root is None or root["process_group"] != process.pid or root["session"] != process.pid:
            raise RuntimeError("Controller must have its own identifiable session")
        self.root = root
        self.records = {process.pid: root}
        self.sample()

    def sample(self):
        pending = [record for record in self.records.values() if matches(record)]
        seen = set()
        while pending:
            parent = pending.pop()
            if parent["pid"] in seen or not matches(parent):
                continue
            seen.add(parent["pid"])
            try:
                tasks = list(Path(f"/proc/{parent['pid']}/task").glob("*/children"))
                children = set()
                for path in tasks:
                    try:
                        children.update(map(int, path.read_text().split()))
                    except FileNotFoundError:
                        pass
            except FileNotFoundError:
                continue
            for pid in children:
                child = identity(pid)
                # Revalidate the parent after reading its children to avoid
                # inheriting ownership from a recycled process identifier.
                if child is not None and child["parent_pid"] == parent["pid"] and matches(parent):
                    self.records[pid] = child
                    pending.append(child)

    def live(self):
        return [record for record in self.records.values() if matches(record)]

    def signal_groups(self, sig):
        # Only groups whose session leader was observed in our owned tree may
        # be targeted. A live, start-identity-matching member anchors each send.
        groups = {row["pid"] for row in self.records.values()
                  if row["pid"] == row["process_group"] == row["session"]}
        sent = []
        for group in sorted(groups, reverse=True):
            anchors = [row for row in self.records.values() if row["process_group"] == group]
            if not any(matches(row) for row in anchors):
                continue
            try:
                os.killpg(group, sig)
                sent.append(group)
            except ProcessLookupError:
                pass
        return sent

    def wait(self, requests, poll=.2):
        while True:
            self.sample()
            requests.check()
            try:
                return self.process.wait(timeout=poll)
            except subprocess.TimeoutExpired:
                pass

    def stop(self, cooperative_seconds=45., terminate_seconds=5., kill_seconds=5.):
        self.sample()
        events = []
        if matches(self.root):
            try:
                # SIGINT raises KeyboardInterrupt in the controller, letting
                # Worker and ExitStack close the separately sessioned reducer.
                os.kill(self.root["pid"], signal.SIGINT)
                events.append({"signal": "SIGINT", "controller": self.root["pid"]})
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + cooperative_seconds
        while time.monotonic() < deadline and self.live():
            self.sample()
            self.process.poll()
            time.sleep(.05)
        for sig, seconds in ((signal.SIGTERM, terminate_seconds), (signal.SIGKILL, kill_seconds)):
            if not self.live():
                break
            self.sample()
            events.append({"signal": signal.Signals(sig).name, "groups": self.signal_groups(sig)})
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and self.live():
                self.process.poll()
                time.sleep(.05)
        self.process.poll()
        return {"observed_owned_processes": list(self.records.values()), "events": events,
                "survivors": self.live(), "scope": "Only observed descendants with matching process-start/session identities"}
