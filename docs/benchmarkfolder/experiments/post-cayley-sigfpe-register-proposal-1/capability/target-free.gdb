set pagination off
set confirm off
set auto-load python-scripts off
set debuginfod enabled off
set startup-with-shell off
set architecture i386:x86-64
handle SIGFPE stop print nopass
info signals SIGFPE
help handle
help info registers
help break
help python
show architecture
maintenance print registers
set breakpoint pending on
break main
info breakpoints
delete breakpoints
python
import gdb
import json
import sys
result = {
    "gdb_version": gdb.VERSION,
    "python_version": sys.version,
    "selected_inferior_pid": gdb.selected_inferior().pid,
    "inferior_count": len(gdb.inferiors()),
    "inferior_has_read_memory_api": hasattr(gdb.Inferior, "read_memory"),
    "frame_has_read_register_api": hasattr(gdb.Frame, "read_register"),
    "stop_event_api": hasattr(gdb.events, "stop"),
    "SignalEvent_class": hasattr(gdb, "SignalEvent"),
    "Breakpoint_class": hasattr(gdb, "Breakpoint"),
    "execute_api": callable(gdb.execute),
    "parse_and_eval_api": callable(gdb.parse_and_eval),
    "pending_breakpoint_was_deleted": gdb.breakpoints() in (None, ()),
    "no_target_file_loaded": gdb.current_progspace().filename is None,
}
assert result["selected_inferior_pid"] == 0
assert result["no_target_file_loaded"]
print("TARGET_FREE_CAPABILITY_JSON " + json.dumps(result, sort_keys=True))
print("TARGET_FREE_SIGINFO " + gdb.execute("print $_siginfo", to_string=True).strip())
end
quit
