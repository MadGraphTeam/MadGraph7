"""GDB-only operand capture; never evaluate target functions or resume SIGFPE."""
import gdb
import hashlib
import json
import os
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
launch = json.loads(Path('launch.json').read_text())
fork = json.loads(Path('fork.json').read_text())
record = {'schema':'post-cayley-stopped-operands-v1', 'captured':False,
          'target_functions_called_by_debugger':False, 'resumed_after_SIGFPE':False,
          'debugger_explicitly_sets_fpe_flags_or_rounding':False, 'events':[]}
phase = 'attach'

def save():
    Path('capture.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')

def stopped(event):
    record['events'].append({'phase':phase, 'type':type(event).__name__,
        'signal':getattr(event, 'stop_signal', None)})

def execute(command):
    output = gdb.execute(command, to_string=True)
    with Path('debugger-commands.log').open('a') as stream:
        stream.write(command + '\n' + output)
    return output

def last_signal():
    return record['events'][-1].get('signal') if record['events'] else None

try:
    assert gdb.selected_inferior().pid == 0
    assert fork['debugger_pid'] == fork['debugger_pgid'] == os.getpid() == os.getpgrp()
    gdb.events.stop.connect(stopped)
    for command in ('set pagination off','set confirm off','set print elements 0',
                    'set print repeats 0','set auto-load off',
                    'set debuginfod enabled off','set may-call-functions off',
                    'set follow-exec-mode same',
                    'handle SIGSTOP nostop noprint nopass',
                    'handle SIGFPE stop print nopass'):
        execute(command)
    execute('attach ' + str(fork['inferior_pid']))
    assert gdb.selected_inferior().pid == fork['inferior_pid']
    assert os.getpgid(fork['inferior_pid']) == os.getpid()
    phase = 'exec'
    execute('catch exec')
    execute('continue')
    assert Path(gdb.current_progspace().filename).resolve() == Path(launch['executable']).resolve()
    assert last_signal() != 'SIGFPE'
    for breakpoint in gdb.breakpoints() or ():
        breakpoint.delete()
    main_address = int(gdb.parse_and_eval('&main'))
    main = gdb.Breakpoint('*' + hex(main_address), internal=True)
    phase = 'main'
    execute('continue')
    assert int(gdb.selected_frame().pc()) == main_address
    main.delete()
    assert os.getpgid(fork['inferior_pid']) == os.getpid()
    # Reuse the exact closed mapping reader, adapting its sole CPU assertion.
    helper_path = Path(launch['mapping_helper'])
    helper_raw = helper_path.read_bytes()
    assert hashlib.sha256(helper_raw).hexdigest() == launch['mapping_helper_sha256']
    helper_text = helper_raw.decode('utf8')
    assert helper_text.count("affinity!=['11']") == 1
    helper_text = helper_text.replace("affinity!=['11']", "affinity!=['12']").replace('CPU11','CPU12')
    helper = {'__file__':str(helper_path), '__name__':'debugger_mapping_reader'}
    exec(compile(helper_text,str(helper_path),'exec'),helper)
    state = {}
    mapping = helper['capture_maps'](fork['inferior_pid'], launch['expected_runtime_sha256'],
        launch['executable'], launch['selected_library'], Path('native-maps.txt'), state)
    assert mapping is not None
    record['actual_mapping'] = mapping
    record['mapping_state'] = state
    Path('mapping-ready').write_text('Verified target mapping before numerical execution\n')
    phase = 'physical'
    save()
    execute('continue')
    assert last_signal() == 'SIGFPE', 'Target did not stop at a SIGFPE'
    assert gdb.selected_inferior().pid == fork['inferior_pid']
    frame = gdb.selected_frame()
    registers = {'rip':hex(int(frame.read_register('rip'))),
                 'eflags':hex(int(frame.read_register('eflags'))),
                 'mxcsr':hex(int(frame.read_register('mxcsr')))}
    for name in ('xmm0','xmm1'):
        values = []
        for index in (0,1):
            bits = int(gdb.parse_and_eval('$' + name + '.v2_int64[' + str(index) + ']')) & ((1<<64)-1)
            value = struct.unpack('<d',bits.to_bytes(8,'little'))[0]
            values.append({'bits':f'{bits:016x}','binary64_hex':value.hex()})
        registers[name] = values
    record['registers'] = registers
    record['siginfo'] = {name:int(gdb.parse_and_eval('$_siginfo.' + name))
                         for name in ('si_signo','si_code','si_errno')}
    Path('siginfo.txt').write_text(execute('p $_siginfo'))
    Path('registers.txt').write_text(execute('info registers rip eflags mxcsr xmm0 xmm1'))
    Path('x87.txt').write_text(execute('info float'))
    Path('instructions.txt').write_text(execute('x/8i $pc'))
    Path('backtrace.txt').write_text(execute('bt'))
    Path('stopped-maps.txt').write_text(Path('/proc/' + str(fork['inferior_pid']) + '/maps').read_text())
    record['captured'] = True
except BaseException as error:
    record['failure'] = {'phase':phase,'type':type(error).__name__,'message':str(error)}
finally:
    if gdb.selected_inferior().pid:
        try:
            record['kill_output'] = execute('kill')
        except BaseException as error:
            record['kill_failure'] = {'type':type(error).__name__,'message':str(error)}
    record['inferior_pid_after_kill'] = gdb.selected_inferior().pid
    save()
