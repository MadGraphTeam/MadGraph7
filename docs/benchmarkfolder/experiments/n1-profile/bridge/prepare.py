#!/usr/bin/env python3
"""Create an instrumented bridge copy; never edit production or install paths."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path(__file__).resolve().parent
BUILD = ROOT / '.tirust-build/n1-profile/bridge'
assert Path.cwd().resolve() == ROOT
BUILD.joinpath('src').mkdir(parents=True, exist_ok=True)
original = ROOT / 'vendor/tirust/src/lib.rs'
code = original.read_text()


def replace(old, new):
    global code
    assert code.count(old) == 1, (old[:100], code.count(old))
    code = code.replace(old, new)


replace('pub const ABI_VERSION: u32 = 1;', 'mod profile;\npub const ABI_VERSION: u32 = 1;')
replace('struct State {\n    budget: u64,', 'struct State {\n    profile_level: u32,\n    budget: u64,')
replace('            budget: DEFAULT_BUDGET,', '            profile_level: 0,\n            budget: DEFAULT_BUDGET,')
# Only the first (full-DP) method gets workspace timers; other paths retain code.
start = code.index('    fn dp(&mut self,')
end = code.index('    fn dp_checked(', start)
part = code[start:end]
part = part.replace('        self.register(key)?;', '''        let setup = profile::start();
        self.register(key)?;
        let config = profile::start();
        let requested = profile::level();
        if self.profile_level != requested {
            for workspace in self.dp.values_mut() {
                workspace.set_profiling(requested >= 2).map_err(|e| e.to_string())?;
            }
            self.profile_level = requested;
        }
        profile::end(10, config);''', 1)
part = part.replace('            let workspace =', '            let mut workspace =', 1)
part = part.replace('            self.dp.insert(key, workspace);', '''            workspace.set_profiling(requested >= 2).map_err(|e| e.to_string())?;
            self.dp.insert(key, workspace);''', 1)
part = part.replace('        // check_view()', '        profile::end(9, setup);\n        // check_view()', 1)
part = part.replace('        let store =', '        let attach = profile::start();\n        let store =', 1)
part = part.replace('        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;', '''        profile::end(5, attach);
        let reduction = profile::start();
        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
        profile::end(6, reduction);
        profile::upstream(workspace.last_profile());
        profile::routes(workspace.last_lane_routes());''', 1)
part = part.replace('            Ok(values) => Ok((SUCCESS, pack_dp(values, mode.poles, 4))),', '''            Ok(values) => {
                let pack = profile::start();
                let output = pack_dp(values, mode.poles, 4);
                profile::end(7, pack);
                Ok((SUCCESS, output))
            },''', 1)
code = code[:start] + part + code[end:]

# Auxiliary APIs are timed separately and are excluded from individual DP keys.
for function, slot in [('ml_tirust_clear_cache', 0), ('ml_tirust_set_cache_budget', 2), ('ml_tirust_register_plan', 4)]:
    start = code.index('pub unsafe extern "C" fn ' + function)
    end = code.index('\n}\n', start) + 3
    part = code[start:end]
    part = part.replace('    guard(|| {', '    let timer = profile::start();\n    let status = guard(|| {', 1)
    assert part.endswith('    })\n}\n')
    part = part[:-len('    })\n}\n')] + f'    }});\n    profile::auxiliary({slot}, timer);\n    status\n}}\n'
    code = code[:start] + part + code[end:]

start = code.index('pub unsafe extern "C" fn ml_tirust_evaluate_dp(')
end = code.index('\n}\n', start) + 3
part = code[start:end]
part = part.replace('    guard(|| {', '''    let total = profile::start();
    profile::begin(n, rank, check, cache);
    let result_status = guard(|| {''', 1)
part = part.replace('        let batch = dp_batch(', '        let input = profile::start();\n        let batch = dp_batch(', 1)
old = '''        let (status, values) = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?
            .dp_checked(key, batch, mode)?;'''
new = '''        profile::end(2, input);
        let locking = profile::start();
        let mut state = unsafe { context(handle)? }.state.lock().map_err(|_| "TIRust context poisoned")?;
        profile::end(3, locking);
        let state_call = profile::start();
        let (status, values) = state.dp_checked(key, batch, mode)?;
        profile::end(4, state_call);
        drop(state);
        let copy = profile::start();'''
assert old in part
part = part.replace(old, new, 1)
part = part.replace('        Ok(status)', '        profile::end(8, copy);\n        Ok(status)', 1)
assert part.endswith('    })\n}\n')
part = part[:-len('    })\n}\n')] + '''    });
    profile::status(result_status);
    profile::end(1, total);
    result_status
}
'''
code = code[:start] + part + code[end:]
BUILD.joinpath('src/lib.rs').write_text(code)
shutil.copyfile(SOURCE / 'profile.rs', BUILD / 'src/profile.rs')
paths = [original, Path(__file__).resolve(), SOURCE / 'profile.rs', BUILD / 'src/lib.rs']
BUILD.joinpath('preparation.json').write_text(json.dumps({
    'scope': 'Isolated diagnostic instrumentation; production/installed library unchanged',
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'temporary_abi': ['int32_t ml_tirust_profile_control(int32_t level)',
                      'size_t ml_tirust_profile_json(char *out, size_t capacity)']}, indent=2) + '\n')
print(BUILD)
