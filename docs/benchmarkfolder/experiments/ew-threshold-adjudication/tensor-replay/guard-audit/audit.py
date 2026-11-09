#!/usr/bin/env python3
"""Freeze a source-only gate audit; never compile or evaluate tensor integrals."""
from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path
import difflib
import hashlib
import json
import subprocess

ROOT = Path.cwd().resolve()
OUT = ROOT / 'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/guard-audit'
if not (ROOT / 'vendor/tirust/src/lib.rs').is_file():
    raise SystemExit('Run from the MadGraph repository root.')


def digest(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

sources = {
    'historical_bridge': ('.tirust-build/bridge-poles/src/lib.rs', [[21, 30], [160, 181], [238, 253], [490, 505]]),
    'current_bridge': ('vendor/tirust/src/lib.rs', [[171, 184], [252, 263], [336, 349], [493, 511]]),
    'patched_plan': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs', [[230, 234], [749, 818], [956, 969]]),
    'patched_executor': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs', [[610, 627], [1632, 1699], [1720, 1804], [1857, 1901], [1981, 2005]]),
    'patched_numeric': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs', [[269, 290], [341, 354]]),
    'upstream_direct_api': ('../references/TIRust/crates/iregi-rust/src/direct.rs', [[430, 449]]),
    'upstream_precision_ladder': ('../references/TIRust/crates/iregi-rust/src/dispatch/direct_hybrid.rs', [[117, 145], [282, 319]]),
    'upstream_embedded': ('../references/TIRust/crates/iregi-rust/src/dispatch/embedded.rs', [[96, 109]]),
    'upstream_reference_test': ('../references/TIRust/crates/iregi-rust/tests/lifted_caps.rs', [[497, 513]]),
}
identities = {}
excerpts = []
for key, (rel, ranges) in sources.items():
    path = ROOT / rel
    identities[key] = {'path_from_mg_root': rel, **digest(path), 'excerpt_line_ranges': ranges}
    lines = path.read_text().splitlines()
    excerpts.append(f'## {key}\n\nPath: `{rel}`\n\nSHA256: `{identities[key]["sha256"]}`\n')
    for start, stop in ranges:
        excerpts.append('```text\n' + '\n'.join(f'{i}: {lines[i-1]}' for i in range(start, stop+1)) + '\n```\n')
(OUT / 'source-excerpts.md').write_text('\n'.join(excerpts))
expected_historical = '1eab09d61652e4dfade900a22f8c63a071b4fecee2e7454e8144f934f17e34e0'
assert identities['historical_bridge']['sha256'] == expected_historical
original = (ROOT / sources['historical_bridge'][0]).read_text()
assert original.count('const MP_BITS: u32 = 256;') == 1
assert original.count('Float::with_val(MP_BITS, 2).pow(136).inv()') == 2
changed = original.replace('const MP_BITS: u32 = 256;', 'const MP_BITS: u32 = 512;')
changed = changed.replace('Float::with_val(MP_BITS, 2).pow(136).inv()',
                          'Float::with_val(MP_BITS, 2).pow(u64::from(MP_BITS - 120)).inv()')
changed = changed.replace('// 256-bit arithmetic with substantial headroom above binary128.',
                          '// 512-bit diagnostic arithmetic; input binary128 semantics unchanged.')
patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), changed.splitlines(keepends=True),
                                   fromfile='a/src/lib.rs', tofile='b/src/lib.rs'))
(OUT / 'diagnostic-512-source.patch').write_text(patch)
examples = [
    ('point2/group5', 'N4R3 top box', '5.22407811366341444758736094099234985181085186192010198472166824e-30'),
    ('point2/group23', 'N4R3 top box', '1.570291785765733699369409285694621170500756678843946610700491458e-15'),
    ('point2/groups32-35 representative', 'N5R3 box child', '9.421746767184087711463930778943764296494882271661533069073200259e-15'),
]
rows = []
with localcontext() as ctx:
    ctx.prec = 80
    for label, topology, condition in examples:
        loss = -Decimal(3) * Decimal(condition).ln() / Decimal(2).ln()
        rows.append({'label': label, 'topology': topology, 'condition': condition,
                     'subset_coefficient_rank': 3, 'heuristic_loss_bits': str(loss),
                     'heuristic_remaining_bits_at_256': str(Decimal(256)-loss),
                     'heuristic_remaining_bits_at_512': str(Decimal(512)-loss),
                     'smallest_integer_bits_for_120_bit_heuristic': int((loss+120).to_integral_value(rounding=ROUND_CEILING)),
                     'smallest_integer_bits_with_additional_8_bit_margin': int((loss+128).to_integral_value(rounding=ROUND_CEILING))})
    gates = {str(bits): {'exponent': -(bits-120), 'decimal': str(Decimal(2)**(-(bits-120)))}
             for bits in (256,384,512)}
bindings = {}
for rel in ['docs/benchmarkfolder/experiments/ew-threshold-adjudication/trace.json',
            'docs/benchmarkfolder/experiments/ew-threshold-adjudication/summary.json',
            'docs/benchmarkfolder/experiments/ew-threshold-adjudication/gram-guard.json']:
    bindings[rel] = digest(ROOT / rel)
record = {
    'schema': 1, 'scope': 'Source-only audit and unexecuted private diagnostic recipe; no numerical qualification or production-policy change.',
    'source_identities': identities, 'trace_bindings': bindings,
    'upstream_revision': subprocess.check_output(['git','-C','../references/TIRust','rev-parse','HEAD'],text=True).strip(),
    'historical_policy': {'input_bits':113, 'working_bits':256, 'fixed_gate_exponent':-136,
                          'lightlike_relative_gate_exponent':-100},
    'diagnostic_policy': {'working_bits':512, 'formula':'2^-(working_bits - 120)',
                          'input_bits_unchanged':113, 'lightlike_relative_gate_exponent_unchanged':-100,
                          'build_executed':False, 'native_evaluation_executed':False,
                          'source_before_sha256':expected_historical,
                          'source_after_sha256':hashlib.sha256(changed.encode()).hexdigest(),
                          'functional_source_changes':3, 'comment_changes':1,
                          'working_precision_gate_table':gates},
    'heuristic_examples':rows,
    'interpretation': {
        'target120_is_derived_not_proven':'256 - 136 = 120, seven bits above binary128 input precision. This is an interpretation of the fixed bridge threshold, not a documented accuracy bound.',
        'roundoff_model':'If the powered estimate g acted as inverse amplification, relative arithmetic error would scale like 2^-p/g. The source estimate is not a certified inverse condition number.',
        'actual_gram_quantity':'min(|det Z|^(1/m)/scale, |det Z|/product(row_norm_2))^subset.coefficient_rank',
        'rank_derivation':'Ordinary undotted D4 N4R3 top box coefficient rank3; N5R3 Denner-Dittmaier rank=min(3,5)=3 propagates rank3 box coefficients. No small-Gram rank boost is enabled by these bridge MP workspaces.',
        'other_consumers':'The same setter also controls scalar/modified Cayley determinant guards, higher-rank pentagon Gram checks and six-point coordinate checks; it is not a Gram-only relaxation.',
        'upstream_support':'Public set_stability_threshold/set_input_precision/last_conditions support caller-managed retries; upstream rescue uses precision-scaled gates but explicitly warns the gate alone bounds nothing.',
        'input_uncertainty':'Higher working precision evaluates the same binary128 inputs more accurately; it cannot recover information already lost when those inputs were formed.',
        'fixed_gate_control':'MP_BITS-only changes arithmetic but leaves the acceptance threshold fixed. An unchanged condition remains refused; the computed condition itself may change with precision.',
        'first_failure_limit':'Current trace reports each lane first failure. Admitting it can expose another failing subset or scalar master;512 is not guaranteed to serve all37.',
        'cache_limit':'Existing whole-QP cache includes MP_BITS but not a freely variable gate. Use one fixed policy per private DSO and cache-disabled independent checks; a future adaptive policy needs explicit cache identity/invalidation.',
        'host_status_limit':'The traced stress amplitude was unchecked status120 and omitted37 refused groups; its wrong aggregate pole is not by itself evidence that an accepted individual tensor was inaccurate.',
    },
    'required_followup': [
        'Compare exact same binary128 decimal records to independently matched NinjaRS tensors, including original and reversed order and ordinary control.',
        'Keep historical256/fixed136 results; if approved run isolated512/scaled392 and optionally512/fixed136 to separate arithmetic from gate admission.',
        'Record actual working bits, gate, per-call status/refusal cause, input/source/DSO hashes, all finite/UV/IR components and original/check cache modes.',
        'For newly served lanes, compare reversed/rotated tensor results and cross-precision convergence; higher precision agreement alone is not an independent truth certificate.',
        'Reconstruct complete physical amplitudes only when all required groups are served or host rescue is proven; apply independent3e-6 truth gate and shared1e-6 stability policy.',
        'Do not promote a production adaptive fallback solely from this heuristic or one diagnostic; require correctness, precision-use accounting and performance evidence.'
    ],
}
write_json('audit.json', record)
index = {p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'evidence-index.json'}
write_json('evidence-index.json', {'schema':1, 'scope':record['scope'], 'files':index})
print(json.dumps({'files':len(index), 'index_sha256':digest(OUT/'evidence-index.json')['sha256'],
                  'source_after_sha256':record['diagnostic_policy']['source_after_sha256']},sort_keys=True))
