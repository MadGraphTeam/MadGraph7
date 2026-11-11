"""Derive literal diagnostic inputs without invoking any numerical code."""
import json
from pathlib import Path
import re

from admission import HERE, OLD, ROOT, read_json, require, save, sha, verify_record


def card_values(raw):
    lines = raw.decode('ascii').splitlines(keepends=True)
    fields = {}
    for i, line in enumerate(lines):
        match = re.fullmatch(r'#([A-Za-z][A-Za-z0-9_]*)\s*', line.rstrip('\r\n'))
        if not match:
            continue
        key = match[1].lower()
        require(key not in fields and i+1 < len(lines), 'Ambiguous card field')
        require(not lines[i+1].lstrip().startswith('#'), 'Missing card value')
        fields[key] = {'line': i+1, 'value': lines[i+1].strip()}
    return lines, fields


def card_derivative(raw, mode):
    require(mode in (1,4), 'Initial native matrix admits only modes 1/4')
    lines, fields = card_values(raw)
    values = {'ctmodeinit':'1', 'ctmoderun':str(mode), 'looppolemode':'0',
              'mlreductionlib':'9', 'tirustcomputefinite':'.TRUE.',
              'tirustglobalcache':'0', 'useqpintegrandfortirust':'.TRUE.'}
    for name, value in values.items():
        require(name in fields, 'Missing diagnostic field '+name)
        old = lines[fields[name]['line']]
        ending = '\r\n' if old.endswith('\r\n') else '\n' if old.endswith('\n') else ''
        lines[fields[name]['line']] = value+ending
    derived = ''.join(lines).encode('ascii')
    _, after = card_values(derived)
    require(all(after[k]['value']==v['value'] for k,v in fields.items() if k not in values),
            'Unrequested card field changed')
    return derived, {'overrides':values,'donor_values':{k:v['value'] for k,v in fields.items()},
                     'unchanged_other_fields':True}


def point_slice(raw, nlegs, point):
    lines = raw.splitlines(keepends=True)
    require(lines and lines[0].strip().isdigit(), 'Point count missing')
    count = int(lines[0])
    require(len(lines)==1+count*nlegs and 1<=point<=count, 'Point layout differs')
    selected = lines[1+(point-1)*nlegs:1+point*nlegs]
    require(all(len(line.split())==4 for line in selected), 'Four decimal components required')
    # Keep every source decimal byte; only the independent one-point header is new.
    return b'1\n'+b''.join(selected)


def prepare(destination):
    inputs = read_json(OLD/'input-bindings.json')
    destination = Path(destination)
    require(not destination.exists(), 'Fresh literal-input directory required')
    destination.mkdir(parents=True)
    records, requests = {}, []
    for key, nlegs in (('cms',4),('ew',5)):
        case = inputs[key]
        raw = {k:verify_record(ROOT/case[k]['path'],case[k])
               for k in ('points','physical_card','numerical_card')}
        (destination/(key+'-param_card.dat')).write_bytes(raw['physical_card'])
        records[key] = {'donor_inputs':{k:case[k] for k in raw},'cards':{},'points':{}}
        for mode in (1,4):
            card, changes = card_derivative(raw['numerical_card'],mode)
            name = key+'-mode'+str(mode)+'-MadLoopParams.dat'
            (destination/name).write_bytes(card)
            records[key]['cards'][str(mode)] = {'file':name,'sha256':sha(card),**changes}
        for point in case['selected_original_point_numbers']:
            data = point_slice(raw['points'],nlegs,point)
            name = key+'-point'+str(point)+'.dat'
            (destination/name).write_bytes(data)
            records[key]['points'][str(point)] = {'file':name,'sha256':sha(data),
                'exact_momentum_slice_sha256':sha(data.split(b'\n',1)[1])}
    # Ordinary parity is a complete 12-worker gate before any stress request.
    for stage, points in (('ordinary',[('cms',1),('ew',1)]),
                          ('stress',[('cms',20),('cms',21),('ew',2)])):
        for key,point in points:
            for mode in (1,4):
                for variant in ('unmodified','disabled','enabled'):
                    requests.append({'id':f'{key}-point{point}-mode{mode}-{variant}',
                        'stage':stage,'case':key,'point':point,'requested_mode':mode,
                        'variant':variant,'trace_enabled':variant=='enabled',
                        'bootstrap_point':1,'bootstrap_outer_entries':1,
                        'maximum_outer_amplitude_entries':2,
                        'bootstrap':key+'-point1.dat','measurement':key+'-point'+str(point)+'.dat',
                        'numerical_card':key+'-mode'+str(mode)+'-MadLoopParams.dat',
                        'physical_card':key+'-param_card.dat'})
    require(len(requests)==30 and sum(r['stage']=='ordinary' for r in requests)==12,
            'Amended request matrix differs')
    save(destination/'derivation.json',records)
    save(destination/'requests.json',{'schema':'current-kinematic-trace-requests-v1',
        'requests':requests,'native_modes':[1,4],'deferred_native_modes':[2,5],
        'maximum_workers':30,'maximum_outer_amplitude_entries':60,
        'bootstrap_readiness':'Restored INIT state only; explicit LoopPoleMode0 bypass contract and measured actual mode are required instead.',
        'force_stability_check':False})
    return records
