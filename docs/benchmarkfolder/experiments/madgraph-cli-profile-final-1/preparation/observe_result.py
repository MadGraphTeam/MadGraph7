"""Observe retained final stock result tags; this is not per-point status proof."""
import math,re

def observe_result(stage):
    record=stage.get('raw_outputs',{}).get('result.dat',{})
    freshness=bool(stage.get('raw_output_freshness_verified',False))
    matches=re.findall(r'(?m)^\s*(BORN|FIN|1EPS|2EPS)\s+([^\s]+)\s*$',record.get('text',''))
    if len(matches)!=4 or len({name for name,_ in matches})!=4:
        return {'available':False,'reason':'missing or duplicate scalar result tags','raw_output_freshness_verified':freshness}
    raw=dict(matches)
    try:
        values={name:float(value.replace('D','E').replace('d','e')) for name,value in matches}
    except ValueError:
        return {'available':False,'reason':'unparseable result tag','raw_output_freshness_verified':freshness,'raw_values':raw}
    nonfinite=[name for name,value in values.items() if not math.isfinite(value)]
    finite=not nonfinite
    return {'available':True,'values':{name:value if math.isfinite(value) else None for name,value in values.items()},'raw_values':raw,'nonfinite_tags':nonfinite,'finite':finite,'born_nonzero':finite and values['BORN']!=0.,'finite_nonzero':finite and values['FIN']!=0.,'all_zero':finite and all(value==0. for value in values.values()),'raw_output_freshness_verified':freshness,'scope':'retained end-of-batch result only; no per-evaluation return codes or tensor routes'}
