#!/usr/bin/env python3
"""Layout source for explicitly supplied closed descriptive timing summaries.

This program never launches a benchmark or discovers active result streams.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

IDS=(1,6,7,8,9)
NAMES={1:'CutTools',6:'Ninja',7:'COLLIER',8:'NinjaRS',9:'TIRust'}
SUMMARY_SCHEMA='madloop-current-source-five-backend-descriptive-timing-v1'


def numerical(case,backend):
    return (case.get('numerical_validation') or {}).get('backends',{}).get(str(backend),{})


def eligible(case,backend):
    return (case['status']=='closed' and case.get('summary') is not None
            and numerical(case,backend).get('regression_pass') is True)


def validate(data):
    if data.get('status')!='closed' or not data.get('evidence'):
        raise ValueError('Rendering measured timings requires explicitly selected closed evidence')
    if len(data.get('cases',[]))!=4 or len({c['case_id'] for c in data['cases']})!=4:
        raise ValueError('Four distinct controller-selected cases required')
    for case in data['cases']:
        summary=case.get('summary')
        if summary is None:
            if case['status'] not in ('pending','failed','unexecuted') or not case.get('reason'):
                raise ValueError('Unavailable cases require their explicit status and reason')
            continue
        if summary.get('schema')!=SUMMARY_SCHEMA or summary['case_id']!=case['case_id']:
            raise ValueError('Wrong case-summary identity')
        if summary['performance_acceptance'] or summary['cpu_reservation'] or summary['scope']!='full amplitude':
            raise ValueError('Timing scope changed')
        if summary['clock']!='SYSTEM_CLOCK wall':
            raise ValueError('Clock differs from the admitted layout')
        if set(summary['backends'])!={str(b) for b in IDS} or summary['cycle_count']!=10:
            raise ValueError('All five backends and ten cycles must remain present')
        for backend in IDS:
            lane=summary['backends'][str(backend)]
            if lane.get('mad_scale')!='unscaled':
                raise ValueError('Spread convention differs from the admitted layout')
            values=[lane['median_seconds_per_point'],lane['mad_seconds_per_point'],lane['median_ratio_to_8']]
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values):
                raise ValueError('Nonfinite or missing timing statistic')
            if values[0]<=0 or values[1]<0 or values[2]<=0:
                raise ValueError('Invalid timing statistic; missing observations are never zero')
            if len(lane['samples'])!=10 or len(lane['ratios_to_8'])!=10:
                raise ValueError('Incomplete timing population')
    return data


def render(data_path,output):
    raw=Path(data_path).read_bytes()
    data=validate(json.loads(raw))
    output=Path(output)
    if any(output.with_suffix(s).exists() for s in ('.pdf','.png','.json')):
        raise FileExistsError('Fresh timing-figure output prefix required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    import numpy as np
    fig=plt.figure(figsize=(10,8),constrained_layout=True)
    grid=fig.add_gridspec(3,2,height_ratios=(1,1,.9))
    ratio_values=np.full((4,5),np.nan)
    for index,case in enumerate(data['cases']):
        axis=fig.add_subplot(grid[index//2,index%2])
        summary=case.get('summary')
        if summary is None:
            axis.axis('off')
            axis.text(.05,.7,case['label'],fontsize=11,color='#1f3b63')
            axis.text(.05,.4,case['status']+': '+case['reason'],fontsize=9,wrap=True)
            continue
        for row,backend in enumerate(IDS):
            item=summary['backends'][str(backend)]
            value=item['median_seconds_per_point']*1000
            spread=item['mad_seconds_per_point']*1000
            valid=eligible(case,backend)
            axis.barh(row,value,color='#1f3b63' if valid else '#b6b6b6',alpha=.88,
                      hatch=None if valid else '//',height=.6)
            axis.errorbar(value,row,xerr=spread,color='#555',capsize=2,linewidth=.8)
            if valid and eligible(case,8):ratio_values[index,row]=item['median_ratio_to_8']
        axis.set_yticks(range(5),[NAMES[b] for b in IDS],fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel('Wall ms / physical evaluation · median ± unscaled MAD',fontsize=8)
        axis.set_title(case['label']+'\n'+str(summary['points_per_traversal'])+' points / traversal; ten cycles',fontsize=10,color='#1f3b63')
        axis.grid(axis='x',alpha=.16)
        axis.set_xlim(left=0)
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(axis='x',labelsize=8)
    heat=fig.add_subplot(grid[2,:])
    palette=LinearSegmentedColormap.from_list('report-ratio',['#1a9e77','#c4e6d9','#ffffff','#fbd0c3','#d73027'])
    palette.set_bad('#eeeeee')
    mapped=np.ma.masked_invalid(np.log2(ratio_values))
    heat.imshow(mapped,cmap=palette,norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4),aspect='auto')
    heat.set_xticks(range(5),[NAMES[b] for b in IDS],fontsize=9)
    heat.set_yticks(range(4),[c['label'] for c in data['cases']],fontsize=8)
    for row in range(4):
        for column in range(5):
            value=ratio_values[row,column]
            heat.text(column,row,'—' if not math.isfinite(value) else f'×{value:.3g}',ha='center',va='center',fontsize=9)
    heat.set_title('Median within-cycle backend / NinjaRS8 ratio · green faster, red slower',fontsize=10,color='#1f3b63')
    heat.spines[:].set_visible(False)
    fig.suptitle('Descriptive shared-host SM timings · full amplitude\nMAD shows batch variability; no confidence interval or local acceptance claim',fontsize=12,color='#1f3b63')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({
        'schema':'madloop-five-backend-timing-figure-v1','input_sha256':hashlib.sha256(raw).hexdigest(),
        'case_ids':[c['case_id'] for c in data['cases']],
        'absolute_statistic':'median SYSTEM_CLOCK wall seconds/(points*loops), displayed in ms',
        'whiskers':'unscaled MAD across ten timed batches; not confidence intervals',
        'ratio_statistic':'median of ten within-cycle backend/NinjaRS8 ratios',
        'numerically_unqualified_ratio_cells_masked':True,'new_native_physics_evaluations':0},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.data,args.output)


if __name__=='__main__':main()
