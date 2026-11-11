#!/usr/bin/env python3
"""Future rendering source; this proposal does not execute it."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def render(data_path, output):
    raw = Path(data_path).read_bytes()
    data = json.loads(raw)
    if (data['schema'] != 'madloop-private-stress-report-supplement-data-v1'
            or data['new_native_evaluations'] != 0
            or data['failed_coefficients_arithmetic_performed'] is not False):
        raise ValueError('Unexpected explicitly supplied report data')
    rows = data['pole_comparisons']
    if len(rows) != 8 or any(not r['eligible_fixed_mode'] or r['return_code']!=120 for r in rows):
        raise ValueError('Only the eight eligible collapsed pole observations may be plotted')
    output = Path(output)
    if any(output.with_suffix(s).exists() for s in ('.pdf','.png','.json')):
        raise FileExistsError('Fresh output prefix required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(9.4,6.2),constrained_layout=True)
    labels = ['p1 CT','p1 private TIR','p20 CT','p20 private TIR']
    for column,pole in enumerate((-2,-1)):
        selected = sorted((r for r in rows if r['pole']==pole), key=lambda r:(r['point'],r['worker']))
        colours = ['#c74943' if 'lib1-' in r['worker'] else '#188469' for r in selected]
        ratio = [float(Decimal(r['observed_over_predicted'])) for r in selected]
        residual = [float(Decimal(r['relative_residual'])) for r in selected]
        if any(x<=0 for x in residual):
            raise ValueError('Zero residuals require explicit annotation; never add a log-axis floor')
        top,bottom = axes[0,column],axes[1,column]
        top.scatter(range(4),ratio,c=colours,s=50,zorder=3)
        top.axhline(1,color='#173451',linewidth=.8)
        top.set_ylim(0,1.12)
        top.set_ylabel('Observed / predicted pole')
        top.set_title('Double pole' if pole==-2 else 'Single IR pole',color='#173451')
        bottom.scatter(range(4),residual,c=colours,s=50,zorder=3)
        bottom.set_yscale('log')
        bottom.set_ylabel('|observed − predicted| / |predicted|')
        for axis in (top,bottom):
            axis.set_xticks(range(4),labels,rotation=12,fontsize=8)
            axis.grid(alpha=.18)
            axis.spines[['top','right']].set_visible(False)
    fig.suptitle('CMS e⁺ e⁻ → μ⁺ μ⁻: conditional universal-pole consistency\n'
                 'Point 1 is bootstrap; point 20 is the completed stress subset',color='#173451',fontsize=12)
    fig.supxlabel('Eligible unchecked fixed-QP workers only; failed legacy Ninja/NinjaRS are excluded',fontsize=8)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({
        'input_sha256':hashlib.sha256(raw).hexdigest(),'plotted_observations':8,
        'note':'Two points and two coefficients, no error distribution or population inference',
        'runtime_coupling_bits_observed':False,'candidate_external_leaves_newly_captured':False,
        'UV_cancellation_separately_measured':False,'finite_term_adjudicated':False,
        'new_native_evaluations':0},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(args.data,args.output)


if __name__=='__main__':
    main()
