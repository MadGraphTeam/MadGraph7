#!/usr/bin/env python3
"""Render actual instrumented READY attribution as a standalone vector figure."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=json.loads(a.data.read_text());controls=data.get('controls',data)
    labels=['Original','Full O2 (unpromoted)'];rows=[controls['baseline'],controls['o2']]
    candidate=data.get('candidate')
    if candidate is not None and candidate.get('measurement_role')=='performance-qualified':
        if candidate.get('actual_source_bound') is not True or candidate.get('functional_qualified') is not True or candidate.get('timing_qualified') is not True or candidate.get('phases') is None:
            raise ValueError('Candidate phase bars require actual qualified phase observations')
        labels.append('Local adaptive candidate');rows.append(candidate['phases'])
    if a.output.exists():raise ValueError('Fresh figure output required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','svg.hashsalt':'madloop-tirust-initialization-profile-v1','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    columns=('setup_seconds','dp_seconds','qp_seconds','outside_dp_qp_setup_seconds')
    colors=('#1baf7a','#2a78d6','#eb6834','#aeb6c3')
    names=('Create + register','DP-entry bridge','QP-entry bridge','Outside DP/QP + setup')
    fig,ax=plt.subplots(figsize=(7.0,.82+.33*len(rows)))
    y=np.arange(len(rows));left=np.zeros(len(rows));totals=[row['ready_seconds'] for row in rows]
    for key,color,name in zip(columns,colors,names):
        widths=np.array([row[key] for row in rows],dtype=float)
        if np.any(~np.isfinite(widths)) or np.any(widths<0):raise ValueError('Invalid measured phase duration')
        ax.barh(y,widths,left=left,color=color,label=name,height=.58,edgecolor='white',linewidth=.3)
        left+=widths
    if not np.allclose(left,totals,rtol=1e-12,atol=1e-9):raise ValueError('Measured segments do not sum to READY')
    for i,value in enumerate(totals):ax.text(value+max(totals)*.012,i,f'{value:.3f} s',va='center',fontsize=8)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,max(totals)*1.16);ax.set_xlabel('Instrumented READY wall time [s]');ax.grid(axis='x',color='#e7eaf0',linewidth=.5);ax.set_axisbelow(True)
    ax.legend(ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.36),fontsize=7.5)
    fig.subplots_adjust(left=.20,right=.98,top=.95,bottom=.37)
    fig.savefig(a.output,format=a.output.suffix.lstrip('.'),metadata={'Date':None},bbox_inches='tight',pad_inches=.03);plt.close(fig)

if __name__=='__main__':main()
