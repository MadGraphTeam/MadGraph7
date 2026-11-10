#!/usr/bin/env python3
"""Quantify every retained contraction error without dropping small components."""
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd().resolve()


def frac(record):
    return Fraction(int(record['numerator']),int(record['denominator']))


def exact(x):
    return {'numerator':str(x.numerator),'denominator':str(x.denominator)}


def dec(x):
    return Decimal(x.numerator)/Decimal(x.denominator)


def ratio(a,b):
    return str(dec(a/b)) if b else None


def compute(source):
    report=json.loads(source.read_text())
    all_rows=[];summaries={};flagged=[]
    with localcontext() as ctx:
        ctx.prec=110
        for mode in (4,5):
            for si,slot in enumerate(('finite','single','double')):
                groups=[g for g in report['groups'] if g['ctmode']==mode]
                assert len(groups)==116
                rows=[]
                host=abs(frac(report['host']['ninjars']['ans'][si+1]))
                l1={'real':Fraction(),'imag':Fraction(),'projected_real':Fraction()}
                error_l1={'real':Fraction(),'imag':Fraction(),'projected_real':Fraction()}
                signed_error={'real':Fraction(),'imag':Fraction(),'projected_real':Fraction()}
                sums={'nrs_projected_real':Fraction(),'tir_projected_real':Fraction()}
                complex_l1=Decimal(0);complex_error_l1=Decimal(0)
                for group in groups:
                    s=group['contracted_laurent_slots'][si]
                    assert s['slot']==slot
                    a=[frac(s[p]['left']) for p in ('real','imag')]
                    b=[frac(s[p]['right']) for p in ('real','imag')]
                    da=[a[i]-b[i] for i in range(2)]
                    aa=sum(x*x for x in a);bb=sum(x*x for x in b);dd=sum(x*x for x in da)
                    an,bn,dn=dec(aa).sqrt(),dec(bb).sqrt(),dec(dd).sqrt()
                    complex_l1+=bn;complex_error_l1+=dn
                    pd=s['actual_dp_projected_real'];pa,pb=frac(pd['left']),frac(pd['right'])
                    imag_projected=s['actual_dp_projected_imag']
                    assert frac(imag_projected['left'])==frac(imag_projected['right'])==0
                    for name,left,right in (('real',a[0],b[0]),('imag',a[1],b[1]),('projected_real',pa,pb)):
                        l1[name]+=abs(right);error_l1[name]+=abs(left-right);signed_error[name]+=left-right
                    sums['nrs_projected_real']+=pb;sums['tir_projected_real']+=pa
                    rows.append({'ctmode':mode,'group':group['group'],'n':group['n'],'rank':group['rank'],'slot':slot,
                                 'real':s['real'],'imag':s['imag'],'dp_projected_real':pd,
                                 'complex_norms_squared':{'tirust':exact(aa),'ninjars':exact(bb),'difference':exact(dd)},
                                 'complex_norms_decimal':{'tirust':str(an),'ninjars':str(bn),'difference':str(dn)},
                                 'complex_symmetric_error':str(2*dn/(an+bn)) if an+bn else '0',
                                 'absolute_projected_error_over_host_abs':ratio(abs(pa-pb),host),
                                 'ninjars_abs_projected_contribution_over_host_abs':ratio(abs(pb),host),
                                 'imaginary_part_projected_to_zero':True})
                summary={'ctmode':mode,'slot':slot,'calls':len(rows),'host_ninjars_abs':exact(host),
                         'ninjars_L1_exact':{k:exact(v) for k,v in l1.items()},
                         'absolute_error_L1_exact':{k:exact(v) for k,v in error_l1.items()},
                         'signed_error_sum_exact':{k:exact(v) for k,v in signed_error.items()},
                         'adapter_projected_sums_exact':{k:exact(v) for k,v in sums.items()},
                         'error_L1_over_ninjars_L1':{k:ratio(error_l1[k],l1[k]) for k in l1},
                         'projected_error_L1_over_host_abs':ratio(error_l1['projected_real'],host),
                         'projected_signed_error_over_host_abs':ratio(abs(signed_error['projected_real']),host),
                         'ninjars_projected_L1_over_abs_signed_sum':ratio(l1['projected_real'],abs(sums['nrs_projected_real'])),
                         'complex_L1_decimal':str(complex_l1),'complex_error_L1_decimal':str(complex_error_l1),
                         'complex_error_L1_over_ninjars_complex_L1':str(complex_error_l1/complex_l1) if complex_l1 else None,
                         'threshold_for_diagnostic_count':'3e-6 symmetric error; not a new per-group acceptance gate',
                         'counts_above_diagnostic_threshold':{p:sum(Decimal(r[p]['symmetric_error'])>Decimal('3e-6') for r in rows) for p in ('real','imag')}}
                summary['counts_above_diagnostic_threshold']['complex']=sum(Decimal(r['complex_symmetric_error'])>Decimal('3e-6') for r in rows)
                for row in rows:
                    for part in ('real','imag'):
                        err=abs(frac(row[part]['left'])-frac(row[part]['right']))
                        row[part+'_absolute_error_over_group_L1']=ratio(err,l1[part])
                    pe=abs(frac(row['dp_projected_real']['left'])-frac(row['dp_projected_real']['right']))
                    row['projected_absolute_error_over_group_L1']=ratio(pe,l1['projected_real'])
                    row['flagged_parts']=[p for p in ('real','imag') if Decimal(row[p]['symmetric_error'])>Decimal('3e-6')]
                    if Decimal(row['complex_symmetric_error'])>Decimal('3e-6'):
                        row['flagged_parts'].append('complex')
                    if row['flagged_parts']:
                        flagged.append(row)
                summaries[f'{mode}:{slot}']=summary
                all_rows.extend(rows)
    return {'schema':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_path':str(source.relative_to(ROOT)),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_executed':False,'runtime_capture_complete':report['runtime_capture_complete'],
            'provisional_raw_comparison':report['provisional_raw_comparison'],'physics_campaign_accepted':False,
            'all_696_complex_slots_included':len(all_rows)==696,'discarded_component_count':0,
            'groups_by_mode_slot':summaries,'flagged_slots':flagged,
            'limits':['Exact rational real/imag errors and squared complex norms are retained; square roots are decimal110-digit evaluations.',
                      'Adapter projected group sums and L1 norms describe recorded outputs; no unstated assembly weights are assumed.',
                      'Ratios to the host amplitude are descriptive scales, not a proof that all other contributions or errors vanish.',
                      'Signed and absolute-error sums are both shown, so compensation cannot hide in the aggregate metric.',
                      'No small or vanishing component is removed; imaginary outputs are explicitly projected out by the retained physical adapter.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    result=compute(args.source.resolve())
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'flagged_slots':len(result['flagged_slots']),
                      'summary':{k:{'counts':v['counts_above_diagnostic_threshold'],'projected_error_L1_over_host':v['projected_error_L1_over_host_abs'],
                                    'projected_error_L1_over_group_L1':v['error_L1_over_ninjars_L1']['projected_real']} for k,v in result['groups_by_mode_slot'].items()}},sort_keys=True))
