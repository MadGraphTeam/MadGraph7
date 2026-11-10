"""Pure checks joining the private COMMON views to frozen generated source."""
import re

COUPLINGS='GC_4 GC_5 GC_6 R2_3GQ R2_3GG R2_GQQ R2_GGQ R2_GGB R2_GGT R2_GGG_1 R2_GGG_2 R2_QQQ UV_3GG_1EPS UV_3GB_1EPS UV_GQQG_1EPS UV_GQQB_1EPS UVWFCT_G_1_1EPS R2_DDZ_V3 R2_UUZ_V2 R2_GGZUP R2_GGZDOWN R2_GGGZVECUP R2_GGGZVECDOWN R2_GGGZAXIALUP R2_GGGZAXIALDOWN UV_3GB UV_3GT UV_GQQB UV_GQQT UVWFCT_G_1 UVWFCT_G_2 GC_21 GC_22 GC_23'.split()

def statements(text):
    result=[]
    for line in text.splitlines():
        if not line.strip() or line[0] in 'cC*!': continue
        content=line[6:].split('!',1)[0].replace(' ','').upper()
        if line[5:6].strip():
            if not result: raise ValueError('orphan continuation')
            result[-1]+=content
        else: result.append(content)
    return result

def common(stmts,name):
    prefix='COMMON/'+name+'/'
    found=[s[len(prefix):].split(',') for s in stmts if s.startswith(prefix)]
    if len(found)!=1: raise ValueError('COMMON count '+name)
    return found[0]

def verify(dp,mp,loop,compute,amp,maximum,observer):
    d,q=statements(dp),statements(mp)
    expected={'STRONG':['G','ALL_G'],'RSCALE':['MU_R','ALL_MU_R'],'WEAK':['GAL'],
        'MASSES':['MDL_'+x for x in ('MB','MH','MT','MTA','MW','MZ')],
        'WIDTHS':['MDL_'+x for x in ('WH','WT','WW','WZ')],'COUPLINGS':COUPLINGS}
    for name,names in expected.items():
        if common(d,name)!=names: raise ValueError('DP COMMON order '+name)
        qnames=['MP__'+x for x in names if not x.startswith('ALL_')]
        if common(q,'MP_'+name)!=qnames: raise ValueError('QP COMMON order '+name)
    # Every scalar's kind and the sole GAL array extent are asserted, including
    # complex QP MU_R rather than wrongly declaring it real128.
    for stmts,prefix,kinds in ((d,'',('DOUBLEPRECISION','DOUBLECOMPLEX')),(q,'MP__',('REAL*16','COMPLEX*32'))):
        for name,names in expected.items():
            kind=kinds[1] if name in ('WEAK','COUPLINGS') or (prefix and name=='RSCALE') else kinds[0]
            decls=[]
            for s in stmts:
                if s.startswith(kind):
                    tail=s[len(kind):]
                    if '::' in tail: tail=tail.split('::',1)[1]
                    decls.extend(tail.split(','))
            for value in names:
                if prefix and value.startswith('ALL_'): continue
                target=prefix+value+('(2)' if value=='GAL' else '')
                if target not in decls: raise ValueError('COMMON scalar kind/extent '+target)
    l,c,a=map(statements,(loop,compute,amp))
    checks={
      'loop':(l,['PARAMETER(NEXTERNAL=5)','PARAMETER(NCOMB=48)',
        'PARAMETER(NLOOPS=224,NLOOPGROUPS=116,NCTAMPS=278)',
        'PARAMETER(NSO=1,NSQUAREDSO=1,NAMPSO=2)',
        'COMPLEX*16LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)','LOGICALS(NSQUAREDSO,NLOOPGROUPS)',
        'COMMON/ML5_0_LOOPRES/LOOPRES,S','COMMON/ML5_0_CT/LSCALE,CTMODE','REAL*8LSCALE',
        'COMMON/ML5_0_I_LIB/I_LIB']),
      'compute':(c,['PARAMETER(NBORNAMPS=8)',
        'COMPLEX*32LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,NLOOPGROUPS)',
        'COMMON/ML5_0_MP_LCOEFS/LOOPCOEFS']),
      'amp':(a,['COMPLEX*32AMP(NBORNAMPS)','COMMON/ML5_0_MP_AMPS/AMP'])}
    for name,(actual,required) in checks.items():
        for text in required:
            if text not in actual: raise ValueError('Generated declaration '+name+': '+text)
    if 'PARAMETER(LOOPMAXCOEFS=70)' not in statements(maximum): raise ValueError('rank capacity')
    normalized=re.sub(r'\s+','',observer.lower())
    for text in ('dc(34)','qc(34)','model_words=254','qco(0:69,1,116)','qamp(8)',
                 'loopres(3,1,116)','stable_groups(1,116)','common/ml5_0_loopres/loopres,stable_groups'):
        if text not in normalized: raise ValueError('Observer declaration '+text)
    return {'nexternal':5,'helicities':48,'groups':116,'qp_coefficients':70,'born_amplitudes':8,
            'couplings':len(COUPLINGS),'model_int64_words':254,'common_order_and_scalar_kinds_asserted':True}
