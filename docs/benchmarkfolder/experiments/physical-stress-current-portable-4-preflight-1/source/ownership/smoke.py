from decimal import Decimal
import re

def check_smoke(records_path,trace_path):
    expected=[['HEADER','1'],['INITIAL','-1','-1','0'],
        ['CARD','1','nominal/param_card.dat','4056CC083126E979','-1','-1','-1','-1','0'],
        ['CARD','2','alternate/param_card.dat','4057000000000000','-1','-1','-1','-1','0'],
        ['CARD','3','nominal/param_card.dat','4056CC083126E979','-1','-1','-1','-1','0'],
        ['DONE','3','-1','-1','0'],['CLOSED','-1','-1','0']]
    records=[line.split() for line in records_path.read_text().splitlines()]
    if records!=expected: raise RuntimeError('Card-only records differ from the frozen complete schema')
    a=analyzer()
    trace=a.parse_trace(trace_path)
    if (not trace['observed'] or trace['order']!=[1,2,3] or len(trace['records'])!=3
            or [len(f['records']) for f in trace['flushes']]!=[1,1,1,0]
            or any(f['token']!=-1 for f in trace['flushes'])):
        raise RuntimeError('Card-only trace is incomplete or contains a native reducer export')
    for action,row in enumerate(trace['records'],1):
        expected_fields=dict(sequence=action,action=action,process=0,kind=1,
            token_before=-1,token_after=-1,status=0,version=0,
            before=[0]*15,after=[0]*15)
        expected_fields.update({name:a.NA for name in a.TRACE_NAMES[7:13]+a.TRACE_NAMES[14:18]})
        if any(row[name]!=value for name,value in expected_fields.items()):
            raise RuntimeError('Card-only BEGIN carries unexpected native metadata')
        if len({row['f'+str(i)] for i in range(4)})!=1 or len({row['r'+str(i)] for i in range(4)})!=1:
            raise RuntimeError('Card-only observer altered the floating environment')
    return {'passed':True,'records':records,'trace_records':3,'native_exports_observed':0,
            'model_sequence':['nominal','alternate','nominal'],'context_generations':[-1,-1,-1]}

def numeric_prevalidation(text):
    """Check all numeric card tokens against the actual reader's width20."""
    numeric=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?\Z')
    tokens=[]
    for line_number,line in enumerate(text.splitlines(),1):
        for token in line.split('#',1)[0].split():
            if numeric.fullmatch(token):
                value=Decimal(token.lower().replace('d','e'))
                if len(token)>20 or not value.is_finite():
                    raise RuntimeError(f'Unsupported numeric token at line{line_number}: {token}')
                tokens.append({'line':line_number,'token':token,'width':len(token)})
    return {'reader_character_width':20,'numeric_tokens':tokens,
            'maximum_width':max((item['width'] for item in tokens),default=0)}
