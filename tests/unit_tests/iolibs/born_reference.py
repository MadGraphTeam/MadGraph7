"""Independent scalar adapters for pre-finalization Born reference sources."""
import re

from madgraph.iolibs import born_support as formatting


def write_reference(source, record, model_binding):
    paths = [source/name for name in ('born.f', 'born_hel.f', 'sborn_sf.f',
                                     'extra_cnt_wrapper.f')]
    paths += sorted(source.glob('b_sf_*.f'))+sorted(source.glob('born_cnt_*.f'))
    lines, skipping = [], False
    for path in paths:
        for line in formatting.statements(path.read_text()):
            match = re.match(r'(?i)\s*(?:[\w*]+\s+)*?(?:subroutine|function)\s+(\w+)\b', line)
            if match:
                skipping = match[1].lower().endswith('_vec')
            if not skipping:
                lines.append(line)
            if line.strip().lower() == 'end':
                skipping = False
    text = '\n'.join(lines)
    text = re.sub(r'(?im)^\s*use\s+driver_vec\s*$', '', text)
    text, count = re.subn(r'(?im)^\s*do\s+i\s*=\s*1\s*,\s*driver_vector_size\s*\n'
                         r'\s*nhel_vec\(:,1:ncomb,i\)\s*=\s*nhel\(:,?:?\)\s*\n'
                         r'\s*enddo\s*$', '', text)
    assert count <= 1
    assert not re.search(r'(?i)\b(?:nhel_vec|driver_vector_size)\b|\buse\s+(?:driver_vec|couplings)\b', text)
    # Keep the original arithmetic and cache branches, independently of the
    # native provider's projection, density contraction and cache adapter.
    for name in ('sborn', 'sborn_sf', 'sborn_hel', 'sborn_onehel'):
        text = re.sub(r'(?i)\b'+name+r'\b', 'ref_'+name+'_explicit', text)
    ng, nh, ns = record['ngraphs'], record['nhelicity'], record['nsqamps']
    common = ["include 'nexternal.inc'", "include 'orders.inc'",
              'real(8) p(0:3,nexternal-1),ans',
              'complex(8) saveamp(%d,%d),ans_cnt(2,nsplitorders)' % (ng, nh),
              'common/to_saveamp/saveamp', 'common/c_born_cnt/ans_cnt']
    bind = ['call init_flv_couplings()'] if model_binding else []
    wrappers = ['subroutine sborn(p,ans)', 'implicit none', *common, *bind,
                'call ref_sborn_explicit(p,ans)', 'end',
                'subroutine sborn_sf(p,m,n,ans)', 'implicit none', *common,
                'integer m,n', 'real(8) soft(amp_split_size)',
                'common/to_amp_split_soft/soft', *bind,
                'call ref_sborn_sf_explicit(p,m,n,ans,ans_cnt,amp_split_cnt,saveamp,soft)', 'end',
                'subroutine sborn_hel(p,ans)', 'implicit none', *common,
                'real(8) hel(%d),helorders(%d,%d)' % (nh,ns,nh),
                'common/c_born_hel/hel', 'common/c_born_hel_split/helorders',
                'hel=0d0', 'helorders=0d0', *bind,
                'call ref_sborn_hel_explicit(p,ans,saveamp,helorders)',
                'hel=sum(helorders,dim=1)', 'end',
                'subroutine sborn_onehel(p,nhel,ihel,ans)', 'implicit none', *common,
                'integer nhel(nexternal-1),ihel', *bind,
                'call ref_sborn_onehel_explicit(p,nhel,ihel,ans,saveamp)', 'end']
    output = source/'scalar_reference.f'
    formatting.write_fortran(output, text.splitlines()+wrappers)
    return output
