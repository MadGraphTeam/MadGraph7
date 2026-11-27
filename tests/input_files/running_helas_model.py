"""Build a small SM variant exercising running custom-propagator inputs."""

import os
import shutil

from madgraph import MG5DIR


def create_model(path, complex_mass=False):
    shutil.copytree(os.path.join(MG5DIR, 'models', 'sm'), path,
                    ignore=lambda _, names: [n for n in names if not n.endswith('.py')])
    with open(os.path.join(path, 'parameters.py'), 'a') as stream:
        stream.write('''
running_base = Parameter(name='running_base', nature='internal', type='real',
                         value='aS / 0.1', texname='r')
running_real = Parameter(name='running_real', nature='internal', type='real',
                         value='running_base**2', texname='r2')
running = Parameter(name='running', nature='internal', type='complex',
                    value='complex(0.25,0.75)', texname='c')
running_complex = Parameter(name='running_complex', nature='internal', type='complex',
                            value='running*running_real', texname='cr2')
running_unused = Parameter(name='running_unused', nature='internal', type='real',
                           value='100*aS', texname='unused')
''')
    width = '' if complex_mass else '+ complex(0,1)*Mass(id)*Width(id)'
    with open(os.path.join(path, 'propagators.py'), 'w') as stream:
        stream.write('''
class Propagator:
    def __init__(self, name, numerator, denominator):
        self.name, self.numerator, self.denominator = name, numerator, denominator
    def __str__(self):
        return self.name

Hrun = Propagator('Hrun', '(1 + running_real + running_complex)',
    '(P(-1,id)**2 - Mass(id)**2 %s)**2')
Zrun = Propagator('Zrun',
    '(-Metric(1,2) + P(1,id)*P(2,id)/Mass(id)**2)*(1 + running_complex + aS + G/(2*cmath.pi))',
    'P(-1,id)**2 - Mass(id)**2*(1 + running_real) + complex(0,1)*Mass(id)*Width(id)')
''' % width)
    with open(os.path.join(path, '__init__.py'), 'a') as stream:
        stream.write('\nfrom . import propagators\n'
                     'particles.H.propagator = propagators.Hrun\n'
                     'particles.Z.propagator = propagators.Zrun\n')
    return path
