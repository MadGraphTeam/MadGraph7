################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Unit test library for the various properties of objects in 
   loop_diagram_generaiton"""

from __future__ import absolute_import
import copy
import itertools
import logging
import math
import os
import sys
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))


import tests.IOTests as IOTests
import tests.unit_tests as unittest


import madgraph.various.misc as misc
import madgraph.core.color_algebra as color
import madgraph.core.drawing as draw_lib
import madgraph.iolibs.drawing_eps as draw
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.iolibs.save_load_object as save_load_object
import models.import_ufo as models
from madgraph import MadGraph5Error

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
_model_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                os.path.pardir,'models')
_color_one = color.ColorString()
#===============================================================================
# Function to load a toy hardcoded Loop Model
#===============================================================================

def loadLoopModel():
    """Setup the NLO model"""
    
    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()

    # A gluon
    mypartlist.append(base_objects.Particle({'name':'g',
                  'antiname':'g',
                  'spin':3,
                  'color':8,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'g',
                  'antitexname':'g',
                  'line':'curly',
                  'charge':0.,
                  'pdg_code':21,
                  'propagating':True,
                  'is_part':True,
                  'self_antipart':True,
                  'counterterm':{('QCD',((),)):{-1:'GWfct'}}}))
    
    # A quark U and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'u',
                  'antiname':'u~',
                  'spin':2,
                  'color':3,
                  'mass':'umass',
                  'width':'zero',
                  'texname':'u',
                  'antitexname':'\bar u',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':2,
                  'propagating':True,
                  'is_part':True,
                  'self_antipart':False}))
    antiu = copy.copy(mypartlist[1])
    antiu.set('is_part', False)
    mypartlist[1].set('counterterm',{('QCD',((),)):{-1:'UQCDWfct'},
                                     ('QED',((),)):{-1:'UQEDWfct'}})

    # A quark D and its antiparticle
    mypartlist.append(base_objects.Particle({'name':'d',
                  'antiname':'d~',
                  'spin':2,
                  'color':3,
                  'mass':'dmass',
                  'width':'zero',
                  'texname':'d',
                  'antitexname':'\bar d',
                  'line':'straight',
                  'charge':-1. / 3.,
                  'pdg_code':1,
                  'propagating':True,
                  'is_part':True,
                  'self_antipart':False}))
    antid = copy.copy(mypartlist[2])
    antid.set('is_part', False)
    mypartlist[2].set('counterterm',{('QCD',((),)):{-1:'DQCDWfct'},
                                     ('QED',((),)):{-1:'DQEDWfct'}})

    # A photon
    mypartlist.append(base_objects.Particle({'name':'a',
                  'antiname':'a',
                  'spin':3,
                  'color':1,
                  'mass':'zero',
                  'width':'zero',
                  'texname':r'\gamma',
                  'antitexname':r'\gamma',
                  'line':'wavy',
                  'charge':0.,
                  'pdg_code':22,
                  'propagating':True,
                  'is_part':True,
                  'self_antipart':True}))

    # A electron and positron
    mypartlist.append(base_objects.Particle({'name':'e-',
                  'antiname':'e+',
                  'spin':2,
                  'color':1,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'e^-',
                  'antitexname':'e^+',
                  'line':'straight',
                  'charge':-1.,
                  'pdg_code':11,
                  'propagating':True,
                  'is_part':True,
                  'self_antipart':False}))
    antie = copy.copy(mypartlist[4])
    antie.set('is_part', False)

    # First set up the base interactions.

    # 3 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 1,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':1}}))

    # 4 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 2,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':2}}))

    # Gluon and photon couplings to quarks
    myinterlist.append(base_objects.Interaction({
                  'id': 3,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 4,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 5,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1}}))

    myinterlist.append(base_objects.Interaction({
                  'id': 6,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    # Coupling of e to gamma

    myinterlist.append(base_objects.Interaction({
                  'id': 7,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1}}))

    # Then set up the R2 interactions proportional to those existing in the
    # tree-level model.

    # 3 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 8,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # 4 gluon vertex
    myinterlist.append(base_objects.Interaction({
                  'id': 9,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # Gluon and photon couplings to quarks
    myinterlist.append(base_objects.Interaction({
                  'id': 10,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))
        
    myinterlist.append(base_objects.Interaction({
                  'id': 11,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 12,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'perturbation_type':'QCD',
                  'loop_particles':[[]],
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 13,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 14,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 15,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':1, 'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 16,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':1, 'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 17,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # Coupling of e to gamma

    myinterlist.append(base_objects.Interaction({
                  'id': 18,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # R2 interactions not proportional to the base interactions

    # Two point interactions

    # The gluon
    myinterlist.append(base_objects.Interaction({
                  'id': 19,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 2),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # The photon
    myinterlist.append(base_objects.Interaction({
                  'id': 20,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 2),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # The electron
    myinterlist.append(base_objects.Interaction({
                  'id': 21,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[4], \
                                         antie]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # The up quark, R2QED
    myinterlist.append(base_objects.Interaction({
                  'id': 22,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # The up quark, R2QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 23,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # The down quark, R2QED
    myinterlist.append(base_objects.Interaction({
                  'id': 24,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antiu]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # The down quark, R2QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 25,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antid]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # The R2 three and four point interactions not proportional to the
    # base interaction

    # 3 photons
    myinterlist.append(base_objects.Interaction({
                  'id': 26,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 3),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # 2 photon and 1 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 27,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0],]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2, 'QCD':1},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # 1 photon and 2 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 28,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0],]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':1, 'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'R2'}))

    # 4 photons
    myinterlist.append(base_objects.Interaction({
                  'id': 29,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3]] * 4),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':4},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'R2'}))

    # 3 photons and 1 gluon
    myinterlist.append(base_objects.Interaction({
                  'id': 30,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':3,'QCD':1},
                  'loop_particles':[[]],
                  'type':'R2'}))

    # 2 photons and 2 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 31,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2,'QCD':2},
                  'loop_particles':[[]],
                  'type':'R2'}))

    # 1 photon and 3 gluons
    myinterlist.append(base_objects.Interaction({
                  'id': 32,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[3],\
                                        mypartlist[0],\
                                        mypartlist[0],\
                                        mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':1,'QCD':3},
                  'loop_particles':[[]],
                  'type':'R2'})) 

    # Finally the UV interactions Counter-Terms

    # 3 gluon vertex CT
    myinterlist.append(base_objects.Interaction({
                  'id': 33,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 3),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))

    # 4 gluon vertex CT
    myinterlist.append(base_objects.Interaction({
                  'id': 34,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[0]] * 4),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G^2'},
                  'orders':{'QCD':4},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))

    # Gluon and photon couplings to quarks CT
    myinterlist.append(base_objects.Interaction({
                  'id': 35,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))
    
    # this is the CT for the renormalization of the QED corrections to alpha_QCD
    myinterlist.append(base_objects.Interaction({
                  'id': 36,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QED':2,'QCD':1},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 37,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QCD':2,'QED':1},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 38,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[1], \
                                         antiu, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 39,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QCD':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 40,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[0]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQQ'},
                  'orders':{'QED':2,'QCD':1},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 41,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QCD':2,'QED':1},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVtree1eps'}))

    myinterlist.append(base_objects.Interaction({
                  'id': 42,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[2], \
                                         antid, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVtree1eps'}))
    
    # alpha_QED to electron CT

    myinterlist.append(base_objects.Interaction({
                  'id': 43,
                  'particles': base_objects.ParticleList(\
                                        [mypartlist[4], \
                                         antie, \
                                         mypartlist[3]]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'GQED'},
                  'orders':{'QED':3},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVtree1eps'}))
      
    # The mass renormalization of the up and down quark granted
    # a mass for the occasion

    # The up quark, UVQED
    myinterlist.append(base_objects.Interaction({
                  'id': 44,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVmass1eps'}))

    # The up quark, UVQCD
    myinterlist.append(base_objects.Interaction({
                  'id': 45,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[2], \
                                         antid]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVmass1eps'}))

    # The down quark, UVQED
    myinterlist.append(base_objects.Interaction({
                  'id': 46,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antiu]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QED':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QED',
                  'type':'UVmass1eps'}))

    # The down quark, UVQCD
    myinterlist.append(base_objects.Interaction({
                  'id': 47,
                  'particles': base_objects.ParticleList([\
                                        mypartlist[1], \
                                         antiu]),
                  'color': [_color_one],
                  'lorentz':['L1'],
                  'couplings':{(0, 0):'G'},
                  'orders':{'QCD':2},
                  'loop_particles':[[]],
                  'perturbation_type':'QCD',
                  'type':'UVmass1eps'}))



    myloopmodel.set('particles', mypartlist)
    myloopmodel.set('couplings', ['QCD','QED'])        
    myloopmodel.set('interactions', myinterlist)
    myloopmodel.set('perturbation_couplings', ['QCD','QED'])
    myloopmodel.set('order_hierarchy', {'QCD':1,'QED':2})

    return myloopmodel    


#===============================================================================
# LoopDiagramGeneration Test
#===============================================================================

class LoopDiagramGenerationTest(unittest.TestCase):
    """Test class for all functions related to the Loop diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()
    
    ref_dict_to0 = {}
    ref_dict_to1 = {}

    myamplitude = diagram_generation.Amplitude()

    def setUp(self):
        """Load different objects for the tests."""
        
        #self.myloopmodel = models.import_full_model(os.path.join(\
        #    _model_file_path,'loop_sm'))
        #self.myloopmodel.actualize_dictionaries()
        self.myloopmodel = loadLoopModel()
        
        self.mypartlist = self.myloopmodel['particles']
        self.myinterlist = self.myloopmodel['interactions']
        self.ref_dict_to0 = self.myinterlist.generate_ref_dict(['QCD','QED'])[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict(['QCD','QED'])[1]
        
    def test_NLOAmplitude(self):
        """test different features of the NLOAmplitude class"""
        ampNLOlist=[]
        ampdefaultlist=[]

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
        dummyproc = base_objects.Process({'legs':myleglist,
                                          'model':self.myloopmodel})

        ampdefaultlist.append(diagram_generation.Amplitude())
        ampdefaultlist.append(diagram_generation.Amplitude(dummyproc))
        ampdefaultlist.append(diagram_generation.Amplitude({'process':dummyproc}))        
        ampdefaultlist.append(diagram_generation.DecayChainAmplitude(dummyproc,False))

        dummyproc.set("perturbation_couplings",['QCD','QED'])
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude({'process':dummyproc}))                
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude())        
        ampNLOlist.append(loop_diagram_generation.LoopAmplitude(dummyproc))        

        # Test the __new__ constructor of NLOAmplitude
        for ampdefault in ampdefaultlist:
            self.assertIsInstance(ampdefault, diagram_generation.Amplitude)
            self.assertNotIsInstance(ampdefault, loop_diagram_generation.LoopAmplitude)
        for ampNLO in ampNLOlist:
            self.assertIsInstance(ampNLO, loop_diagram_generation.LoopAmplitude)

        # Now test for the usage of getter/setter of diagrams.
        ampNLO=loop_diagram_generation.LoopAmplitude(dummyproc)
        mydiaglist=base_objects.DiagramList([loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':1}),\
                                             loop_base_objects.LoopDiagram({'type':2}),\
                                             loop_base_objects.LoopDiagram({'type':3}),\
                                             loop_base_objects.LoopDiagram({'type':4}),\
                                             loop_base_objects.LoopUVCTDiagram()])        
        ampNLO.set('diagrams',mydiaglist)
        self.assertEqual(len(ampNLO.get('diagrams')),11)
        self.assertEqual(len(ampNLO.get('born_diagrams')),6)
        self.assertEqual(len(ampNLO.get('loop_diagrams')),4)  
        self.assertEqual(len(ampNLO.get('loop_UVCT_diagrams')),1)              
        mydiaglist=base_objects.DiagramList([loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0}),\
                                             loop_base_objects.LoopDiagram({'type':0})])
        ampNLO.set('born_diagrams',mydiaglist)
        self.assertEqual(len(ampNLO.get('born_diagrams')),3)        

    def test_diagram_generation_epem_ddx(self):
        """Test the number of loop diagrams generated for e+e->dd~ (s channel)
           with different choices for the perturbation couplings and squared orders.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        ordersChoices=[({},['QCD'],{},1),\
                       ({},['QED'],{},7),\
                       ({},['QCD','QED'],{},8),\
                       ({},['QED','QCD'],{'QED':-1},1),\
                       ({},['QED','QCD'],{'QCD':-1},7)]
        for (bornOrders,pert,sqOrders,nDiagGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            
            #self.assertEqual(len([1 for diag in \
            #  myloopamplitude.get('loop_diagrams') if not isinstance(diag,
            #  loop_base_objects.LoopWavefunctionCTDiagram)]),nDiagGoal)

            ### This is to plot the diagrams obtained
            #diaglist=[diag for diag in \
            #  myloopamplitude.get('loop_diagrams') if not isinstance(diag,
            #  loop_base_objects.LoopUVCTDiagram)]
            #diaglist=myloopamplitude.get('loop_diagrams')
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/tmp/' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(diaglist),#myloopamplitude['loop_diagrams'],
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                 amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)

            ### This is to display some informations
            #mydiag1=myloopamplitude.get('loop_diagrams')[0]
            #mydiag2=myloopamplitude.get('loop_diagrams')[5]      
            #print "I got tag for diag 1=",mydiag1['canonical_tag']
            #print "I got tag for diag 2=",mydiag2['canonical_tag']
            #print "I got vertices for diag 1=",mydiag1['vertices']
            #print "I got vertices for diag 2=",mydiag2['vertices']
            #print "mydiag=",str(mydiag)
            #mydiag1.tag(trial,5,6,self.myloopmodel)
            #print "I got tag=",mydiag['tag']
            #print "I got struct[0]=\n",myloopamplitude['structure_repository'][0].nice_string()
            #print "I got struct[2]=\n",myloopamplitude['structure_repository'][2].nice_string()   
            #print "I got struct[3]=\n",myloopamplitude['structure_repository'][3].nice_string()

    def test_diagram_generation_uux_ga(self):
        """Test the number of loop diagrams generated for uu~>g gamma (s channel)
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},19,10,4,18), 
                       ({},['QCD'],{},11,4,2,10)]
        
                
        for (bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVmassGoal,nUVCTGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
#            self.assertEqual(sumUV, nUVmassGoal)
            sumUVCT=0
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUVCT+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUVCT,nUVCTGoal)

    def test_diagram_generation_gg_ng(self):
        """Test the number of loop diagrams generated for gg>ng. n being in [1,2,3]
        """
        
        # For quick test 
        nGluons = [(1,8,0,1,4),(2,81,0,10,23)]
        # For a longer one
        # (still 4 need to be re-tested)
        # nGluons += [(3,905,0,105,190),(4,11850,0,1290,2075)]

        for (n, nDiagGoal, nUVmassGoal, nR2Goal, nUVCTGoal) in nGluons:
            myleglist=base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, (n+3))])
            myleglist[0].set('state',False)
            myleglist[1].set('state',False)        

            myproc=base_objects.Process({'legs':myleglist,
                                       'model':self.myloopmodel,
                                       'orders':{},
                                       'squared_orders': {},
                                       'perturbation_couplings':['QCD']})

            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            sumUVCT=0
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUVCT+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')), nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVmassGoal)   
            self.assertEqual(sumUVCT, nUVCTGoal)                        

    def test_diagram_generation_uux_ddx(self):
        """Test the number of loop diagrams generated for uu~>dd~ for different choices
           of orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        ordersChoices=[({},['QCD','QED'],{},17,7,6,12),\
                       ({},['QCD','QED',],{'QED':-99},24,10,8,16),\
                       ({},['QCD'],{},9,3,2,4),\
                       ({},['QED'],{},2,2,2,4),\
                       ({'QED':0},['QCD'],{},9,3,2,4),\
                       ({'QCD':0},['QED'],{},7,3,2,4),\
                       ({},['QCD','QED'],{'QED':-1},9,3,2,4),\
                       ({},['QCD','QED'],{'QCD':-1},7,3,2,4),\
                       ({},['QCD','QED'],{'QED':-2},17,7,6,12),\
                       ({},['QCD','QED'],{'QED':-3},24,10,8,16)]
        
        for (bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVGoal,nUVWfctGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            
            sumUVwfct=0
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                for coupl in loop_UVCT_diag.get('UVCT_couplings'):
                    if not isinstance(coupl,str) or not 'Wfct' in coupl:
                        sumUV+=1
                    else:
                        sumUVwfct+=1
                    
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')), nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVGoal)  
            self.assertEqual(sumUVwfct, nUVWfctGoal)          

    def test_diagram_generation_ddx_ddx(self):
        """Test the number of loop diagrams generated for dd~>dd~ for different choices
           of orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        ordersChoices=[({},['QCD'],{},18,6,0,12),\
                       ({},['QED'],{},4,4,0,12),\
                       ({},['QCD','QED'],{},34,14,0,36),\
                       ({},['QCD','QED',],{'QED':-99},48,20,0,48),\
                       ({'QED':0},['QCD'],{},18,6,0,12),\
                       ({'QCD':0},['QED'],{},14,6,0,12),\
                       ({},['QCD','QED'],{'QED':-1},18,6,0,12),\
                       ({},['QCD','QED'],{'QCD':-1},14,6,0,12)]
        
        for (bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVmassGoal,nUVCTGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            #print "testing:",bornOrders,pert,sqOrders,nDiagGoal,nR2Goal,nUVmassGoal,nUVCTGoal
            #print "I got diagrams"
            #for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
            #    print "diagram #%i is %s"%(i,diag.nice_string())
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVmassGoal)
            sumUVCT=0
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUVCT+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUVCT,nUVCTGoal)
            
    def test_CT_vertices_generation_gg_gg(self):
        """ test that the Counter Term vertices are correctly
            generated by adding some new CT interactions to the model and
            comparing how many CT vertices are generated on the 
            process gg_gg for different R2 specifications. """
            
        newLoopModel=copy.deepcopy(self.myloopmodel)
        newInteractionList=base_objects.InteractionList()
        for inter in newLoopModel['interactions']:
            if inter['type']=='base':
                newInteractionList.append(inter)
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
                
        newInteractionList.append(base_objects.Interaction({
                      'id': 666,
                      # a dd~d~ R2
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]]*4),
                      'color': [_color_one],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':4},
                      # We don't specify the loop content here
                      'loop_particles':[[]],
                      'type':'R2'}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':newLoopModel,
                                       'orders':{},
                                       'perturbation_couplings':['QCD'],
                                       'squared_orders':{'WEIGHTED':99}})

        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        CTChoice=[([[]],{'QCD':4},1),
                  ([[21,]],{'QCD':4},1),
                  ([[1,]],{'QCD':4},1),
                  ([[2,]],{'QCD':4},1),
                  ([[2,],[21,],[1,]],{'QCD':4},3), 
                  ([[2,],[21,2],[1,],[22,21,1]],{'QCD':4},2),                 
                  ([[21,]],{'QCD':4,'QED':1},0),
                  ([[1,2]],{'QCD':4},0)]
        
        for (parts,orders,nCTGoal) in CTChoice:
            newInteractionList[-1]['loop_particles']=parts
            newInteractionList[-1]['orders']=orders            
            newLoopModel.set('interactions',newInteractionList)
            myloopamplitude['process']['model']=newLoopModel
            for diag in myloopamplitude.get('loop_diagrams'):
                diag['CT_vertices']=base_objects.VertexList()
            myloopamplitude.set_LoopCT_vertices()            
            sumR2=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(newLoopModel,'R2'))
            self.assertEqual(sumR2, nCTGoal)

    def test_diagram_generation_ddxuux_split_orders(self):
        """ Test the implementation of the various way of specifying the squared
        order constraints at NLO, using the process  d d~ > u u~ as reference. """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,'state':False}))
        myleglist.append(base_objects.Leg({'id':2,'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,'state':True}))

        ordersChoices=[
          ({},['QCD'],{},{},9,3,0,6,[(4,0,4)],[],[(6,0,6)],[]),\
          ({},['QED'],{},{},2,2,0,6,[(4,0,4)],[],[(4,2,8)],[]),\
          ({},['QCD'],{'QED':-1},{'QED':'>'},11,5,0,12,
           [(2,2,6),(0,4,8)],[(4,0,4)],[(4,2,8),(2,4,10)],[(6,0,6)]),\
          ({},['QCD','QED'],{},{},17,7,0,18,
           [(4,0,4),(2,2,6),(0,4,8)],[],[(6,0,6),(4,2,8),(2,4,10)],[]),
          ({},['QCD','QED'],{'QED':-1},{'QED':'>'},24,10,0,24,
           [(2,2,6),(0,4,8)],[(4,0,4)],[(4,2,8),(2,4,10),(0,6,12)],[(6,0,6)]),\
          ({},['QCD','QED'],{'WEIGHTED':8},{'WEIGHTED':'<='},17,7,0,18,
           [(4,0,4),(2,2,6),(0,4,8)],[],[(6,0,6),(4,2,8)],[(2,4,10)]),
          ({},['QCD','QED'],{'WEIGHTED':8},{'WEIGHTED':'=='},17,7,0,18,
           [(0,4,8)],[(4,0,4),(2,2,6)],[(4,2,8)],[(6,0,6),(2,4,10)]),
          ({},['QCD','QED'],{'QED':-2},{'QED':'=='},17,7,0,18,
           [(2,2,6)],[(4,0,4),(0,4,8)],[(4,2,8)],[(6,0,6),(2,4,10)]),
          ({},['QCD','QED'],{'QED':2},{'QED':'>'},15,7,0,18,
           [(0,4,8)],[(4,0,4),(2,2,6)],[(2,4,10),(0,6,12)],[(4,2,8)]),
          ({'QCD':2},['QCD','QED'],{'QCD':-8},{'QCD':'=='},0,0,0,0,[],[],[],[]),
          # The case below is a bit academic but ok
          ({'QCD':0,'QED':0},['QCD','QED'],{},{},8,4,0,0,
           [],[],[(4,4,12)],[]),
        ]
        
        for (bornOrders,pert,sqOrders,sqOrders_types,nDiagGoal,nR2Goal, nUVmassGoal,
              nUVCTGoal,bo_kept,bo_extra,loop_kept,loop_extra) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders,
                                           'sqorders_types':sqOrders_types})
            
#            print " testing ",(bornOrders,pert,sqOrders,sqOrders_types,nDiagGoal,
#              nR2Goal, nUVmassGoal,nUVCTGoal,bo_kept,bo_extra,loop_kept,loop_extra)
            
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
#            print "I got diagrams"
#            for i, diag in enumerate(myloopamplitude.get('born_diagrams')):
#                print "diagram #%i is %s"%(i,diag.nice_string())
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            self.assertEqual(sumR2, nR2Goal)
            self.assertEqual(sumUV, nUVmassGoal)
            sumUVCT=0
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUVCT+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUVCT,nUVCTGoal)
            
            (born_orders_kept, born_orders_extra, loop_orders_kept, 
                  loop_orders_extra) = myloopamplitude.print_split_order_infos()
            
            self.assertEqual(born_orders_kept,bo_kept)
            self.assertEqual(born_orders_extra,bo_extra)
            self.assertEqual(loop_orders_kept,loop_kept)
            self.assertEqual(loop_orders_extra,loop_extra)

    def test_CT_vertices_generation_ddx_ddx(self):
        """ test that the Counter Term vertices are correctly
            generated by adding some new CT interactions to the model and
            comparing how many CT vertices are generated on the 
            process ddx_ddx for different R2 specifications. """
            
        newLoopModel=copy.deepcopy(self.myloopmodel)
        newInteractionList=base_objects.InteractionList()
        for inter in newLoopModel['interactions']:
            if inter['type']=='base':
                newInteractionList.append(inter)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
                
        antid=copy.copy(self.mypartlist[2])
        antid.set('is_part',False)
        newInteractionList.append(base_objects.Interaction({
                      'id': 666,
                      # a dd~d~ R2
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2],
                                             antid,
                                             self.mypartlist[2],
                                             antid,]),
                      'color': [_color_one],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':4},
                      # We don't specify the loop content here
                      'loop_particles':[[]],
                      'type':'R2'}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':newLoopModel,
                                       'orders':{},
                                       'perturbation_couplings':['QCD','QED'],
                                       'squared_orders':{'WEIGHTED':99}})
    
        myloopamplitude = loop_diagram_generation.LoopAmplitude()
        myloopamplitude.set('process', myproc)
        myloopamplitude.generate_diagrams()
        
        CTChoice=[([[]],{'QCD':4},1),
                  ([[1,21]],{'QCD':4},1),
                  ([[1,22]],{'QED':4},1),
                  ([[1,22],[1,22]],{'QED':4},2),
                  ([[1,21]],{'QED':4},0),
                  ([[1,22]],{'QCD':4},0),
                  ([[21,]],{'QCD':4},0)]
        
        for (parts,orders,nCTGoal) in CTChoice:
            newInteractionList[-1]['loop_particles']=parts
            newInteractionList[-1]['orders']=orders            
            newLoopModel.set('interactions',newInteractionList)
            myloopamplitude['process']['model']=newLoopModel
            for diag in myloopamplitude.get('loop_diagrams'):
                diag['CT_vertices']=base_objects.VertexList()
            myloopamplitude.set_LoopCT_vertices()            
            sumR2=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(newLoopModel,'R2'))
            self.assertEqual(sumR2, nCTGoal)
            
#===============================================================================
# LoopDiagramFDStruct Test
#===============================================================================
class LoopDiagramFDStructTest(unittest.TestCase):
    """Test class for the tagging functions of LoopDiagram and FDStructure classes"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myproc = base_objects.Process()
    myloopdiag = loop_base_objects.LoopDiagram()

    def setUp(self):
        """ Setup a toy-model with gluon and down-quark only """

        # A gluon
        self.mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark D and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'dmass',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[1])
        antid.set('is_part', False)

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [_color_one],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [_color_one],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon coupling to the down-quark
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [_color_one],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)
        self.myproc.set('model',self.mymodel)

    def test_loop_identification_tag_supports_flavor_couplings(self):
        """Loop-identification tags must contain immutable coupling data.

        A FLV_Coupling is a mutable PhysicsObject and therefore cannot itself
        be a dictionary key.  Its canonical key must also distinguish two
        physically different flavor tables even if their generated names are
        the same.
        """
        first = base_objects.FLV_Coupling(
            'FLV_TEST', {(1, 1): 'GC_1', (2, 2): 'GC_2'})
        second = base_objects.FLV_Coupling(
            'FLV_TEST', {(1, 1): 'GC_1', (2, 2): 'GC_3'})

        interaction = self.mymodel.get_interaction(3)
        original = interaction.get('couplings')[(0, 0)]
        interaction.get('couplings')[(0, 0)] = first
        try:
            diagram = loop_base_objects.LoopDiagram()
            diagram.set('canonical_tag', [[1, [], 3]])
            tag_first = diagram.build_loop_tag_for_diagram_identification(
                self.mymodel, loop_base_objects.FDStructureList())
            hash(tag_first)

            interaction.get('couplings')[(0, 0)] = second
            tag_second = diagram.build_loop_tag_for_diagram_identification(
                self.mymodel, loop_base_objects.FDStructureList())
            hash(tag_second)
        finally:
            interaction.get('couplings')[(0, 0)] = original

        self.assertNotEqual(tag_first, tag_second)

    def test_gg_5gglgl_bubble_tag(self):
        """ Test the gg>ggggg g*g* tagging of a bubble"""

        # Five gluon legs with two initial states
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 10)])
        myleglist[7].set('loop_line', True)
        myleglist[8].set('loop_line', True)
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]
        l7=myleglist[6]
        l8=myleglist[7]
        l9=myleglist[8]
        lfinal=copy.copy(l9)
        lfinal.set('number',9)

        self.myproc.set('legs',myleglist)

        l67 = base_objects.Leg({'id':21,'number':6,'loop_line':False})
        l56 = base_objects.Leg({'id':21,'number':5,'loop_line':False})
        l235 = base_objects.Leg({'id':21,'number':2,'loop_line':False}) 
        l24 = base_objects.Leg({'id':21,'number':2,'loop_line':False})
        l28 = base_objects.Leg({'id':21,'number':2,'loop_line':True})
        l128 = base_objects.Leg({'id':21,'number':1,'loop_line':True})
        l19 = base_objects.Leg({'id':21,'number':1,'loop_line':True})
        l18 = base_objects.Leg({'id':21,'number':1,'loop_line':True})
        l12 = base_objects.Leg({'id':21,'number':1,'loop_line':True})

        vx19 = base_objects.Vertex({'legs':base_objects.LegList([l1, l9, l19]), 'id': 1})
        vx67 = base_objects.Vertex({'legs':base_objects.LegList([l6, l7, l67]), 'id': 1})
        vx56 = base_objects.Vertex({'legs':base_objects.LegList([l5, l67, l56]), 'id': 1})
        vx235 = base_objects.Vertex({'legs':base_objects.LegList([l2, l3, l56, l235]), 'id': 2})
        vx24 = base_objects.Vertex({'legs':base_objects.LegList([l4, l235, l24]), 'id': 1})
        vx28 = base_objects.Vertex({'legs':base_objects.LegList([l235, l8, l28]), 'id': 1})
        vx0 = base_objects.Vertex({'legs':base_objects.LegList([l19, l28]), 'id': 0})

        myVertexList=base_objects.VertexList([vx19,vx67,vx56,vx235,vx24,vx28,vx0])

        myBubbleDiag=loop_base_objects.LoopDiagram({'vertices':myVertexList,'type':21})

        myStructRep=loop_base_objects.FDStructureList()
        myStruct=loop_base_objects.FDStructure()

        goal_canonicalStruct=(((2, 3, 4, 5, 6, 7), 1), ((2, 3, 5, 6, 7), 2), ((5, 6, 7), 1), ((6, 7), 1))
        canonicalStruct=myBubbleDiag.construct_FDStructure(5, 0, 2, myStruct)
        self.assertEqual(canonicalStruct, goal_canonicalStruct)
        
        goal_vxList=base_objects.VertexList([vx67,vx56,vx235,vx24])
        myStruct.set('canonical',canonicalStruct)
        myStruct.generate_vertices(self.myproc)
        self.assertEqual(myStruct['vertices'],goal_vxList)

        goal_tag=[[21, [1], 1], [21, [0], 1]]
        vx18_tag=base_objects.Vertex({'legs':base_objects.LegList([l1, l8, l18]), 'id': 1})
        vx12_tag=base_objects.Vertex({'legs':base_objects.LegList([l24, l18, l12]), 'id': 1})
        closing_vx=base_objects.Vertex({'legs':base_objects.LegList([l12, lfinal]), 'id': -1})
        goal_vertices=base_objects.VertexList([vx18_tag,vx12_tag,closing_vx])
        myBubbleDiag.tag(myStructRep,self.myproc['model'],8,9)
        self.assertEqual(myBubbleDiag.get('canonical_tag'), goal_tag)
        self.assertEqual(myBubbleDiag.get('vertices'), goal_vertices)

    def test_gg_4gdldxl_penta_tag(self):
        """ Test the gg>gggg d*dx* tagging of a quark pentagon"""

        # Five gluon legs with two initial states
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'loop_line':False}) \
                                              for num in range(1, 7)])
        myleglist.append(base_objects.Leg({'id':1,'number':7,'loop_line':True}))
        myleglist.append(base_objects.Leg({'id':-1,'number':8,'loop_line':True}))                         
        l1=myleglist[0]
        l2=myleglist[1]
        l3=myleglist[2]
        l4=myleglist[3]
        l5=myleglist[4]
        l6=myleglist[5]
        l7=myleglist[6]
        l8=myleglist[7]

        self.myproc.set('legs',myleglist)

        # One way of constructing this diagram, with a three-point amplitude
        l17 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l12 = base_objects.Leg({'id':1,'number':1,'loop_line':True})
        l68 = base_objects.Leg({'id':-1,'number':6,'loop_line':True}) 
        l56 = base_objects.Leg({'id':-1,'number':5,'loop_line':True})
        l34 = base_objects.Leg({'id':21,'number':3,'loop_line':False})
        l617 = base_objects.Leg({'id':1,'number':1,'loop_line':True})

        vx17 = base_objects.Vertex({'legs':base_objects.LegList([l1, l7, l17]), 'id': 3})
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l17, l2, l12]), 'id': 3})
        vx68 = base_objects.Vertex({'legs':base_objects.LegList([l6, l8, l68]), 'id': 3})
        vx56 = base_objects.Vertex({'legs':base_objects.LegList([l5, l68, l56]), 'id': 3})
        vx34 = base_objects.Vertex({'legs':base_objects.LegList([l3, l4, l34]), 'id': 1})
        vx135 = base_objects.Vertex({'legs':base_objects.LegList([l12, l56, l34]), 'id': 3})

        myVertexList1=base_objects.VertexList([vx17,vx12,vx68,vx56,vx34,vx135])

        myPentaDiag1=loop_base_objects.LoopDiagram({'vertices':myVertexList1,'type':1})

        myStructRep=loop_base_objects.FDStructureList()
        myStruct=loop_base_objects.FDStructure()
        
        goal_tag=[[1, [0], 3], [1, [1], 3], [1, [2], 3], [1, [3], 3], [1, [4], 3]]
        myPentaDiag1.tag(myStructRep, self.myproc['model'], 7, 8)
        self.assertEqual(myPentaDiag1.get('canonical_tag'), goal_tag)

        vx17_tag=base_objects.Vertex({'legs':base_objects.LegList([l1, l7, l17]), 'id': 3})
        vx12_tag=base_objects.Vertex({'legs':base_objects.LegList([l2, l17, l12]), 'id': 3})
        vx13_tag=base_objects.Vertex({'legs':base_objects.LegList([l34, l12, l17]), 'id': 3})
        vx15_tag=base_objects.Vertex({'legs':base_objects.LegList([l5, l17, l17]), 'id': 3})
        vx168_tag=base_objects.Vertex({'legs':base_objects.LegList([l6, l17, l617]), 'id': 3})  
        closing_vx=base_objects.Vertex({'legs':base_objects.LegList([l617, l8]), 'id': -1})              
        goal_vertices=base_objects.VertexList([vx17_tag,vx12_tag,vx13_tag,vx15_tag,vx168_tag,closing_vx])
        self.assertEqual(myPentaDiag1.get('vertices'), goal_vertices)

class LoopEWDiagramGenerationTest(unittest.TestCase):
    """Test class for all functions related to the Loop diagram generation with model LoopSMEWTest."""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    myloopmodel = loop_base_objects.LoopModel()
    
    ref_dict_to0 = {}
    ref_dict_to1 = {}

    myamplitude = diagram_generation.Amplitude()

    @IOTests.set_global(unitary=False)
    def setUp(self):
        """Load different objects for the tests."""
        
        # Make sure to only load the model once.  The reference diagram
        # counts are for physical (ungrouped) flavours.
        if len(self.myloopmodel['particles'])==0:
            self.myloopmodel = models.import_model(os.path.join(\
            _input_file_path,'LoopSMEWTest'),
            options={'apply_flavor_grouping': False})
        self.myloopmodel.actualize_dictionaries()
        
        self.mypartlist = self.myloopmodel['particles']
        self.myinterlist = self.myloopmodel['interactions']
        self.ref_dict_to0 = self.myloopmodel['ref_dict_to0']
        self.ref_dict_to1 = self.myloopmodel['ref_dict_to1']
        
    def test_diagram_generation_aa_ttx_EW(self):
        """Test the number of loop diagrams generated for a a > t t~
           with different choices for the perturbation couplings and squared orders.
        """
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))

        ordersChoices=[({},['QCD'],{},8),\
                       ({},['QED'],{},198),\
                       ({},['QCD','QED'],{},206)]
        for (bornOrders,pert,sqOrders,nDiagGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nDiagGoal)
            

            ### This is to plot the diagrams obtained
            #diaglist=[diag for diag in \
            #  myloopamplitude.get('loop_diagrams') if not isinstance(diag,
            #  loop_base_objects.LoopUVCTDiagram)]
            #diaglist=myloopamplitude.get('loop_diagrams')
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/tmp/' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(diaglist),#myloopamplitude['loop_diagrams'],
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
    def test_diagram_generation_gg_ttxh_EW(self):
        """Test the number of loop diagrams generated for g g > t t~ h
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},625,171,260),
                       ({},['QCD'],{},140,71,188),
                       ({},['QED'],{},485,100,72)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_epem_ttxa_EW(self):
        """Test the number of loop diagrams generated for e+ e- > t t~ a
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},941,256,154),
                       ({},['QCD'],{},20,16,40),
                       ({},['QED'],{},921,240,114)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_epem_ttxg_EW(self):
        """Test the number of loop diagrams generated for e+ e- > t t~ g
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},363,136,108),
                       ({},['QCD'],{},28,18,52),
                       ({},['QED'],{},335,118,56)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_gg_ttxg_EW(self):
        """Test the number of loop diagrams generated for g g > t t~ g
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},978,413,535),
                       ({},['QCD'],{},384,234,431),
                       ({},['QED'],{},594,179,104)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_ttx_wpwm_EW(self):
        """Test the number of loop diagrams generated for t t~ > w+ w-
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},409,77,50),
                       ({},['QCD'],{},7,6,18),
                       ({},['QED'],{},402,71,32)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)
            
    def test_diagram_generation_aa_wpwm_EW(self):
        """Test the number of loop diagrams generated for a a > w+ w-
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))

        ordersChoices=[
                       ({},['QED'],{},553,63,34)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)
            
    def test_diagram_generation_uux_epem_EW(self):
        """Test the number of loop diagrams generated for u u~>e+ e-
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD','QED'],{},79,53,16),
                       ({},['QCD'],{},2,2,0),
                       ({},['QED'],{},77,51,16)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)
                        
    def test_diagram_generation_uux_ga_EW(self):
        """Test the number of loop diagrams generated for uu~>g a
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD'],{},11,6,14),
                       ({},['QED'],{},27,12,12),
                       ({},['QCD','QED'],{},38,18,26)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_hh_hh_EW(self):
        """Test the number of loop diagrams generated for h h>h h
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))

        ordersChoices=[
                       ({},['QED'],{},582,20,20)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

    def test_diagram_generation_hh_hhh_EW(self):
        """Test the number of loop diagrams generated for h h>h h h
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':True}))

        ordersChoices=[
                       ({},['QED'],{},7841,210,210)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)        
            
    def test_diagram_generation_gg_ggg_EW(self):
        """Test the number of loop diagrams generated for g g>g g g
           with different choices for the perturbation couplings and squared orders.
        """

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        ordersChoices=[
                       ({},['QCD'],{},1139,735,575)]
        
                
        for (bornOrders,pert,sqOrders,nLoopGoal,nR2Goal,nUVGoal) in ordersChoices:
            myproc = base_objects.Process({'legs':copy.copy(myleglist),
                                           'model':self.myloopmodel,
                                           'orders':bornOrders,
                                           'perturbation_couplings':pert,
                                           'squared_orders':sqOrders})
    
            myloopamplitude = loop_diagram_generation.LoopAmplitude()
            myloopamplitude.set('process', myproc)
            myloopamplitude.generate_diagrams()
            
            ### This is to plot the diagrams obtained
            #options = draw_lib.DrawOption()
            #filename = os.path.join('/Users/erdissshaw/Works', 'diagramsVall1_' + \
            #              myloopamplitude.get('process').shell_string() + ".eps")
            #plot = draw.MultiEpsDiagramDrawer(myloopamplitude.get('diagrams'),
            #                                  filename,
            #                                  model=self.myloopmodel,
            #                                  amplitude=myloopamplitude,
            #                                  legend=myloopamplitude.get('process').input_string())
            #plot.draw(opt=options)
            
            sumR2=0
            sumUV=0
            for i, diag in enumerate(myloopamplitude.get('loop_diagrams')):
                sumR2+=len(diag.get_CT(self.myloopmodel,'R2'))
                sumUV+=len(diag.get_CT(self.myloopmodel,'UV'))
            self.assertEqual(len(myloopamplitude.get('loop_diagrams')),nLoopGoal)
            self.assertEqual(sumR2, nR2Goal)
            for loop_UVCT_diag in myloopamplitude.get('loop_UVCT_diagrams'):
                sumUV+=len(loop_UVCT_diag.get('UVCT_couplings'))
            self.assertEqual(sumUV,nUVGoal)

class GroupedPhysicalLoopTest(unittest.TestCase):
    """Closed fermion cycles must retain physical charges and CKM choices."""

    def test_forbidden_ckm_counterterms_match_physical_rows(self):
        from collections import Counter

        physical = models.import_model(os.path.join(_input_file_path, 'LoopSMEWTest'),
            restrict=False, options={'apply_flavor_grouping': False})
        for pdg in (1, 2, 3, 4):
            physical.get_particle(pdg).set('mass', 'ZERO')
        physical.reset_dictionaries()
        grouped = copy.deepcopy(physical)
        for original, duplicate in zip(physical['interactions'], grouped['interactions']):
            duplicate.set('color', original['color'])
        grouped.merge_flavor([1, 2, 3, 4])
        grouped.reset_dictionaries()

        def count(model, quark, forbidden):
            model.actualize_dictionaries()
            q = next((merged for merged, ids in model['merged_particles'].items()
                      if quark in ids), quark)
            process = base_objects.Process({'model': model,
                'legs': base_objects.LegList([base_objects.Leg({'id': pdg,
                    'number': n, 'state': n > 2})
                    for n, pdg in enumerate((q, -q, 11, -11), 1)]),
                'orders': {'QCD': 0}, 'perturbation_couplings': ['QED'],
                'squared_orders': {}, 'forbidden_particles': forbidden})
            amplitude = loop_diagram_generation.LoopAmplitude()
            amplitude.set('process', process)
            amplitude.generate_diagrams()
            counts = Counter()
            for diagram in amplitude['loop_diagrams']:
                for vertex in diagram['CT_vertices']:
                    inter = model.get_interaction(vertex['id'])
                    flavor = tuple(quark if abs(p.get_pdg_code()) == 81 else 0
                                   for p in inter['particles'])
                    for coupling in inter['couplings'].values():
                        if isinstance(coupling, base_objects.FLV_Coupling):
                            coupling = coupling['flavors'].get(flavor)
                        if coupling:
                            counts[inter['type'], coupling] += 1
            return counts

        for forbidden in ([3], [1, 3]):
            for quark in (2, 3, 4, 6):
                with self.subTest(forbidden=forbidden, quark=quark):
                    # An excluded internal species is still a legal external row.
                    self.assertEqual(count(grouped, quark, forbidden),
                                     count(physical, quark, forbidden))

    def test_closed_ckm_cycles_match_ungrouped_diagrams(self):
        from collections import Counter
        from unittest.mock import patch

        model = models.import_model(os.path.join(
            _input_file_path, 'LoopSMEWTest'), restrict=False,
            options={'apply_flavor_grouping': False})
        # Keep all symbolic CKM entries, but give the four light quarks the
        # identical kinematic properties required for grouping.
        for pdg in (1, 2, 3, 4):
            model.get_particle(pdg).set('mass', 'ZERO')
        model.reset_dictionaries()
        grouped = copy.deepcopy(model)
        # Color algebra objects must not be deep-copied (same precaution as
        # Model.merge_flavor itself).
        for original, duplicate in zip(model['interactions'],
                                       grouped['interactions']):
            duplicate.set('color', original.get('color'))
        grouped.merge_flavor([1, 2, 3, 4])
        grouped.reset_dictionaries()

        def cycles(current, forbidden, incoming=21):
            current.actualize_dictionaries()
            process = base_objects.Process({
                'legs': base_objects.LegList([
                    base_objects.Leg({'id': pdg, 'number': number,
                                      'state': number > 2})
                    for number, pdg in enumerate((incoming, incoming, 24, -24), 1)]),
                'model': current, 'orders': {}, 'has_born': incoming == 22,
                'perturbation_couplings': ['QCD' if incoming == 21 else 'QED'],
                'squared_orders': {},
                'forbidden_particles': forbidden})
            amplitude = loop_diagram_generation.LoopAmplitude()
            amplitude.set('process', process)
            # Compare physical diagrams before numerical identification can
            # combine equal flavours into a multiplier.
            with patch.object(loop_diagram_generation.LoopAmplitude,
                              'identify_loop_diagrams', return_value=0):
                amplitude.generate_diagrams()
            if current.get('merged_particles'):
                registry = current._loop_interactions.copy()
                self.assertTrue(registry)
                self.assertFalse(set(registry).intersection(
                    inter['id'] for inter in current['interactions']))
                # Rebuilding caches must retain lookup without adding physical
                # variants to any subsequent tree-generation dictionary.
                current.set('interactions', current['interactions'])
                current.reset_dictionaries()
                current.actualize_dictionaries()
                for key, interaction in registry.items():
                    self.assertIs(current.get_interaction(key), interaction)
                generated = {vertex for values in current.get('ref_dict_to1').values()
                             for unused, vertex in values}
                self.assertFalse(set(registry).intersection(generated))
            signatures = Counter()
            for diagram in amplitude.get('loop_diagrams'):
                if not diagram.is_fermion_loop(current):
                    continue
                sequence = []
                for leg, structures, inter_id in diagram['tag']:
                    self.assertNotIn(abs(leg['id']),
                                     current.get('merged_particles'))
                    self.assertNotIn(abs(leg['id']), forbidden)
                    interaction = current.get_interaction(inter_id)
                    self.assertTrue(all(isinstance(c, str) for c in
                                        interaction['couplings'].values()))
                    bindings = tuple(sorted(
                        (amplitude['structure_repository'][sid]['binding_leg']['id'],
                         tuple(l['number'] for l in
                         amplitude['structure_repository'][sid]['external_legs']))
                        for sid in structures))
                    sequence.append((leg['id'], bindings,
                                     interaction.canonical_repr()))
                signatures[tuple(sequence)] += 1
                self.assertEqual(diagram['vertices'][-1]['legs'][0]['id'],
                                 diagram['tag'][0][0]['id'])
            self.assertTrue(signatures)
            return signatures

        for cutting in ('optimal', 'default'):
            for forbidden in ([], [3]):
                with self.subTest(cutting=cutting, forbidden=forbidden), \
                     patch.object(loop_base_objects.LoopDiagram,
                                  'cutting_method', cutting):
                    self.assertEqual(cycles(grouped, forbidden),
                                     cycles(model, forbidden))

        # Charged-lepton and neutrino merging revisits W interactions. Retain
        # original sources and scalar structures through either merge order.
        for pdg in (11, 13, 15):
            model.get_particle(pdg).set('mass', 'ZERO')
        model.reset_dictionaries()
        for groups in (([11, 13, 15], [12, 14, 16]),
                       ([12, 14, 16], [11, 13, 15])):
            with self.subTest(groups=groups):
                grouped_leptons = copy.deepcopy(model)
                for original, duplicate in zip(model['interactions'],
                                               grouped_leptons['interactions']):
                    duplicate.set('color', original.get('color'))
                for group in groups:
                    grouped_leptons.merge_flavor(group)
                grouped_leptons.reset_dictionaries()
                self.assertEqual(cycles(grouped_leptons, [], incoming=22),
                                 cycles(model, [], incoming=22))


class GroupedLoopCounterTermTest(unittest.TestCase):
    """Counterterms and closed loops of merged (flavour-grouped) quarks."""

    grouped_loop_sm = None

    def setUp(self):
        if GroupedLoopCounterTermTest.grouped_loop_sm is None:
            GroupedLoopCounterTermTest.grouped_loop_sm = \
                models.import_model('loop_sm')
        self.model = GroupedLoopCounterTermTest.grouped_loop_sm
        self.model.actualize_dictionaries()
        self.assertEqual(self.model.get('merged_particles')[81], [1, 2, 3, 4])

    def test_grouped_gqq_counterterms_are_not_collapsed(self):
        """Each g q q~ R2/UV counterterm survives flavour merging.

        R2_GQQ and the UV counterterms share particles and orders.  Only the
        flavour partners of one counterterm may be merged, and the number of
        loop_particles entries (one contribution each) must not change."""

        found = sorted(
            (inter.get('type'),
             [coupling for coupling in inter.get('couplings').values()],
             inter.get('loop_particles'))
            for inter in self.model.get('interactions')
            if inter.get('type') != 'base' and sorted(
                abs(p.get_pdg_code()) for p in inter.get('particles')) ==
            [21, 81, 81])
        self.assertEqual(found, sorted([
            ('R2', ['R2_GQQ'], [[21, 81]]),
            ('UVloop1eps', ['UV_GQQb_1eps'], [[81], [81], [81]]),
            ('UVloop1eps', ['UV_GQQb_1eps'], [[81]]),
            ('UVloop', ['UV_GQQb'], [[5]]),
            ('UVloop1eps', ['UV_GQQb_1eps'], [[5]]),
            ('UVloop', ['UV_GQQt'], [[6]]),
            ('UVloop1eps', ['UV_GQQb_1eps'], [[6]]),
            ('UVloop1eps', ['UV_GQQg_1eps'], [[21]])]))

    def test_forbidden_uvloop_flavours_match_physical_counterterms(self):
        """Apply internal-flavour exclusions before CT loop keys are merged."""
        from collections import Counter

        def counterterms(model, physical_quark, forbidden):
            quark = 81 if model['merged_particles'] else physical_quark
            references = [copy.deepcopy(model.get(name))
                          for name in ('ref_dict_to0', 'ref_dict_to1')]
            process = base_objects.Process({
                'legs': base_objects.LegList([
                    base_objects.Leg({'id': pdg, 'number': number,
                                      'state': number > 2})
                    for number, pdg in enumerate((quark, -quark, 6, -6), 1)]),
                'model': model, 'orders': {'QED': 0},
                'perturbation_couplings': ['QCD'], 'squared_orders': {},
                'forbidden_particles': forbidden})
            amplitude = loop_diagram_generation.LoopAmplitude()
            amplitude.set('process', process)
            amplitude.generate_diagrams()
            self.assertEqual(references, [model.get(name)
                             for name in ('ref_dict_to0', 'ref_dict_to1')])
            self.assertNotIn('UVCT_SPECIAL', model['order_hierarchy'])
            self.assertTrue(all('UVCT_SPECIAL' not in inter['orders']
                                for inter in model.get('interaction_dict').values()))
            result = Counter()

            def add(interaction, multiplicity=1):
                flavor = tuple(physical_quark if abs(p.get_pdg_code()) == 81 else 0
                               for p in interaction['particles'])
                for coupling in interaction['couplings'].values():
                    if isinstance(coupling, base_objects.FLV_Coupling):
                        coupling = coupling['flavors'].get(flavor)
                    if coupling:
                        result[interaction['type'], coupling] += multiplicity

            for diagram in amplitude['loop_diagrams']:
                for vertex in diagram['CT_vertices']:
                    add(model.get_interaction(vertex['id']))
            for diagram in amplitude['loop_UVCT_diagrams']:
                for vertex in diagram['vertices']:
                    interaction = model.get_interaction(vertex['id'])
                    if interaction and interaction.is_UVtree():
                        add(interaction, diagram['UVCT_couplings'][0])
            return result

        for uv_type in ('UVloop', 'UVtree'):
            physical = models.import_model('loop_sm',
                options={'apply_flavor_grouping': False})
            if uv_type == 'UVtree':
                # Synthetic factorizing UVtree vertices exercise the same
                # physical loop-content multiplicities through Born generation.
                for interaction in physical['interactions']:
                    if interaction.is_UVloop():
                        interaction.set('type', interaction['type'].replace('UVloop', 'UVtree'))
            grouped = copy.deepcopy(physical)
            for original, duplicate in zip(physical['interactions'], grouped['interactions']):
                duplicate.set('color', original['color'])
            grouped.merge_flavor([1, 2, 3, 4])
            for model in (physical, grouped):
                model.actualize_dictionaries()
            for forbidden in ([3], [1, 3]):
                for quark in (q for q in (1, 2, 3, 4) if q not in forbidden):
                    with self.subTest(kind=uv_type, forbidden=forbidden, quark=quark):
                        self.assertEqual(counterterms(grouped, quark, forbidden),
                                         counterterms(physical, quark, forbidden))

    def test_grouped_closed_light_quark_loop(self):
        """A closed merged-quark loop sums its flavours and keeps its R2s.

        Ungrouped, the u/d/s/c loops of q q~ > t t~ are one identified
        diagram with multiplier 4 and four R2 counterterms; the physically
        expanded loop of the grouped model must be equivalent."""

        legs = base_objects.LegList([
            base_objects.Leg({'id': 81, 'state': False}),
            base_objects.Leg({'id': -81, 'state': False}),
            base_objects.Leg({'id': 6, 'state': True}),
            base_objects.Leg({'id': -6, 'state': True})])
        process = base_objects.Process({
            'legs': legs, 'model': self.model, 'orders': {'QED': 0},
            'perturbation_couplings': ['QCD'], 'squared_orders': {}})
        amplitude = loop_diagram_generation.LoopAmplitude()
        amplitude.set('process', process)
        amplitude.generate_diagrams()

        closed = [diag for diag in amplitude.get('loop_diagrams')
                   if set(abs(self.model.get_particle(tag[0]).get_pdg_code())
                          for tag in diag['canonical_tag']).issubset({1, 2, 3, 4})]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].get('multiplier'), 4)
        self.assertEqual(
            [self.model.get_interaction(ct.get('id')).get('couplings')
             for ct in closed[0].get('CT_vertices')],
            [{(0, 0): 'R2_GGq'}] * 4)
        self.assertEqual(sum(len(diag.get('CT_vertices'))
                             for diag in amplitude.get('loop_diagrams')), 27)

    def test_grouped_flavour_dependent_closed_loop_matches_physical(self):
        """Photon-coupled loops preserve physical multiplicities and CTs."""

        model = models.import_model(os.path.join(
            _input_file_path, 'LoopSMEWTest'))
        self.assertTrue(model.get('merged_particles'))
        legs = base_objects.LegList([
            base_objects.Leg({'id': 22, 'state': False}),
            base_objects.Leg({'id': 22, 'state': False}),
            base_objects.Leg({'id': 6, 'state': True}),
            base_objects.Leg({'id': -6, 'state': True})])
        process = base_objects.Process({
            'legs': legs, 'model': model, 'orders': {},
            'perturbation_couplings': ['QED'], 'squared_orders': {}})
        amplitude = loop_diagram_generation.LoopAmplitude()
        amplitude.set('process', process)
        amplitude.generate_diagrams()
        physical = copy.copy(process)
        physical.set('model', models.import_model(os.path.join(
            _input_file_path, 'LoopSMEWTest'),
            options={'apply_flavor_grouping': False}))
        reference = loop_diagram_generation.LoopAmplitude()
        reference.set('process', physical)
        reference.generate_diagrams()
        for key in ('born_diagrams', 'loop_diagrams'):
            self.assertEqual(len(amplitude[key]), len(reference[key]))
        self.assertEqual(sum(d['multiplier'] for d in amplitude['loop_diagrams']),
                         sum(d['multiplier'] for d in reference['loop_diagrams']))
        self.assertEqual(sum(len(d['CT_vertices']) for d in amplitude['loop_diagrams']),
                         sum(len(d['CT_vertices']) for d in reference['loop_diagrams']))


if __name__ == '__main__':
        # Save this model so that it can be loaded by other loop tests
        save_load_object.save_to_file(os.path.join(_input_file_path, 'test_toyLoopModel.pkl'),loadLoopModel())
        print("test_toyLoopModel.pkl created.")
        #unittest.main()
