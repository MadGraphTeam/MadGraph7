##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
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
from __future__ import absolute_import
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph

import madgraph.interface.master_interface as master_cmd
import madgraph.interface.madgraph_interface  as mg_cmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as mecmd
import madgraph.interface.amcatnlo_run_interface as amcatnlo_run
import madgraph.various.misc as misc
import os
import re
import shutil
import tempfile
from io import StringIO
import logging

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class MGerror(Exception): pass

class TestMadEventCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
        
    def test_v31_syntax_crash(self):
        """Check that process with ambiguous syntax correctly crashes if the flag is not set correctly
        """
        #cmd = mecmd.aMCatNLOInterface()
        #category = set()
        #valid_command = [c for c in dir(cmd) if c.startswith('do_')]

        interface = master_cmd.MasterCmd()
        interface.no_notification()
        interface.do_import('model loop_qcd_qed_sm')
        
        #run_cmd('import model %s' % model)

        stream = StringIO()
        handler_stream = logging.StreamHandler(stream)
        log = logging.getLogger('cmdprint')
        log.setLevel(logging.CRITICAL)

        for handler in log.handlers: 
            log.removeHandler(handler)
        log.addHandler(handler_stream)


        def check_message(line):
            """return False if not warning is raised, return True is a warning is raised"""
            
            stream.seek(0)
            stream.truncate(0)
            myprocdef = interface.extract_process(line)
            try: 
                interface.proc_validity(myprocdef,'aMCatNLO_all')
            except Exception as error:
                if '1804.10017' in str(error):
                    raise MGerror

            text = stream.getvalue()
            if '1804.10017' in text:
                return True
            else:
                return False

        # force the option to not by bypassed
        interface.options['acknowledged_v3.1_syntax'] = False

        # check case where the code crash
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=1 [QED]")
        self.assertRaises(MGerror,check_message, "p p > t t~ QCD=1 [QED QCD]")

        # check case where the code write a warning (critical level)
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=1 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QCD=1 QED=0 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=98 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=99 QCD=1 [QCD]")
        # check case where the code does not complain
        self.assertFalse(check_message("p p > t t~ QED=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ / z QCD=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QCD=1 QED^2==2 [QCD]"))
        self.assertFalse(check_message("p p > w+ w- QED=99 [QCD]"))
        self.assertFalse(check_message("p p > w+ w- j j $ h QED<=99 [QCD]"))

        # force the option to not  be by bypassed
        interface.options['acknowledged_v3.1_syntax'] = True 
        # and check that no crash/warning is raised anymore
        self.assertFalse(check_message( "p p > t t~ QED=1 [QED]"))
        self.assertFalse(check_message( "p p > t t~ QCD=1 [QED QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=1 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QCD=1 QED=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=98 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=99 QCD=1 [QCD]"))


class TestNoFileOutsideOutputDir(unittest.TestCase):
    """Running NLO from the prompt must not write in the directory MG was
    started from: only the output directory belongs to MG."""

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        self.startdir = pjoin(self.tmpdir, 'start')
        os.mkdir(self.startdir)
        os.chdir(self.startdir)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def bare_cmd():
        """an aMCatNLOCmd with just what run_exe needs"""
        cmd = amcatnlo_run.aMCatNLOCmd.__new__(amcatnlo_run.aMCatNLOCmd)
        # no RunWeb to remove when the bare instance is deleted
        cmd.stop_for_runweb = True
        cmd.cluster_mode = 2
        return cmd

    def test_missing_executable_names_job_directory(self):
        """The error names the directory where the executable was looked
        for, not the one MG was started from."""

        jobdir = pjoin(self.tmpdir, 'SubProcesses', 'P0_gg_ttx', 'GF1')
        os.makedirs(jobdir)
        self.assertRaisesRegex(amcatnlo_run.aMCatNLOError,
            'Cannot find executable reweight_xsec_events.local in %s$'
                                                     % re.escape(jobdir),
            self.bare_cmd().run_exe, 'reweight_xsec_events.local',
            ['events.lhe', '1'], 'Running reweight', cwd=jobdir)

    def test_reweight_job_files_in_job_directory(self):
        """The files of a reweight job (multicore/cluster) are relative to the
        job directory, so they must be created there."""

        class FakeCluster(object):
            def submit2(self, prog, argument=[], cwd=None, **opts):
                self.submitted = dict(opts, prog=prog, cwd=cwd)

        jobdir = pjoin(self.tmpdir, 'SubProcesses', 'P0_gg_ttx', 'GF1')
        os.makedirs(jobdir)
        open(pjoin(jobdir, 'reweight_xsec_events.local'), 'w').close()
        # complete output of a previous reweighting of this directory
        with open(pjoin(jobdir, 'events.lhe.rwgt'), 'w') as stale:
            stale.write('</LesHouchesEvents>\n')

        cmd = self.bare_cmd()
        cmd.cluster = FakeCluster()
        cmd.get_pdf_input_filename = lambda: ''
        cmd.run_exe('reweight_xsec_events.local', ['events.lhe', '1'],
                    'Running reweight', cwd=jobdir)

        self.assertEqual(os.listdir(self.startdir), [])
        job_files = ['events.lhe.rwgt', 'reweight_xsec_events.output',
                     'scale_pdf_dependence.dat']
        for name in job_files:
            # present (for the transfer to the node) but empty: a job that
            # fails cannot leave the stale events in place
            self.assertEqual(os.path.getsize(pjoin(jobdir, name)), 0, name)
        submitted = cmd.cluster.submitted
        self.assertEqual(submitted['cwd'], jobdir)
        # copied back from the node into cwd, so relative to it
        self.assertEqual(submitted['output_files'], job_files)
        self.assertEqual(submitted['required_output'], job_files)

    def test_switch_interface_leaves_no_file(self):
        """Switching to the NLO interface checks the command dispatch and
        must only write 'additional_command' when a command is missing."""

        interface = master_cmd.MasterCmd()
        interface.no_notification()
        interface.change_principal_cmd('aMC@NLO')
        interface.change_principal_cmd('MadGraph')

        self.assertEqual(os.listdir(self.startdir), [])
