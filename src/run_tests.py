#!/usr/bin/env python

'''A script with generates Simulated Data for a Sort Seq Experiment 
    with a given mutation rate and wild type sequence.''' 
from __future__ import division
import os
import subprocess
import re
#from . import SortSeqError
from __init__ import SortSeqError
import glob
import sys
import time
from pkg_resources import resource_filename

# Define commandline wrapper
def wrapper(args):
    """ Commandline wrapper for main()
    """  
    

    # Start clock
    start_time = time.time()
    
    # Commands must be executed in cwd to work
    #command_file_path = os.path.abspath(resource_filename('mpathic_tests', 'test_fast.py'))

    command_file_path = os.path.abspath(resource_filename('MPAthic_tests', 'test_fast.py'))
    #print(command_file_path)

    command_path = os.path.dirname(command_file_path)
    print(command_path)
    cwd = os.path.join(command_path,'input')
    print(cwd)

    # If files passed via commandline, use those.
    #if len(sys.argv)>1:
    #     filenames = sys.argv[1:]
    # Otherwise, run on all files in commands/
    filenames_path = os.path.join(command_path,'commands','*.txt')

    filenames = glob.glob(filenames_path)

    # testing with only one file
    filenames = []
    filenames.append('/Users/tareen/Desktop/Desktop_Tests/MPathic/mpathic/MPAthic_tests/commands/evaluate_model_test.txt')

    print 'Testing SortSeqTools on commands in the following files:'
    print '-----'
    print '\n'.join(filenames)
    print '-----'

    total_tests = 0
    failed_tests = []

    # Process each file
    for filename in filenames:

        print 'Processing commands in %s...'%filename
        print 'Input files are drawn from %s' %cwd
        # Read lines. Strip whitespace
        with open(filename) as f:
            lines = [l.strip() for l in f.readlines()]

        # Evaluate each individual command
        validation = ''
        for line in lines:

            # If emtpy or comment line, just print
            if not line or '#'==line[0]:
                print '  '+line
                continue

            # Extract command
            try:
                groups = re.search('^\s*(\w+)\s*:\s*(.*)',line)
                test_type = groups.group(1)
                command = groups.group(2)
            except:
                raise SortSeqError('Could not interpret line: %s'%line)

            # If specifies validation, record and move to next line
            if test_type=='validation':
                print '> '+line
                validation = command
                continue

            # Append validation routine to command only if good outcome expected
            if test_type=='good':
                command += ' ' + validation
                line += ' ' + validation
            #edit command so that all input files are located in input directory
            editted_command = command.replace('-m ','-m %s' %(cwd + '/'))
            editted_command = editted_command.replace('-i ','-i %s' %(cwd + '/'))
            editted_command = editted_command.replace('-ds ','-ds %s' %(cwd + '/'))
            editted_command = editted_command.replace('cat ','cat %s' %(cwd + '/'))
            editted_command = editted_command.replace('--LS_means_std ','--LS_means_std %s' %(cwd + '/'))
            editted_command = editted_command.replace('--tagkeys ','--tagkeys %s' %(cwd + '/'))
            # Run command, get stdout and stderr

            p = subprocess.Popen(editted_command,shell=True,\
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stdout_str, stderr_str = p.communicate()

            # Run checks on stdout and stderr
            prepend = '. '
            if test_type=='good' and stderr_str:
                prepend = 'E '
                print stderr_str
                failed_tests.append(line)

            elif test_type=='bad' and not ('SortSeqError' in stderr_str):
                prepend = 'E '
                failed_tests.append(line)
        
            if not test_type in ('good','bad'):
                raise SortSeqError('Unrecognized test type %s'%repr(test_type))

            print prepend + line
            total_tests += 1

        print '  Done with %s.\n'%filename

    # Stop clock
    testing_time = time.time() - start_time
    print '------------FAILED TESTS-------------'
    print '\n'.join(failed_tests)
    print '-------------------------------------'
    print 'Time to run tests: %.2f min.'%(testing_time/60.0)
    print 'Results: %d tests, %d failures.'%(total_tests,len(failed_tests))
    print '-------------------------------------'

# Connects argparse to wrapper
def add_subparser(subparsers):
    p = subparsers.add_parser('run_tests')
    p.set_defaults(func=wrapper)
    return p
