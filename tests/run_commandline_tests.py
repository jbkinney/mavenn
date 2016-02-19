#!/usr/bin/env python
import subprocess
import re
from sortseq_tools import SortSeqError
import glob
import sys
import time

# Start clock
start_time = time.time()

# Commands must be execulted in cwd to work
cwd = 'input/'

# If files passed via commandline, use those.
if len(sys.argv)>1:
    filenames = sys.argv[1:]
# Otherwise, run on all files in commands/
else:
    filenames = glob.glob('commands/*.txt')

print 'Testing SortSeqTools on commands in the following files:'
print '-----'
print '\n'.join(filenames)
print '-----'

total_tests = 0
failed_tests = []

# Process each file
for filename in filenames:

    print 'Processing commands in %s...'%filename

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

        # Run command, get stdout and stderr
        p = subprocess.Popen(command,shell=True,cwd=cwd,\
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout_str, stderr_str = p.communicate()

        # Run checks on stdout and stderr
        prepend = '. '
        if test_type=='good' and stderr_str:
            prepend = 'E '
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



