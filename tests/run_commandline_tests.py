#!/usr/bin/env python
import subprocess
import re
from sst import SortSeqError
import glob
import sys

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

# Process each file
for filename in filenames:

    print 'Processing commands in %s...'%filename

    # Read lines. Strip whitespace
    with open(filename) as f:
        lines = [l.strip() for l in f.readlines()]

    # Evaluate each individual command
    for line in lines:

        # If emtpy or comment line, just print
        if not line or '#'==line[0]:
            print '\t'+line
            continue

        # Extract command
        try:
            groups = re.search('^\s*(\w+)\s*:\s*(.*)',line)
            test_type = groups.group(1)
            command = groups.group(2)
        except:
            raise SortSeqError('Could not interpret line: %s'%line)

        # Run command, get stdout and stderr
        p = subprocess.Popen(command,shell=True,cwd=cwd,\
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout_str, stderr_str = p.communicate()

        # Run checks on stdout and stderr
        if test_type=='good':
            if (not stdout_str) or stderr_str:
                print 'ERROR!',

        elif test_type=='bad':
            if not ('SortSeqError' in stderr_str):
                print 'ERROR!',
        else:
            print '\t'+line
            raise SortSeqError('Unrecognized test type %s'%test_type)

        print '\t'+line

    print '\tDone.'

