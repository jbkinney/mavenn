#!/usr/bin/env python
import subprocess
import re
from sst import SortSeqError
import glob

cwd = 'input/'

for filename in glob.glob('commands/*.txt'):

    print 'Processing commands in %s...'%filename

    # Read lines. Strip whitespace
    with open(filename) as f:
        lines = [l.strip() for l in f.readlines()]

    # Evaluate each individual command
    for line in lines:

        # If emtpy or comment line, just print
        print '\t',
        if not line or '#'==line[0]:
            print line
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
                print 'ERROR! ',
            print '(good) ',

        elif test_type=='bad':
            if not ('SortSeqError' in stderr_str):
                print 'ERROR! ',
            print '(bad)  ',
        else:
            print command
            raise SortSeqError('Unrecognized test type %s'%test_type)
        print command
    print '\tDone.'

