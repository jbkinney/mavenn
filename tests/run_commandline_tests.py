#!/usr/bin/env python
import subprocess
import re
from sst import SortSeqError

cwd = 'input/'

# Read in commands; remove empty lines
with open('command_list.txt') as f:
    lines = filter(None,[l.strip() for l in f.readlines()])

# Evaluate each individual command
for line in lines:

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
        if stdout_str and (not stderr_str):
            print '(OK)    '+line
        else:
            print '(ERROR) '+line
    elif test_type=='bad':
        if (not stdout_str) and ('SortSeqError' in stderr_str):
            print '(OK)    '+line
        else:
            print '(ERROR) '+line
    else:
        raise SortSeqError('Unrecognized')

