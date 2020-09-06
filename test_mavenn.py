#!/usr/bin/env python

# Insert mavenn at beginning of path
import sys
path_to_mavenn_local = '.'
sys.path.insert(0, path_to_mavenn_local)

#Load mavenn and check path
import mavenn
print(mavenn.__path__)

# Run tests
mavenn.run_tests()