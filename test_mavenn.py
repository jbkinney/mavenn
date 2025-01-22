#!/usr/bin/env python
# Tested 25.01.22 by JBK

# Insert mavenn at beginning of path
import os
import sys
abs_path_to_mavenn = os.path.dirname(os.path.abspath(__file__)) + './'
sys.path.insert(0, abs_path_to_mavenn)

#Load mavenn and check path
import mavenn
print(mavenn.__path__)

# Run tests
mavenn.run_tests()